import logging
import pdb
import re

import jieba

from ..BertEntityRelationClassification.BertERClsPredict import \
    predict as BertERCls
from ..common.AsyncNeoDriver import get_driver
from ..common.HITBert import cosine_word_similarity
from ..config import config


class AbstractRelExtractor():
    def __init__(self):
        raise NotImplementedError

    def extract_rel(self, sent, ent=None):
        raise NotImplementedError


class BertRelExtractor(AbstractRelExtractor):
    def __init__(self):
        pass

    def extract_rel(self, sent, linked_ent):
        _, rel, _ = BertERCls(sent)
        ent = linked_ent['ent']
        res = []
        if rel in ent:
            res.append({
                'id': ent['neoId'],
                'entity': ent['name'],
                'rel_name': rel,
                'rel_val': ent[rel],
                'link_score': linked_ent['score'],
                'rel_score': 1
            })
        return res


class MatchRelExtractor(AbstractRelExtractor):
    def __init__(self, driver=None):
        if driver is None:
            self.driver = get_driver()
        else:
            self.driver = driver
        self.remove_prop = {'subname', 'description',
                            'label', 'taglist', 'neoId', 'keyId', 'id', 'score', 'rel', 'hidden'}

    def normalize_prop(self, prop):
        if '地点' in prop:
            return '地点'
        if '时间' in prop:
            return '时间'
        return prop

    def normalize_ratio(self, word, prop):
        ratio = 1.0
        if '时间' in prop and '时间' in word or '时间' in prop and '几点' in word:
            ratio = 1.5
        if '地址' in word and '地点' in prop or '地点' in prop and '地方' in word or '地点' in prop and '几楼' in word or '怎么走' in word and '地点' in prop or '位置' in word and '地点' in prop:
            ratio = 1.5
        if '联系' in prop and '电话' in word or '电话' in prop and '联系' in word:
            ratio = 1.5
        money = ['价格', '费', '钱']
        for p_i in money:
            for w_i in money:
                if p_i in prop and w_i in word:
                    ratio = 1.5
        return ratio



    def extract_rel(self, sent_cut,
                    linked_ent,
                    limits=None,
                    thresh=config.prop_ths):
        ent = linked_ent['ent']
        mention = linked_ent.get('mention', ent['name'])
        # extract all prop， 限制支持一个
        props_dict = {}

        for prop, value in ent.items():
            if prop not in self.remove_prop:
                props_dict[prop] = str(value)

        # 计算满足限制
        '''
        try:
            res_limit = self.cal_limit(limits, props_dict)
        except:
            pdb.set_trace()

        wrong_restriction = ''
        accepted_limit = {}
        if res_limit is None:
            return None
        for limit in res_limit.keys():
            if not res_limit[limit]:
                wrong_restriction += ', ' + limit + ' 限制错误'
            else:
                accepted_limit[limit] = props_dict[limit]

        # cut
        limit_list = list(map(lambda x: x[1], list(limits.items())))
        rest_words = list(filter(
            lambda x: x not in cand_name and '机场' not in x and x not in limit_list, cut_words))
        '''
        rest_words = [
            w for w in sent_cut if w not in mention and '机场' not in w]
        props_set = list(props_dict.keys())
        props_set.remove('name')
        # cal prop rel similarity
        res = []
        used_pairs = set()

        for prop in props_set:
            for word in rest_words:
                score = cosine_word_similarity(word, self.normalize_prop(prop))
                ratio = self.normalize_ratio(word, prop)
                score = ratio if ratio > 1 else score
                if word in prop and len(word) > 1:
                    score *= 1.2
                if score > thresh and (word, prop) not in used_pairs:
                    used_pairs.add((word, prop))
                    # res.append([neoId, cand_name, ent_name, {
                    #    prop: props_dict[prop]}, accepted_limit, score, ent_score])
                    res.append({
                        'id': ent['neoId'],
                        'mention': mention,
                        'entity': ent['name'],
                        'rel_name': prop,
                        'rel_val': props_dict[prop],
                        'link_score': linked_ent['score'],
                        'rel_score': score
                    })
        if len(res) == 0:
            return None
        res.sort(key=lambda x: x['rel_score'], reverse=True)
        res_lang = []
        # for item in res:
        #     rel = list(item[3].keys())[0]
        #     val = item[3][rel]
        #     ans = item[2] + '的' + rel + '是' + val
        #     if wrong_restriction != '':
        #         ans += ' ' + wrong_restriction
        #     res_lang.append([ans] + item)
        # 如果前两个属性都很高，那返回两个答案
        sel_num = 1
        if len(res) > 1 and res[1]['rel_score'] > 0.91:
            sel_num += 1
        # sel_num = 2 if res_lang[0][res] > 0.91 and res_lang[0][6] > 0.91 else 1
        # return res_lang[:min(len(res), sel_num)]
        return res[:sel_num]
