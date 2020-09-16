from fuzzywuzzy import fuzz

from ..BertEntityRelationClassification.BertERClsPredict import \
    predict as BertERCls
from ..common.AsyncNeoDriver import get_driver
from ..common.HITBert import cosine_word_similarity
from ..config import config


def contain_chinese(s):
    s = s.replace('-', '').lower()
    # if s in {'wifi', 'atm', 'vip', 'kfc', 'ktv'}:
    #     return True
    for c in s:
        if ('\u4e00' <= c <= '\u9fa5'):
            return True
    return False

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
                'mention': ent['name'],
                'entity': ent['name'],
                'rel_name': rel,
                'rel_val': ent[rel],
                'link_score': linked_ent['score'],
                'rel_score': 1,
                'rel_source': 'bert'
            })
        return res


class MatchRelExtractor(AbstractRelExtractor):
    def __init__(self, driver=None):
        if driver is None:
            self.driver = get_driver()
        else:
            self.driver = driver
        self.remove_prop = {'subname', 'description','label', 'taglist', 'neoId', 'keyId', 'id', 'score', 'rel', 'hidden', 'entity_label',
                            '一级类型','二级类型'}

    def normalize_prop(self, prop):
        if '地点' in prop:
            return '地点'
        if '时间' in prop:
            return '时间'
        return prop

    def normalize_ratio(self, word, prop):
        ratio = 1.0
        # if '时间' in prop and '时间' in word or '时间' in prop and '几点' in word:
        #     ratio = 1.5

        # if '地址' in word and '地点' in prop or '地点' in prop and '地方' in word or '地点' in prop and '几楼' in word or '怎么走' in word and '地点' in prop or '位置' in word and '地点' in prop:
        place_words = ['地址', '地点', '总部', '位置']
        place_flag = False
        for w in place_words:
            if w in word:
                place_flag = True
                break
        if place_flag:  # '地点' in prop and
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
        mention = linked_ent.get('mention', ent['name'])  # 链接结果中该实体的指代名称
        props_dict = {}

        for prop, value in ent.items():
            # 去除不需要的属性，如id等
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
            w for w in sent_cut if w not in mention]  # 除了指代以外的其他分词（在这些词中找关系）
        props_set = list(props_dict.keys())

        props_set.remove('name')
        if '名称' in props_set:
            props_set.remove('名称')

        # cal prop rel similarity
        res = []
        used_pairs = set()
        for prop in props_set:
            old_prop = prop
            for word in rest_words:
                # prop = prop.replace('服务', '')
                cos_score = cosine_word_similarity(
                    word, self.normalize_prop(prop))
                text_score = fuzz.UQRatio(word, prop)/100
                ratio = 0.6
                score = ratio*cos_score + (1-ratio)*text_score
                # rule_score = self.normalize_ratio(word, prop)  # 暂停用规则抽取得分
                # score = rule_score if rule_score > 1 else score
                if word in prop and len(word) > 1:
                    if word not in ['定义']:
                        score *= 1.2
                if score > thresh and (word, prop) not in used_pairs:
                    used_pairs.add((word, prop))
                    # res.append([neoId, cand_name, ent_name, {
                    #    prop: props_dict[prop]}, accepted_limit, score, ent_score])
                    res.append({
                        'id': ent['neoId'],
                        'mention': mention,
                        'entity': ent['name'],
                        'rel_name': old_prop,
                        'rel_val': props_dict[old_prop],
                        'link_score': linked_ent['score'],
                        'rel_score': score,
                        'rel_source': 'match'
                    })
        if len(res) == 0:
            return []
        res.sort(key=lambda x: x['rel_score'], reverse=True)
        # res_lang = []
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
