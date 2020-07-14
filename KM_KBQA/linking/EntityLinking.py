import logging
import os
import re
from functools import lru_cache

from fuzzywuzzy import fuzz, process

from ..BertEntityRelationClassification.BertERClsPredict import \
    predict as BertERCls
from ..common import AsyncNeoDriver
from ..common.HITBert import cosine_word_similarity
from ..config import config
from .LinkUtil import retrieve_mention

logger = logging.getLogger('qa')
exception_subgenre = {'临时身份证办理'}


def contain_chinese(s):
    s = s.replace('-', '').lower()
    if s in {'wifi', 'atm', 'vip', 'kfc', 'ktv'}:
        return True
    for c in s:
        if ('\u4e00' <= c <= '\u9fa5'):
            return True
    return False


def contain_english(s):
    return bool(re.search('[A-Za-z]', s))


def load_ent_alias(fname):
    ent2alias = {}
    if os.path.isfile(fname):
        logger.info('load entity alias file %s' % fname)
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                ent, alias = line.split(':')
                alias = [a.strip() for a in alias.split(',')]
                ent2alias[ent] = alias
    return ent2alias


def make_mention2ent(ent2alias):
    mention2ent = {}
    for ent, alias in ent2alias.items():
        for mention in alias:
            if mention in mention2ent:
                # logger.warning('指称映射冲突：%s -> [%s, %s]' %
                            #    (mention, mention2ent[mention], ent))
                mention2ent[mention].append(ent)
            else:
                mention2ent[mention] = [ent]
    return mention2ent


class RuleLinker():
    def __init__(self, driver=None):
        if driver is None:
            self.driver = AsyncNeoDriver.get_driver(name='default')
        else:
            self.driver = driver
        self.load_all_entities()
        ent2alias = load_ent_alias(config.ENT_ALIAS_PATH)
        self.mention2ent = make_mention2ent(ent2alias)

    def is_special_entity(self, ent_dict):
        return ent_dict['entity_label'] == 'Genre' \
            and (
            '类' in ent_dict['name']
            or '航空公司' in ent_dict['name']
            or '行李安检' in ent_dict['name']
        )

    def load_all_entities(self, entity_labels=['Instance', 'SubGenre', 'Genre']):
        all_entities = []
        for entity_label in entity_labels:
            tmp_entities = self.driver.get_all_entities(entity_label).result()
            for e in tmp_entities:
                e['entity_label'] = entity_label
            tmp_entities = [e for e in tmp_entities
                            if not self.is_special_entity(e)]
            all_entities += tmp_entities
        self.id2ent = {x['neoId']: x for x in all_entities}
        self.ent_names = {x['name'] for x in all_entities}

    # @lru_cache(maxsize=128)
    def link(self, sent, sent_cut, pos_tag, limits=None):
        # use bert embedding to fuzzy match entities
        # mention_list = recognize_entity(sent)
        mention_list = retrieve_mention(sent_cut, pos_tag)
        if mention_list == []:
            return []
        logger.debug('指称: ' + str(mention_list))
        # self.sent_cut = LTP.customed_jieba_cut(sent, cut_stop=True)
        # print('cut:', self.cut)
        res = []
        for mention in mention_list:
            mention = mention.lower()
            one_res = []
            if self.is_not_entity(mention):
                continue
            # cand_name = self.convert_abstract_verb(
            #     mention, sent, limits)
            cand_names = self.convert_mention2ent(mention)
            for ent in self.id2ent.values():
                # for ent_name in self.ent_names:
                ent_name = ent['name']
                ent_name_rewrite = self.rewrite_ent_name(ent_name)
                if ent_name_rewrite == '':
                    continue
                for cand_name in cand_names:
                    # 该实体为英文而问的有汉语或相反
                    if contain_chinese(cand_name) and not contain_chinese(ent_name) or contain_english(
                            cand_name) and not contain_english(ent_name):
                        continue
                    RATIO = 0.5
                    score = cosine_word_similarity(cand_name, ent_name_rewrite)
                    score1 = fuzz.UQRatio(cand_name, ent_name_rewrite)/100
                    score = RATIO*score + (1-RATIO) * score1
                    one_res.append({
                        'ent': ent,
                        'mention': mention,
                        'id': ent['neoId'],
                        'score': score,
                        'source': 'rule'
                    })
            one_res.sort(key=lambda x: x['score'], reverse=True)
            for a_res in one_res[:3]:
                if a_res['score'] > config.simi_ths:
                    res.append(a_res)
        res.sort(key=lambda x: x['score'], reverse=True)
        return res

    def convert_abstract_verb(self, word, sent, limits):
        convert_dict = config.ABSTRACT_DICT
        if word in convert_dict:
            # TODO ** 和词典严重耦合的硬编码 **
            if word == '换':
                if limits is not None and limits['币种'] != '' or '货币' in sent or '外币' in sent:
                    return convert_dict.get(word)[0]
                elif '尿' in sent:
                    return convert_dict.get(word)[1]
                else:
                    return word
            # if wd == '换':
            #     if '货币' or '外币'
            return convert_dict[word]
        else:
            # 去除"服务"字段的影响
            return word.replace('服务', '')

    def convert_mention2ent(self, mention) -> list:
        ent_names = self.mention2ent.get(mention, None)
        if ent_names is not None:
            return ent_names
        return [mention.replace('服务', '')]

    def is_not_entity(self, item):
        for wd in config.airport.filter_words:
            if wd in item:
                return True
        for wd in config.airport.remove_words:
            if wd == item:
                return True
        # or not is_contain_chinese(item):
        if re.search(r'(时间|地点|位置|地方|收费|价格|限制|电话)', item) is not None\
                and item not in self.mention2ent:
            return True
        return False

    def rewrite_ent_name(self, ent_name):
        ent_name = ent_name.split('(')[0].split('（')[0].lower()
        if '服务' in ent_name and ent_name != '服务':
            ent_name = ent_name.replace('服务', '')
        if ent_name in {'柜台', '其他柜台', '行李', '咨询'}:
            ent_name = ''
        return ent_name


class BertLinker():
    def __init__(self, driver=None):
        if driver is None:
            self.driver = AsyncNeoDriver.get_driver(name='default')
        else:
            self.driver = driver

    def link(self, sent, sent_cut=None, pos_tag=None):
        _, _, ent_type_top3 = BertERCls(sent)
        # print(ent_type_top3)
        instances_top3 = [self.driver.get_instance_of_genre(ent_type, genre='SubGenre') +
                          (self.driver.get_entities_by_name(ent_type).result()
                           if ent_type in exception_subgenre else [])
                          for ent_type in ent_type_top3]
        res = []
        for rank, instances in enumerate(instances_top3):
            for e in instances:
                ent = {
                    'ent': e,
                    'id': e['neoId'],
                    'mention': e['name'],
                    'rank': rank+1,
                    'source': 'bert',
                    'score': 1/(rank+1) - 0.05
                }
                res.append(ent)
        return res


class CommercialLinker():
    def __init__(self, driver=None):
        if driver is None:
            self.driver = AsyncNeoDriver.get_driver(name='default')
        else:
            self.driver = driver
        self.content2entId = self.build_revert_index()
        self.ban_word = {'机场'}

    def link(self, sent, sent_cut, pos_tag=None):
        id2ent = {}
        for word in sent_cut:
            if word in self.ban_word:
                continue
            content_keys = self.retrieve_content_keys(word)
            for content, score in content_keys:
                ent_ids = self.content2entId[content]
                score /= 100
                if score > 0.5:
                    for ent_id in ent_ids:
                        e = self.driver.get_entity_by_id(ent_id).result()[0]
                        if '服务内容' in e:
                            e.pop('服务内容')
                        if ent_id in id2ent:
                            old_score = id2ent[ent_id]['score']
                            id2ent[ent_id]['score'] = max(score, old_score)
                        else:
                            ent = {
                                'ent': e,
                                'mention': word,
                                'id': ent_id,
                                'score': score,
                                'source': 'commercial',
                                'content': content
                            }
                            id2ent[ent_id] = ent
        res = list(id2ent.values())
        if '买' in sent or '卖' in sent or '吃' in sent:
            for ent in res:
                ent['score'] += 0.3
        return res

    def build_revert_index(self):
        entities = self.driver.get_entities_by_genre('Instance').result()
        # entities += self.driver.get_entities_by_genre('SubGenre').result()
        content2entId = {}
        for ent in entities:
            ent_id = ent['neoId']
            content_str = ent.get('服务内容', '').replace('服务', '')
            content = content_str.split(';')
            for c in content:
                if c == '':
                    continue
                if c in content2entId:
                    content2entId[c].append(ent_id)
                else:
                    content2entId[c] = [ent_id]
        return content2entId

    def retrieve_content_keys(self, sent):
        # words = LTP.customed_jieba_cut(sent)
        # sent = ''.join(words).replace('服务', '')
        sent = sent.replace('服务', '')
        if sent == '':
            return []
        res = process.extract(sent, self.content2entId.keys(),
                              scorer=fuzz.UQRatio,
                              limit=2)

        return res


def test_bert_linker():
    bert_linker = BertLinker()
    sent = '有可以玩游戏的地方吗？'
    # sent = '东航的值机柜台在哪？'
    res = bert_linker.link(sent)
    print(res)


def test_commercial_linker():
    commercial_linker = CommercialLinker()
    # print(commercial_linker.content2entId)
    sent = '过了安检里面有没有书吧？'
    # sent = '东航的值机柜台在哪？'
    res = commercial_linker.link(sent)
    print(res)


if __name__ == '__main__':
    # test_bert_linker()
    test_commercial_linker()
    # test_rule_linker()
