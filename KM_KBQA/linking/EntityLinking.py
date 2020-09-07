import logging
import os
import re, time
from functools import lru_cache

from fuzzywuzzy import fuzz, process

from ..BertEntityRelationClassification.BertERClsPredict import \
    predict as BertERCls
from ..common import AsyncNeoDriver
from ..common.HITBert import cosine_word_similarity
from ..config import config
from .LinkUtil import retrieve_mention

logger = logging.getLogger('qa')


# exception_subgenre = {'临时身份证办理'}


def contain_chinese(s):
    s = s.replace('-', '').lower()
    # if s in {'wifi', 'atm', 'vip', 'kfc', 'ktv'}:
    #     return True
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
                ent, alias = line.split(': ')
                alias = [a.strip() for a in alias.split(',')]
                ent2alias[ent] = alias
    return ent2alias


def make_mention2ent(ent2alias):
    mention2ent = {}
    for ent, alias in ent2alias.items():
        for mention in alias:
            if mention in mention2ent:
                mention2ent[mention].append(ent)
            else:
                mention2ent[mention] = [ent]
    return mention2ent


#  获取所有实体的所有名称，包括所有别名 zsh
def get_entity_all_names(all_entities):
    all_entities_tmp = []
    for x in all_entities:
        names = x['名称'] if '名称' in x else x['name']
        if "[" in names:  # list 类型的名称，说明有别名
            names = eval(names.split(";;;")[0])
            all_entities_tmp += [name for name in names]
        else:
            all_entities_tmp.append(names.split(";;;")[0])
    return all_entities_tmp


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
                   # or '航空公司' in ent_dict['name']
                   # or '行李安检' in ent_dict['name']
               )

    def load_all_entities(self, entity_labels=['Instance', 'SubGenre', 'SubGenre_child', 'Genre']):
        all_entities = []
        for entity_label in entity_labels:
            tmp_entities = self.driver.get_all_entities(entity_label).result()
            for e in tmp_entities:
                e['entity_label'] = entity_label
            tmp_entities = [e for e in tmp_entities
                            if not self.is_special_entity(e)]
            all_entities += tmp_entities
        self.id2ent = {x['neoId']: x for x in all_entities}
        self.ent_names = get_entity_all_names(all_entities)

        # @lru_cache(maxsize=128)

    def link(self, sent, sent_cut, pos_tag, limits=None):
        # use bert embedding to fuzzy match entities
        # mention_list = recognize_entity(sent)
        mention_list = retrieve_mention(sent_cut, pos_tag)
        if mention_list == []:
            return []
        is_list = False
        if '哪些' in mention_list or '哪几个' in mention_list :
            is_list = True
        logger.debug('指称: ' + str(mention_list))
        res = []
        country_list = ['俄罗斯','挪威','美国','蒙古','泰国', '韩国']
        if is_list:
            if all([word in mention_list for word in ['中国', '航空公司']]):
                for ent in self.id2ent.values():
                    if '类别' not in ent:
                        continue
                    if ent['类别'] == '国内航空公司':
                        res.append({
                            'ent': ent,
                            'mention': ''.join(['中国', '航空公司']),
                            'rel_name': '类别',
                            'rel_val': '中国航空公司',
                            'id': ent['neoId'],
                            'score': 1.5,
                            'source': 'rule'})
            elif any([word in mention_list for word in country_list]) and '航空公司' in mention_list:
                for c in country_list:
                    if c in mention_list:
                        country = c
                        continue
                word = [word for word in mention_list if word in country_list][0]
                for ent in self.id2ent.values():
                    flag = False
                    if '类别' not in ent:
                        continue
                    if ent['类别'] == '国外航空公司':
                        if '别名' in ent and any([word in name for name in eval(ent['别名'])]):
                            flag = True
                        if '公司名称' in ent and word in ent['公司名称']:
                            flag = True
                        if flag:
                            res.append({
                                'ent': ent,
                                'mention': '国外航空公司',
                                'rel_name': '类别',
                                'rel_val': country+'航空公司',
                                'id': ent['neoId'],
                                'score': 1.5,
                                'source': 'rule'})
            if (len(res) != 0):
                return res
                
        for mention in mention_list:
            mention = mention.lower()
            one_res = []
            if not contain_chinese(mention):
                search_list = []
                if '机场' in mention_list:
                    search_list = ['机场三字码', 'ICAO机场代码']
                elif '航空公司' in mention_list or '航空' in mention_list:
                    search_list = ['IATA代码', 'ICAO代码']
                for ent in self.id2ent.values():
                    if (len(search_list) == 0) or (not any([key in ent for key in search_list])):
                        continue
                    ent_iata = ''
                    ent_icao = ''
                    ent_three = ''
                    ent_icao_a = ''
                    if 'IATA代码' in ent:
                        ent_iata = ent['IATA代码']
                    if 'ICAO代码' in ent:
                        ent_icao = ent['ICAO代码']
                    if '机场三字码' in ent:
                        ent_three = ent['机场三字码']
                    if 'ICAO机场代码' in ent:
                        ent_icao_a = ent['ICAO机场代码']
                    if mention.upper() == ent_iata or mention.upper() == ent_icao or \
                        mention.upper() == ent_three or mention.upper() == ent_icao_a:
                        res.append({
                            'ent': ent,
                            'mention': mention,
                            'id': ent['neoId'],
                            'score': 2.5,
                            'source': 'rule'})
                # continue  # 存在实体名称也是英文，所以不能continue
            if self.is_not_entity(mention):
                continue
            # cand_name = self.convert_abstract_verb(
            #     mention, sent, limits)
            cand_names = self.convert_mention2ent(mention)  # entity别名设置
            for ent in self.id2ent.values():
                # for ent_name in self.ent_names:
                if 'name' not in ent:
                    continue
                if '机场' not in mention and (ent['类别'] == '国外机场' or ent['类别'] == '国内机场'):
                    continue
                ent_name = ent['name']
                ent_name_rewrite = self.rewrite_ent_name(ent_name)
                if ent_name_rewrite == '':
                    continue
                for cand_name in cand_names:
                    # 该实体为英文而问的有汉语或相反
                    # 原因：当时用bert encode以后进行相似度匹配的时候，输入的实体或者图谱中的实体有英文的话有时语义不相近bert也会给出较高的值，所以过滤掉只有一方出现英文的情况
                    if contain_chinese(cand_name) and not contain_chinese(ent_name) or contain_english(
                        cand_name) and not contain_english(ent_name):
                        continue

                    RATIO = 0.5
                    score = cosine_word_similarity(cand_name, ent_name_rewrite)
                    score1 = fuzz.UQRatio(cand_name, ent_name_rewrite) / 100
                    score = RATIO * score + (1 - RATIO) * score1
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
            if not isinstance(ent_names, list):
                ent_names = [ent_names]
            return ent_names
        return [mention.replace('服务', '')]

    def is_not_entity(self, item):
        if re.search(r'(时间|地点|位置|地方|收费|价格|限制|电话)', item) is not None \
            and item not in self.mention2ent:
            return True

    def rewrite_ent_name(self, ent_name):
        ent_name = ent_name.split('(')[0].split('（')[0].lower()  # 中英文括号前面的部分
        # if '服务' in ent_name and ent_name != '服务':  # zsh
        #     ent_name = ent_name.replace('服务', '')
        # if ent_name in {'柜台', '其他柜台', '行李', '咨询'}:
        #     ent_name = ''
        return ent_name


class BertLinker():
    def __init__(self, driver=None):
        if driver is None:
            self.driver = AsyncNeoDriver.get_driver(name='default')
        else:
            self.driver = driver

    def link(self, sent, sent_cut=None, pos_tag=None):
        _, _, ent_type_top3 = BertERCls(sent)  # get input_ids, input_mask and segment_ids from sent
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
                    'rank': rank + 1,
                    'source': 'bert',
                    'score': 1 / (rank + 1) - 0.05
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