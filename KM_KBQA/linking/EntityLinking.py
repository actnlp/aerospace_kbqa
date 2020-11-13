import logging
import os,json,sets
import re, time
from functools import lru_cache

from fuzzywuzzy import fuzz, process

from ..BertEntityRelationClassification.BertERClsPredict import \
    predict as BertERCls
from ..common import AsyncNeoDriver
from ..common.HITBert import cosine_word_similarity,encode,all_entity_encode
from ..config import config
from .LinkUtil import retrieve_mention


logger = logging.getLogger('qa')


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
            mention = mention.lower()
            if mention in mention2ent:
                mention2ent[mention].append(ent.lower())
            else:
                mention2ent[mention] = [ent.lower()]
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
        self.load_all_entities()  # entity_labels=['Instance']
        self.alias2manufacturers = self.load_manufacturer_alias()
        ent2alias = load_ent_alias(config.ENT_ALIAS_PATH)
        self.mention2ent = make_mention2ent(ent2alias)

    #  获取飞机制造公司的所有名称，包括所有别名
    def load_manufacturer_alias(self):
        manufacturers_info = self.driver.get_entities_by_property('类别', '民航企业').result()
        manufacturers2alias,  _, _ = self.get_entity_all_alias(manufacturers_info)  # manufacturers_alias name:[所有别名]
        # reverse dict
        alias2ent = {}
        for ent, alias in manufacturers2alias.items():
            for word in alias:
                word = word.lower()
                if word not in alias2ent:
                    alias2ent[word] = ent.lower()
            if ent not in alias2ent:
                alias2ent[ent] = ent
        return alias2ent

    #  获取所有实体的所有名称，包括所有别名
    def get_entity_all_alias(self, all_entities):
        all_alias = {}
        all_names, prop_list, alias_filter = [], [], []
        alias_prop_list = ['别名', '全称', '简称', '名称', '呼号', '外文', '中文', '英文', 'IATA', 'ICAO', '三字', '二字','曾用','前身']

        for e in all_entities:
            alias = []
            name = e['name']
            country = []
            for place_prop in ['总部国家', '国家', '总部地点', '所在城市']:
                if place_prop in e:
                    country.append(e[place_prop])
            country = list(set(country))
            for prop, val in e.items():
                for word in alias_prop_list:
                    if word in prop:  # 属于别名体系
                        if word == '名称' and "母公司" in prop:
                            continue
                        # 值为list类型
                        if prop in ['名称', '别名'] and val[0] == "[":
                            val = eval(val)
                            alias += [i for i in val]
                            continue
                        # 值是正常字符串类型
                        if ";;;" in val:
                            val = val.split(";;;")[0]
                        alias.append(val)

                        if prop not in prop_list:
                            prop_list.append(prop)
            alias = list(set(alias))
            all_names += alias.copy()
            if name in alias:
                alias.remove(name)
            else:
                all_names += [name]

            if alias:
                all_alias[name] = alias
        all_names = list(set(all_names))
        return all_alias, all_names, prop_list

    def is_special_entity(self, ent_dict):
        return ent_dict['entity_label'] == 'Genre' \
               and (
                   '类' in ent_dict['name']
                   # or '航空公司' in ent_dict['name']
                   # or '行李安检' in ent_dict['name']
               )

    # 保存别名体系
    def save_ent2alias(self, ent2alias, filename):
        f = open(filename, "w")
        f = open(filename, "a")
        for ent, alias in ent2alias.items():
            alias_info = ",".join(alias).strip("\n")
            alias_info = alias_info.replace(": ","")
            f.write(": ".join([ent, alias_info]) + "\n")
        print("新别名体系更新完成")

    # 保存实体字典体系
    def save_lexicon(self, ent_names, filename):
        f = open(filename, "w")
        f = open(filename, "a")
        f.write("\n".join(ent_names))
        print("实体字典更新完成")

    # 生成分词词表：实体、属性名、非纯数字且长度不超过30的属性值
    def save_cut_lexicon(self, ents):
        all_entity_encode_path = 'KM_KBQA/res/entity_encode_1029.txt'
        all_words = []
        for k, item in ents.items():
            for k, v in item.items():
                v = v.replace("\n", '')
                all_words.append(k)
                if k in ['名称', '别名'] and v[0] == "[":
                    v = eval(v)
                    all_words += [i for i in v]
                    continue
                else:
                    all_words.append(v)

        all_words = list(set(all_words))
        f = open("KM_KBQA/lexicons/aerospace_lexicon.txt", 'w')
        for i in all_words:
            if not i or len(i) > 30 or "（" in i or re.findall(r"^\d{1,}$", i):
                continue
            s = i + " n"
            f.write(s + "\n")
        print("新词表saved")
        exit()

    # 保存实体和属性encode
    def save_words_encode(self, ents):
        all_entity_encode_path = 'KM_KBQA/res/entity_encode_1029.txt'
        all_words = []
        for k,item in ents.items():
            for k,v in item.items():
                v = v.replace("\n",'')
                all_words.append(k)
                if k in ['名称', '别名'] and v[0] == "[":
                    v = eval(v)
                    all_words += [i for i in v]
                    continue
                else:
                    all_words.append(v)

        all_words = list(set(all_words))
        for word in all_words:
            if word not in all_entity_encode:
                all_entity_encode[word] = encode(word).numpy().tolist()
                print(word,' encoded')

        f = open(all_entity_encode_path,"w")
        f = open(all_entity_encode_path,"a")
        f.write(json.dumps(all_entity_encode,ensure_ascii=True))
        print("更新")
        exit()


    def gene_name2ent(self,all_entities):
        name2ent = dict()
        # 如果同名实体中有Instance和其他labels的实体，则只保留Instance实体
        for ent in all_entities:
            name = ent['name'].lower()  # 为方便后续英文实体匹配，将所有英文部分转化为小写
            print(ent)
            if name not in name2ent:
                name2ent[name] = {ent['neoId']:[ent['entity_label'],ent]}
            else:
                name2ent[name][ent['neoId']]=[ent['entity_label'],ent]

        final = dict()
        for name,ents in name2ent.items():
            if len(ents)==1:
                final[name]=list(ents.values())[0][1]
            for id,info in ents.items():
                if info[0]!='Instance':
                    continue
                else:
                    final[name] = info[1]
        return final

    def load_all_entities(self, entity_labels=['Instance', 'SubGenre', 'SubGenre_child', 'Genre']):
        all_entities = []
        for entity_label in entity_labels:
            tmp_entities = self.driver.get_all_entities(entity_label, filter_label=['标签']).result()
            for e in tmp_entities:
                e['entity_label'] = entity_label
            # tmp_entities = [e for e in tmp_entities
            #                 if not self.is_special_entity(e)]
            all_entities += tmp_entities
        self.id2ent = {x['name'].lower(): x for x in all_entities}  # 为方便后续英文实体匹配，将所有英文部分转化为小写
        # self.id2ent = self.gene_name2ent(all_entities)
        self.all_alias, self.ent_names, _ = self.get_entity_all_alias(all_entities)

    def load_location(self, tname):
        location = []
        if os.path.isfile(tname):
            with open(tname, 'r', encoding='utf-8') as f:
                for line in f:
                    location.append(line.strip('\n'))
        return location

    # 删除航空公司名称中共有且无意义的字符串，防止因为这些内容导致相似度过高
    def remove_air_word_in_entity(self,entity_name):
        entity_name = entity_name.replace('中国', '')
        for w in config.AIR_COMPANY_TAG:
            if w in entity_name:
                entity_name = entity_name.replace(w, '')
        return entity_name

    # 判断某些词语在是否存在与mention中
    def word_in_mentionlist(self,word_list, mention_list):
        for word in word_list:
            for mention in mention_list:
                if word in mention:
                    return word
        return False

    # @lru_cache(maxsize=128)
    def link(self, sent, sent_cut, pos_tag, is_list, limits=None):
        # use bert embedding to fuzzy match entities
        # mention_list = recognize_entity(sent)
        is_comparison = True if any([word in sent for word in ["区别",'比较','相比']]) else False
        mention_list = retrieve_mention(sent_cut, pos_tag)

        # 特殊处理
        if '机场' in sent and '面积' in sent and '航站楼面积' not in mention_list:
            mention_list.append('航站楼面积')
        logger.debug('指称: ' + str(mention_list))
        print('指称: ' + str(mention_list))

        if not mention_list:
            return []
        # list_words = ['哪些', '几个', '列举', '都有','哪几大']  # "几个"覆盖词汇：哪几个，有几个
        # is_list = any([w in sent for w in list_words])

        country_list = ['俄罗斯', '挪威', '美国', '蒙古', '泰国', '韩国']
        company_word = ["航司", "航空"]  # 航空公司标志词
        Headquaters_location = self.load_location(config.HEADQUATERS_LOCATION)
        province = self.load_location(config.PROVINCE)
        city = self.load_location(config.CITY)

        res = []
        if is_list:  # 分国内航空公司，国外航空公司，航空公司和机场分具体国家，省份，城市
        # 机型列举
            if any([word in sent for word in ['机型', '系列','飞机']]):
                if any([word in sent for word in self.alias2manufacturers]):
                    manufacturer = None
                    # 链接 制造商
                    for word in self.alias2manufacturers:
                        if word in sent:
                            manufacturer = self.alias2manufacturers[word]
                            break
                    if manufacturer:
                        linked_ent = self.driver.get_genres_by_relation(f_genre='Instance',c_genre='SubGenre',f_name=manufacturer,relation='制造').result()
                        if linked_ent:
                            for ent in linked_ent:
                                ent['entity_label'] = 'SubGenre'
                                res.append({
                                    'ent': ent,
                                    'mention': word,
                                    'rel_name': '类别',
                                    'rel_val': '飞机型号',
                                    'id': ent['neoId'],
                                    'score': 1,
                                    'source': 'rule'})
                else:
                    if "型" in sent:
                        type='飞机型号'
                        entity_label = 'SubGenre'
                    else:
                        type='飞机'
                        entity_label = 'Instance'

                    linked_ent = self.driver.get_entities_by_property('类别', type).result()
                    country = self.word_in_mentionlist(country_list,mention_list)
                    if linked_ent:
                        for ent in linked_ent:
                            if ent['name']=='飞机':
                                continue
                            ent['entity_label'] = entity_label
                            res.append({
                                'ent': ent,
                                'mention': '飞机',
                                'rel_name': '类别',
                                'rel_val': '飞机',
                                'id': ent['neoId'],
                                'score': 1,
                                'source': 'rule'})
        # 民航院校
            elif any([word in sent for word in ['民航院校', '航空院校']]):
                for ent in self.id2ent.values():
                    if '类别' not in ent:
                        continue
                    if ent['类别'] == '民航院校':
                        res.append({
                            'ent': ent,
                            'mention': '民航院校',
                            'rel_name': '类别',
                            'rel_val': '民航院校',
                            'id': ent['neoId'],
                            'score': 1,
                            'source': 'rule'})
        # 航司列举
            elif any([word in mention for word in company_word for mention in mention_list]):
                if self.word_in_mentionlist(['国内','中国'],mention_list):
                    key_word = self.word_in_mentionlist(['国内','中国'],mention_list)
                    # 国内航空公司
                    for ent in self.id2ent.values():
                        if '类别' not in ent:
                            continue
                        if ent['类别'] == '国内航空公司':
                            res.append({
                                'ent': ent,
                                'mention': ''.join([key_word, '航空公司']),
                                'rel_name': '类别',
                                'rel_val': '中国航空公司',
                                'id': ent['neoId'],
                                'score': 1,
                                'source': 'rule'})
                elif self.word_in_mentionlist(country_list,mention_list):
                    # 某个城市的航空公司
                    country = self.word_in_mentionlist(country_list,mention_list)
                    # word = [word for word in mention_list if word in country_list][0]
                    for ent in self.id2ent.values():
                        flag = False
                        if '类别' not in ent:
                            continue
                        if ent['类别'] == '国外航空公司':
                            if '别名' in ent and any([country in name for name in eval(ent['别名'])]):
                                flag = True
                            if '公司名称' in ent and country in ent['公司名称']:
                                flag = True
                            if flag:
                                res.append({
                                    'ent': ent,
                                    'mention': country + '航空公司',
                                    'rel_name': '类别',
                                    'rel_val': '国外航空公司',
                                    'id': ent['neoId'],
                                    'score': 1,
                                    'source': 'rule'})
                elif any([word in Headquaters_location for word in mention_list]):
                    # 根据总部地点查找航司
                    for word in mention_list:
                        if word in Headquaters_location:
                            head_l = word
                    for ent in self.id2ent.values():
                        if '总部地点' not in ent:
                            continue
                        if ent['总部地点'] == head_l:
                            res.append({
                                'ent': ent,
                                'mention': ''.join([head_l, '航空公司']),
                                'rel_name': '总部地点',
                                'rel_val': head_l + '航空公司',
                                'id': ent['neoId'],
                                'score': 1,
                                'source': 'rule'})
                elif any(['旗下' in mention or '母公司' in mention for mention in mention_list]):
                    relation = '母公司'
                    def find_index(list,word):
                        for i, item in enumerate(list):
                            if item == word:
                                return i
                        return False
                    # 分词排序
                    pos_rank = {word:sent.find(word) for word in sent_cut}  # 按分词在原句子中的位置排序
                    sent_cut_rank = sorted(pos_rank.items(),key=lambda x:x[1])
                    sent_cut_rank = [item[0] for item in sent_cut_rank]  # ['海航', '旗下', '哪些', '航空公司']

                    find_p_company = False  # 是否找到母公司名
                    prev_relation = ""
                    # 旗下
                    if '旗下' in sent:
                        relation_word = '旗下'
                        relation_pos = find_index(sent_cut_rank, '旗下')
                        if relation_pos != 0:
                            prev_relation = sent_cut_rank[relation_pos - 1]
                    elif '母公司' in sent:  # 母公司
                        relation_word = '母公司'
                        relation_pos = find_index(sent_cut_rank, '母公司')
                        prev_relation = sent_cut_rank[relation_pos + 1]

                    is_not_ent, ent_name = self.is_not_entity(prev_relation)
                    if is_not_ent is False:
                        find_p_company = True
                        parent_company = ent_name[0]

                    if find_p_company:
                        for ent in self.id2ent.values():
                            if '母公司' not in ent:
                                continue
                            if ent['母公司'] == parent_company:
                                res.append({
                                    'ent': ent,
                                    'mention': prev_relation+relation_word,
                                    'rel_name': relation,
                                    'rel_val': parent_company,
                                    'id': ent['neoId'],
                                    'score': 1.5,
                                    'source': 'rule'})
                else:
                    for ent in self.id2ent.values():
                        if '类别' not in ent:
                            continue
                        if ent['类别'] in ['国外航空公司','国内航空公司']:
                            res.append({
                                'ent': ent,
                                'mention': ent['类别'],
                                'rel_name': '类别',
                                'rel_val': ent['类别'],
                                'id': ent['neoId'],
                                'score': 1,
                                'source': 'rule'})
        # 机场列举
            elif any(["机场" in mention for mention in mention_list]):
                if any([word in mention for word in ['国内', '中国'] for mention in mention_list]):
                    # 国内机场
                    for word in mention_list:
                        if word in province:
                            pro = word
                    for ent in self.id2ent.values():
                        if '类别' not in ent:
                            continue
                        if ent['类别'] == '国内机场':
                            res.append({
                                'ent': ent,
                                'mention': '国内机场',
                                'rel_name': '类别',
                                'rel_val': '国内机场',
                                'id': ent['neoId'],
                                'score': 1,
                                'source': 'rule'})
                elif any([word in province for word in mention_list]):
                    # 某个省份的机场
                    for word in mention_list:
                        if word in province:
                            pro = word
                    for ent in self.id2ent.values():
                        if '省份' not in ent:
                            continue
                        if ent['省份'] == pro:
                            res.append({
                                'ent': ent,
                                'mention': ''.join([pro, '的机场']),
                                'rel_name': '省份',
                                'rel_val': pro + '的机场',
                                'id': ent['neoId'],
                                'score': 1,
                                'source': 'rule'})
                elif any([word in city for word in mention_list]):
                    # 某个城市的机场
                    for word in mention_list:
                        if word in city:
                            ci = word
                            break
                    for ent in self.id2ent.values():
                        if '所在城市' not in ent or '类别' not in ent or ent['类别'] not in ['国内机场', '国外机场']:
                            continue
                        if ent['所在城市'] == ci:
                            res.append({
                                'ent': ent,
                                'mention': ''.join([ci, '的机场']),
                                'rel_name': '所在城市',
                                'rel_val': ci + '的机场',
                                'id': ent['neoId'],
                                'score': 1,
                                'source': 'rule'})

            if res:
                return res

        for mention in mention_list:
            mention = mention.lower()
            one_res = []
            if not contain_chinese(mention):
                search_list = []
                if '机场' in mention_list:
                    search_list = ['机场三字码', 'ICAO机场代码']
                elif any([word in sent for word in company_word]):
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
            is_not_ent, cand_names = self.is_not_entity(mention)
            if is_not_ent and "-" in mention:
                is_not_ent, cand_names = self.is_not_entity(mention.replace("-",''))
            # cand_names = self.convert_mention2ent(mention)  # 查找 mention代表的实体名称
            # 别名字典中找到了mention代表的实体,直接返回候选实体
            if cand_names:
                for linked_ent_name in cand_names:
                    # 通过name查找实体信息
                    ent = self.id2ent[linked_ent_name]
                    one_res.append({
                        'ent': ent,
                        'mention': mention,
                        'id': ent['neoId'],
                        'score': 1,
                        'source': 'rule'
                    })
            else:
                cand_names = [mention]
            # 没有通过别名字典直接匹配到实体，则进行相似度匹配
            if not one_res:
                for ent in self.id2ent.values():
                    if 'name' not in ent:
                        continue
                    if '机场' not in mention and "类别" in ent and (ent['类别'] == '国外机场' or ent['类别'] == '国内机场'):
                        continue

                    ent_name = ent['name']
                    ent_name_rewrite = self.rewrite_ent_name(ent_name)  # 取中英文括号前面的部分
                    if ent_name_rewrite == '':
                        continue
                    for cand_name in cand_names:
                        # 删除航空公司名称中共有且无意义的字符串，防止因为这些内容导致相似度过高
                        if '航空' in cand_name and ('航空' in ent_name_rewrite and '公司' in ent_name_rewrite):
                            cand_name = self.remove_air_word_in_entity(cand_name)
                            ent_name_rewrite = self.remove_air_word_in_entity(ent_name_rewrite)
                        if '机场' in cand_name and '机场' in ent_name_rewrite:
                            cand_name = cand_name.replace('机场', '')
                            ent_name_rewrite = ent_name_rewrite.replace('机场', '')
                        RATIO = 0.5
                        # 该实体为英文而问的有汉语或相反
                        # 原因：当时用bert encode以后进行相似度匹配的时候，输入的实体或者图谱中的实体有英文的话有时语义不相近bert也会给出较高的值，所以过滤掉只有一方出现英文的情况
                        if contain_chinese(cand_name) and not contain_chinese(ent_name_rewrite) or contain_english(
                            cand_name) and not contain_english(ent_name_rewrite):
                            score = fuzz.UQRatio(cand_name, ent_name_rewrite) / 100
                        else:
                            score = cosine_word_similarity(cand_name, ent_name_rewrite)
                            # score_cos = score
                            score1 = fuzz.UQRatio(cand_name, ent_name_rewrite) / 100
                            score = RATIO * score + (1 - RATIO) * score1
                        one_res.append({
                            'ent': ent,
                            'mention': mention,
                            'id': ent['neoId'],
                            'score': score,
                            'source': 'rule'
                        })
                        """
                        if score == 1:  # score为1 ，说明mention和实体名相同，直接连接到了正确实体，其他实体不用再匹配
                            break
                        """  # 因为存在同名实体，如术语PCN和皮克顿机场机场的三字码都是PCN，所以不能用score=1判断匹配截止
            # 连接到的实体中事发后有非术语类型的实体, 是, 则降低民航术语实体分值
            link_not_terms_flag = any([ent['ent']['类别']!='民航术语' and ent['ent']['name'] not in config.FREQUENT_MATCHED_ENT for ent in one_res if ent['score'] > config.simi_ths])
            # 降低常见词、民航术语的分值
            # 常见词 或 民航术语 在其他类别实体的属性中出现的几率大，如果有匹配到非术语类实体，则需要降低置信度
            if link_not_terms_flag:
                for i, a_res in enumerate(one_res):
                    if a_res['score']> config.simi_ths:
                        if a_res['ent']['name'] in config.FREQUENT_MATCHED_ENT :
                            one_res[i]['score'] = 0.8
                        elif a_res['ent']['类别'] == "民航术语":
                            one_res[i]['score'] = 0.81

            if not one_res:
                continue
            one_res.sort(key=lambda x: x['score'], reverse=True)

            choose_top_n = 1 if is_comparison else 3
            res_id2ent = {ent['id']: ent for ent in res}
            for a_res in one_res[:choose_top_n]:
                if a_res['score'] > config.simi_ths:
                    # 如果多次链接同个实体，只保留score最高的结果
                    if a_res['id'] not in res_id2ent or \
                        (a_res['id'] in res_id2ent and res_id2ent[a_res['id']]['score'] < a_res['score']):
                        res_id2ent[a_res['id']] = a_res
            res = [item for id, item in res_id2ent.items()]
        # 如果链接到了非术语类型的实体，则降低术语类型实体的分值
        link_not_terms_ent = any([ent['ent']['类别'] != '民航术语' and ent['ent']['name'] not in config.FREQUENT_MATCHED_ENT for ent in res])
        if link_not_terms_ent:
            for i, r in enumerate(res):
                if r['score']> config.simi_ths and (r['ent']['类别'] == "民航术语" or r['ent']['name'] in config.FREQUENT_MATCHED_ENT):
                    res[i]['score'] = 0.8
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

    def convert_mention2ent(self, mention) -> list:
        if not contain_chinese(mention):
            word_list = [mention, mention.upper()]  # 英文字符串同时判断大写和小写
        else:
            word_list = [mention]

        for item in word_list:
            ent_names = self.mention2ent.get(item, None)
            if ent_names is not None:
                if not isinstance(ent_names, list):
                    ent_names = [ent_names]
                return ent_names
        return [mention]

    def merge_dict(self,dict1,dict2):
        tmp = dict2
        return tmp.update(dict1)

    def is_not_entity(self, mention):
        """
        判断是否是实体
        return bool 实体是否是实体, 代表的实体名名称
        """
        ent_names = []
        all_ent_names = [item.lower() for item in self.ent_names]
        if mention in all_ent_names:  # self.ent_names是所有实体name和别名，根据self.mention2ent可以找到实体的name
            # 是否属于实体别名
            ent_names = self.mention2ent.get(mention, [])
            if ent_names:
                if not isinstance(ent_names, list):
                    ent_names = [ent_names.lower()]
            else:
                ent_names.append(mention)

        if ent_names:
            return False, ent_names
        else:
            return True, None


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
