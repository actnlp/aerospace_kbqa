import logging
import os
import pdb
import re
from copy import deepcopy

import torch
from fuzzywuzzy import fuzz

from ..config import config
from ..common import LTP, AsyncNeoDriver, HITBert
from ..linking import EntityLinking
from ..linking.EntityLinking import recognize_entity
from .Limiter import Limiter
from .ListQuestion import check_list_questions

logging.basicConfig(level=logging.INFO)

# 检查是否存在中文字符，包括一些特定英文字符
def is_contain_chinese(s):
    ss = s.replace('-', '').lower()
    if ss == 'wifi' or ss == 'atm' or 'vip' in ss or ss == 'kfc':
        return True
    for c in s:
        if ('\u4e00' <= c <= '\u9fa5'):
            return True
    return False


# 检查是否存在英文字符
def is_contain_english(s):
    return bool(re.search('[A-Za-z]', s))


# 过滤词汇
def filter_sentence(sent, stop_list, rpl_list):
    for stop in stop_list:
        sent = sent.replace(stop, '')
    for replace in rpl_list:
        sent = sent.replace(replace[0], replace[1])
    return sent


class Node:
    def __init__(self, neoId, name, id, properties, type, ans_type):
        self.neoId = neoId
        self.name = name
        self.id = id
        self.properties = properties
        self.type = type
        self.ans_type = ans_type


class Edge:
    def __init__(self, src_id, tgt_id, name, ans_type):
        self.src = src_id
        self.tgt = tgt_id
        self.name = name
        self.ans_type = ans_type


class QAHelper():
    remove_prop = ['subname', 'description',
                   'label', 'taglist', 'neoId', 'keyId', 'id', 'score', 'rel', 'hidden']
    stop_list = ['吗', '里面']
    rpl_list = [('在哪儿', '的地点'), ('在哪里', '的地点'), ('在哪', '的地点'), ('哪里', '地点'), ('哪里', '地点'), ('哪有', '地点'), ('属于', '在'), ('vip', '贵宾')]

    def __init__(self, driver, linker):
        self.driver = driver
        self.linker = linker
        self.hit_encode = HITBert.encode
        self.cut = []
        self.entity_cache = {}

        self.all_entities = self.driver.get_all_entities('Instance').result()
        self.all_entities = [x['name'] for x in self.all_entities]
        self.all_entity_embeddings = {}
        self.cache = {}
        self.limits = []

        self.ret_id = 0

    def load_dictionary(self, ent_emb):
        self.all_entity_embeddings = ent_emb
        self.cache = deepcopy(ent_emb)
        return

    def link_entity(self, sent):
        def convert_abstract_verb(wd, sent, limits):
            convert_dict = config.ABSTRACT_DICT
            if convert_dict.get(wd) is not None:
                if wd == '换':
                    if limits['币种'] != '' or '货币' in sent or '外币' in sent:
                        return convert_dict.get(wd)[0]
                    elif '尿' in sent:
                        return convert_dict.get(wd)[1]
                    else:
                        return wd
                # if wd == '换':
                #     if '货币' or '外币'
                return convert_dict[wd]
            else:
                # 去除"服务"字段的影响
                return ''.join(wd.split('服务'))

        def filter_q_entity(item):
            for wd in config.airport.filter_words:
                if wd in item:
                    return True
            for wd in config.airport.remove_words:
                if wd == item:
                    return True
            if '时间' in item or '地点' in item or '位置' in item or '地方' in item or '收费' in item or '价格' in item or '限制' in item or item == '电话':  # or not is_contain_chinese(item):
                return True
            return False

        def filter_key(item):
            item = item.split('(')[0].split('（')[0].lower()
            if '服务' in item and item != '服务':
                item = ''.join(item.split('服务'))
            if item == '柜台' or item == '其他柜台' or item == '行李' or item =='咨询':
                item = ''
            return item

        # use bert embedding to fuzzy match entities
        mention_list = recognize_entity(sent)
        if mention_list == []:
            return []
        if len(self.cut) == 0:
            self.cut = LTP.customed_jieba_cut(sent, os.path.join(config.LEX_PATH, 'stopwords.txt'), cut_stop=True)
        # print('cut:', self.cut)
        res = []
        for item in mention_list:
            one_res = []
            if filter_q_entity(item):
                continue
            converted_item = convert_abstract_verb(item, sent, self.limits)
            for key in self.all_entity_embeddings.keys():
                # 该实体为英文而问的有汉语或相反
                if is_contain_chinese(converted_item) and not is_contain_chinese(key) or is_contain_english(
                        converted_item) and not is_contain_english(key):
                    continue
                filtered_key = filter_key(key)
                if filtered_key == '':
                    continue
                score = self.calculate_word_similarity(self.hit_encode, converted_item, filtered_key)
                if fuzz.token_sort_ratio(' '.join(converted_item), ' '.join(filtered_key)) > 70 and fuzz.token_sort_ratio(converted_item, filtered_key) >= 50 and len(
                        converted_item) > 1:
                    score *= 1.2
                '''if item in key and len(item) > 1:
                    score *= 1.1'''
                # punish english words
                '''if not is_contain_chinese(key) or len(key) == 1:
                    score *= 0.8'''
                one_res.append([key, score])
            one_res = sorted(one_res, key=lambda x: x[1], reverse=True)
            for a_res in one_res[:3]:
                if a_res[1] > config.simi_ths:
                    res.append([item] + a_res)
        res = sorted(res, key=lambda x: x[2], reverse=True)

        return res

    def get_child_in_father(self, sg_name, c_label, f_label):
        instances = self.driver.get_genres_by_relation(f_label, c_label, sg_name, reverse=True).result()
        neoIds = [int(x['neoId']) for x in instances]
        instances = [x['name'] for x in instances]
        return instances, neoIds

    def list_extraction(self, neoId, cand_name, ent_name, limits, ent_label, ent_score):
        # extract all prop， 限制支持一个
        flag = True
        for key in limits.keys():
            if limits[key] != '':
                flag = False
        props_tmp_list = self.driver.get_props_by_id(neoId, ent_label).result()
        props_list = {}
        for prop in props_tmp_list.keys():
            if prop not in self.remove_prop:
                props_list[prop] = str(props_tmp_list[prop])
        if flag:
            # 无限制
            for prop in props_tmp_list.keys():
                if '地点' in prop:
                    return [ent_name + ', 无限制, 地点: ' + props_tmp_list[prop], neoId, cand_name, ent_name, None, None,
                            ent_score, ent_score]
            return [ent_name + ', 无限制', neoId, cand_name, ent_name, None, None, ent_score, ent_score]
        res_limit = self.cal_limit(limits, props_list)

        if res_limit is None:
            for limit in limits.keys():
                if limits[limit] != '':
                    return [ent_name + ', ' + limit + ' 无信息，不满足题意', neoId, cand_name, ent_name, None, None, ent_score,
                            ent_score]

        ret_limit = ''
        accepted_limit = {}
        for limit in res_limit.keys():
            if not res_limit[limit]:
                return [ent_name + ', ' + props_list[limit] + ' 不满足题意', neoId, cand_name, ent_name, None, None,
                        ent_score, ent_score]
            else:
                ret_limit += limit + ', ' + props_list[limit] + ' '
                accepted_limit[limit] = props_list[limit]

        return [ent_name + ', ' + ret_limit, neoId, cand_name, ent_name, None, accepted_limit, ent_score, ent_score]

    def cal_limit(self, limits, props):
        # check whether limit is available
        exist_limit = []
        res_limit = {}
        for limit in limits.keys():
            if limits[limit] != '' and limits[limit] is not None:
                flag = True
                for prop in props.keys():
                    if limit in prop or limit in props[prop] or limits[limit][0] in prop or limits[limit][0] in props[
                        prop]:
                        flag = False
                        res_limit[prop] = True
                        exist_limit.append((prop, limit))
                if flag:
                    return None

        # filter result
        time_pattern = re.compile(r'\d+[:, ：]')
        for prop, limit in exist_limit:
            limit_content = props[prop].lower()
            if '地点' in limit:
                for item in limits['地点']:
                    if item not in limit_content:
                        res_limit[prop] = False
            elif '时间' in limit:
                for item in limits['时间']:
                    #  ''' or item == '最早' or item == '最晚'''''
                    if (item == '24小时' or item == '最早' or item == '最晚') and item not in limit_content:
                        res_limit[prop] = False
                        continue
                    bg_ed = time_pattern.findall(limit_content)
                    bg_ed = [int(x[:-1]) for x in bg_ed]
                    if '时' not in item:
                        if len(bg_ed) == 2:
                            if not (bg_ed[0] < int(item) < bg_ed[1]):
                                res_limit[prop] = False

            elif '币种' in limit or '银行' in limit or '航空公司' in limit or '价格' in limit:
                for item in limits[limit]:
                    if item not in limit_content:
                        res_limit[prop] = False

        return res_limit

    def rel_extraction(self, neoId, cand_name, ent_name, limits, cut_words, ent_label, ent_score, thres=config.prop_ths):
        # extract all prop， 限制支持一个
        props_list = {}
        props_tmp_list = self.driver.get_props_by_id(neoId, ent_label).result()

        for prop in props_tmp_list.keys():
            if prop not in self.remove_prop:
                props_list[prop] = str(props_tmp_list[prop])
        try:
            res_limit = self.cal_limit(limits, props_list)
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
                accepted_limit[limit] = props_list[limit]

        # cut
        limit_list = list(map(lambda x: x[1], list(limits.items())))
        rest_words = list(filter(lambda x: x not in cand_name and '机场' not in x and x not in limit_list, cut_words))
        props_set = list(props_list.keys())
        props_set.remove('name')
        # cal prop rel similarity
        res = []
        used_pairs = set()

        def normalize_prop(prop):
            if '地点' in prop:
                return '地点'
            if '时间' in prop:
                return '时间'
            return prop

        def normalize_ratio(word, prop):
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

        for prop in props_set:
            for word in rest_words:
                score = self.calculate_word_similarity(self.hit_encode, word, normalize_prop(prop))
                ratio = normalize_ratio(word, prop)
                score = ratio if ratio > 1 else score
                if word in prop and len(word) > 1:
                    score *= 1.2
                if score > thres and (word, prop) not in used_pairs:
                    used_pairs.add((word, prop))
                    res.append([neoId, cand_name, ent_name, {prop: props_list[prop]}, accepted_limit, score, ent_score])
        if len(res) == 0:
            return None
        print('rel_res:', res)
        res = sorted(res, key=lambda x: x[5], reverse=True)
        res_lang = []
        for item in res:
            rel = list(item[3].keys())[0]
            val = item[3][rel]
            ans = item[2] + '的' + rel + '是' + val
            if wrong_restriction != '':
                ans += ' ' + wrong_restriction
            res_lang.append([ans] + item)
        # 如果前两个属性都很高，那返回两个答案
        sel_num = 2 if res_lang[0][6] > 0.91 and res_lang[0][6] > 0.91 else 1
        return res_lang[:min(len(res), sel_num)]

    def qa(self, sent):
        # initialization
        self.cut = []
        # 替换规则
        sent = filter_sentence(sent, self.stop_list, self.rpl_list)
        # 获取限制
        limiter = Limiter(sent)
        limits = limiter.check()
        self.limits = limits
        print(['limits', limits])

        # 合并两个集合
        link_res = self.link_entity(sent)


        # 判断是否是列举类型
        sent_is_list = check_list_questions(sent, self.link_entity)
        print('cut words: ', self.cut)
        # eng_letters = [chr(x) for x in range(ord('a'), ord('z') + 1)] + [chr(x) for x in range(ord('A'), ord('Z') + 1)]
        # check_list_sent = WordCLS(
        #     sent=sent,
        #     bc=self.bc,
        #     thres=55,
        #     conf_lw=['几个', '哪些', 'have'], # , '什么', '多少']
        #     ignore_w=eng_letters # never match English letters
        # )
        # sent_is_list = check_list_sent.has_similar_words(['是什么', '是多少', '多少钱'])

        # obtain labels and neoId
        print('链接结果:', link_res)
        print('is_list: ', sent_is_list)
        all_link_res = []
        for entity in link_res:
            label_id = self.driver.get_labels_by_name(entity[1]).result()
            for label, neoId in label_id:
                all_link_res.append(entity + label + [neoId])

        # entity: [cand_name, ent_name, ent_score, label, neoId]
        # all_result: [ans, neoId, cand_name, name, {rel: rel_val}, {rst: rst_val}, rel_score, ent_score]

        # 关系提取
        all_result = []
        for cand_name, ent_id, ent_score, ent_label, ent_neoId in all_link_res:
            if ent_label == 'SubGenre' or ent_label == 'Genre':
                c_instances, c_neoIds = self.get_child_in_father(ent_id, 'Instance', ent_label)
                # 该父节点无子节点
                if c_instances is None or len(c_instances) == 0:
                    continue
                # 父节点有子节点
                for i, instance in enumerate(c_instances):
                    if sent_is_list:
                        ret_list = self.list_extraction(c_neoIds[i], cand_name, instance, limits, 'Instance', ent_score)
                        if ret_list is not None and len(ret_list) > 0:
                            all_result.append(ret_list)
                    else:
                        ret_rel = self.rel_extraction(c_neoIds[i], cand_name, instance, limits, self.cut, 'Instance',
                                                      ent_score)
                        if ret_rel is not None and len(ret_rel) > 0:
                            all_result += ret_rel
                        # 若实体识别分之很高，那么就转而进行列举类，返回常见属性
                        elif ent_score > 0.9:
                            ret_list = self.list_extraction(c_neoIds[i], cand_name, instance, limits, 'Instance', ent_score)
                            if ret_list is not None and len(ret_list) > 0:
                                all_result.append(ret_list)

            elif ent_label == 'Instance':
                if sent_is_list:
                    ret_list = self.list_extraction(ent_neoId, cand_name, ent_id, limits, ent_label, ent_score)
                    if ret_list is not None and len(ret_list) > 0:
                        all_result.append(ret_list)
                else:
                    ret_rel = self.rel_extraction(ent_neoId, cand_name, ent_id, limits, self.cut, ent_label, ent_score)
                    if ret_rel is not None and len(ret_rel) > 0:
                        all_result += ret_rel
                    # 若实体识别分之很高，那么就转而进行列举类，返回常见属性
                    elif ent_score > 0.9:
                        ret_list = self.list_extraction(ent_neoId, cand_name, ent_id, limits, ent_label, ent_score)
                        if ret_list is not None and len(ret_list) > 0:
                            all_result.append(ret_list)

        # 问题只有机场
        def is_airport_alias(name):
            for alias in config.airport.alias:
                if name == alias:
                    return True
            return False

        # 找到实体却没有找到属性或只有一个光秃秃的机场名
        if len(all_result) == 0 and (
                    is_airport_alias(sent) or (len(all_link_res) > 0 and all_link_res[0][2] > config.exact_ent_ths)):
            sent_is_list = True
            # 去除所有score大于0.9的Instance
            all_linked_instances = list(
                filter(lambda x: x[2] > config.exact_ent_ths and x[3] == 'Instance', all_link_res))
            if len(all_link_res) > 0:
                ent_id, ent_score, ent_label, neoId = all_link_res[0][1], all_link_res[0][2], all_link_res[0][3], \
                                                      all_link_res[0][4]
            else:
                label_id = self.driver.get_labels_by_name(config.airport.name).result()[0]
                ent_id, ent_score, ent_label, neoId = config.airport.name, 1.0, label_id[0][0], label_id[1]
            # 判断该节点是否为根节点
            if ent_label == 'Instance':
                for c_cand_name, c_ent_name, c_ent_score, c_ent_label, c_ent_neoId in all_linked_instances:
                    ret_list = self.list_extraction(c_ent_neoId, c_ent_name, c_ent_name, limits, c_ent_label,
                                                    c_ent_score)
                    if ret_list is not None and len(ret_list) > 0:
                        all_result.append(ret_list)
            else:
                instances, neoIds = self.get_child_in_father(ent_id, 'Instance', ent_label)
                if instances is None or len(instances) == 0:
                    ret_list = self.list_extraction(neoId, ent_id, ent_id, limits, ent_label, ent_score)
                    if ret_list is not None and len(ret_list) > 0:
                        all_result.append(ret_list)
                else:
                    for i, instance in enumerate(instances):
                        ret_list = self.list_extraction(neoIds[i], ent_id, instance, limits, 'Instance', ent_score)
                        if ret_list is not None and len(ret_list) > 0:
                            all_result.append(ret_list)

        # 最后查看是不是机场的属性
        if len(all_result) == 0 and '机场' in sent:
            # check if it is a question related to the airport
            label_id = self.driver.get_labels_by_name(config.airport.name).result()[0]
            ret_rel = self.rel_extraction(label_id[1], config.airport.name, config.airport.name, limits, self.cut,
                                          'Center', 1.0, thres=0.85)
            if ret_rel is not None and len(ret_rel) > 0:
                print('机场属性：', ret_rel)
                sent_is_list = False
                all_result += ret_rel

        # 排序
        if not sent_is_list:
            # 针对关系类问题，限制正确且实体识别辨识度高的选项放在前面
            satisfied_result = list(filter(lambda x: x[0][-4:] != '限制错误' and x[0][-5:] != '不满足题意' and x[-1] >= config.exact_ent_ths, all_result))
            unsatisfied_result = list(filter(lambda x: x[0][-4:] == '限制错误' or x[0][-5:] == '不满足题意' or x[-1] < config.exact_ent_ths, all_result))
            # 再根据关系匹配值排序
            satisfied_result = sorted(satisfied_result, key=lambda x: x[-2], reverse=True)
            unsatisfied_result = sorted(unsatisfied_result, key=lambda x: x[-2], reverse=True)
            all_result = satisfied_result + unsatisfied_result
        else:
            # 针对列举类问题，首先列举满足题意的项目
            satisfied_res = list(filter(lambda x: x[0][-5:] != '不满足题意' and x[0][-4:] != '限制错误' and x[-1] >= config.exact_ent_ths, all_result))
            unsatisfied_res = list(filter(lambda x: x[0][-5:] == '不满足题意' or x[0][-4:] == '限制错误' or x[-1] < config.exact_ent_ths, all_result))
            all_result = satisfied_res + unsatisfied_res

        # 去重
        id_prop_set = set()
        filtered_all_result = []
        for r in all_result:
            ent_id, ent_rel = r[1], list(r[4].keys())[0] if r[4] is not None else r[4]
            if (ent_id, ent_rel) not in id_prop_set:
                id_prop_set.add((ent_id, ent_rel))
                filtered_all_result.append(r)
        print('top3:')
        for ar in filtered_all_result[:3]:
            print('top3', ar)
        if len(filtered_all_result) == 0:
            # cannot answer this question due to limited data
            return {'type': 'not recognized', 'genre': ''}, [
                ['系统没有检测到对应的实体或属性，请重新输入！', None, None, None, None, None, None, None]]

        if sent_is_list:  # 只针对类型为SubGenre的排，没有的话就是数据库不存在
            return {'type': 'list', 'genre': ''}, filtered_all_result

        return {'type': 'exact', 'genre': ''}, filtered_all_result

    def calculate_word_similarity(self, encode, w1, w2, type='bert'):
        # calculate bert
        if (w1, w2) in self.cache.keys():
            bert_score = self.cache[(w1, w2)]
        elif (w2, w1) in self.cache.keys():
            bert_score = self.cache[(w2, w1)]
        else:
            if w1 in self.entity_cache.keys():
                e1 = [self.entity_cache[w1]]
            else:
                e1 = encode(w1)
                self.entity_cache[w1] = e1[0]
            if w2 in self.entity_cache.keys():
                e2 = [self.entity_cache[w2]]
            else:
                e2 = encode(w2)
                self.entity_cache[w2] = e2[0]
            a1, a2 = torch.tensor(e1), torch.tensor(e2)
            dist = torch.sum((a1 - a2) ** 2).item()
            bert_score = torch.cosine_similarity(a1, a2).item()
            self.cache[(w1, w2)] = bert_score
        '''bert_score = (bert_score - 0.8) * 4
        # calculate cuzzy
        fuzzy_score = 0.002 * fuzz.token_sort_ratio(w1, w2)
        cos = fuzzy_score + bert_score'''
        return bert_score

    def add_node(self, neoId, ans_type, name=''):
        # 如果为属性值节点
        if neoId < 0:
            entity = Node(neoId, name, self.ret_id, {}, 'Property', ans_type)
        else:
            ent_name = self.driver.get_name_by_id(neoId).result()[0]
            ent_type = self.driver.get_label_by_id(neoId).result()[0]
            ent_prop = self.driver.get_props_by_id(neoId, ent_type).result()
            entity = Node(neoId, ent_name, self.ret_id, ent_prop, ent_type, ans_type)
        self.ret_id += 1
        return entity

    def add_edge(self, src, tgt, name, ans_type):
        rel = Edge(src, tgt, name, ans_type)
        return rel

    def fetch_surrounding_entities(self, route, num=5):
        surround = []
        ent_route = list(filter(lambda x: isinstance(x, Node), route))
        ent_route.reverse()
        # rel_route = list(map(lambda x: isinstance(x, Edge)))
        for i, ent in enumerate(ent_route[:-1]):
            # 针对链上的每一个实体
            if ent.type == 'Center':
                tgt_type = 'Genre'
            elif ent.type == 'Genre':
                tgt_type = 'SubGenre'
            elif ent.type == 'SubGenre':
                tgt_type = 'Instance'
            else:
                raise KeyError('Wrong Label Detected!')

            cand_surround = self.driver.get_relations_by_id(ent.neoId, ent.type, tgt_type).result()
            if cand_surround is not None and len(cand_surround) > 0 and len(cand_surround[0]) == 3:
                for j, cand in enumerate(cand_surround):
                    rel, c_ent_name, c_ent_neoId = cand
                    # 计算route中子节点与
                    score = self.calculate_word_similarity(self.hit_encode, c_ent_name, ent_route[i + 1].name)
                    cand_surround[j].append(score)
            cand_surround = sorted(cand_surround, key=lambda x: x[3], reverse=True)

            for rel, c_ent_name, c_ent_neoId, score in cand_surround[:num]:
                if c_ent_neoId == ent_route[i + 1].neoId:
                    continue
                s_entity = self.add_node(c_ent_neoId, 'surrounding_entity')
                s_rel = self.add_edge(ent.id, s_entity.id, rel, 'surrounding_relation')
                surround.append(s_entity)
                surround.append(s_rel)

        return surround

    def fetch_route_entities(self, ent):
        # 判断ans_ent的label
        route = [ent]
        if ent.type == 'Instance':
            relation, f_name, f_neoId = self.driver.get_relations_by_id(ent.neoId, '', '').result()[0]
            linked_ent = self.add_node(f_neoId, 'route_entity')
            linked_edge = self.add_edge(linked_ent.id, ent.id, relation, 'route_relation')
            route.append(linked_edge)
            route.append(linked_ent)
            ent = linked_ent
        if ent.type == 'SubGenre':
            relation, f_name, f_neoId = self.driver.get_relations_by_id(ent.neoId, ent.type, 'Genre').result()[0]
            linked_ent = self.add_node(f_neoId, 'route_entity')
            linked_edge = self.add_edge(linked_ent.id, ent.id, relation, 'route_relation')
            route.append(linked_edge)
            route.append(linked_ent)
            ent = linked_ent
        if ent.type == 'Genre':
            relation, f_name, f_neoId = self.driver.get_relations_by_id(ent.neoId, ent.type, 'Center').result()[0]
            linked_ent = self.add_node(f_neoId, 'route_entity')
            linked_edge = self.add_edge(linked_ent.id, ent.id, relation, 'route_relation')
            route.append(linked_edge)
            route.append(linked_ent)
            ent = linked_ent

        assert ent.name == config.airport.name

        return route

    def decorate(self, qa_res, res_type, ans_num=3):
        ret = {"answers": []}
        # 判断根据指示图谱能否回答
        if qa_res[0][0] == '系统没有检测到对应的实体或属性，请重新输入！':
            return {'answers': [{'answer': '系统没有检测到对应的实体或属性！', 'nodes': [], 'edges': []}]}
        # 获得答案各部分
        for r in qa_res[:ans_num]:
            self.ret_id = 0
            ans, neoId, cand_name, name, rel, rstct, _, _ = r
            nodes, edges = [], []
            # 若没有答案则跳过
            if neoId is None:
                continue
            # 先构造问题实体，问题属性，问题限制
            ans_ent = self.add_node(neoId, 'answer_entity')
            if rel is not None:
                for rel_name in rel.keys():
                    ans_prop_val = self.add_node(-1, 'answer_property_value', rel[rel_name])
                    ans_prop_rel = self.add_edge(ans_ent.id, ans_prop_val.id, rel_name, 'answer_property')
                    nodes.append({'name': ans_prop_val.name, 'id': ans_prop_val.id, 'property': ans_prop_val.properties,
                                  'type': ans_prop_val.type, 'is_ans': ans_prop_val.ans_type})
                    edges.append({'source': ans_prop_rel.src, 'target': ans_prop_rel.tgt, 'name': ans_prop_rel.name,
                                  'is_ans': ans_prop_rel.ans_type})
            if rstct is not None:
                for rst_name in rstct.keys():
                    ans_rstct_val = self.add_node(-1, 'answer_restriction_value', rstct[rst_name])
                    ans_rstct_rel = self.add_edge(ans_ent.id, ans_rstct_val.id, rel, 'answer_restriction')
                    nodes.append(
                        {'name': ans_rstct_val.name, 'id': ans_rstct_val.id, 'property': ans_rstct_val.properties,
                         'type': ans_rstct_val.type, 'is_ans': ans_rstct_val.ans_type})
                    edges.append({'source': ans_rstct_rel.src, 'target': ans_rstct_rel.tgt, 'name': ans_rstct_rel.name,
                                  'is_ans': ans_rstct_rel.ans_type})

            # 从答案实体开始寻找到黄花机场的路径
            route = self.fetch_route_entities(ans_ent)
            surround = self.fetch_surrounding_entities(route)

            for item in route + surround:
                if isinstance(item, Node):
                    nodes.append({'name': item.name, 'id': item.id, 'property': item.properties, 'type': item.type,
                                  'is_ans': item.ans_type})
                if isinstance(item, Edge):
                    edges.append({'source': item.src, 'target': item.tgt, 'name': item.name, 'is_ans': item.ans_type})

            nodes = sorted(nodes, key=lambda x: x['id'], reverse=False)
            # pdb.set_trace()
            ans = self.answer_generation(res_type, ans.split(',')[0], cand_name, name, rel, ans_ent)
            ret['answers'].append({'answer': ans, 'nodes': nodes, 'edges': edges})

        return ret

    def answer_generation(self, res_type, ans, cand_name, ent_name, rel, ent_node):
        ans_name = ent_name if fuzz.token_sort_ratio(' '.join(cand_name), ' '.join(ent_name)) < 80 else cand_name
        airline_ans = ''
        if '航司代码' in ent_node.properties.keys():
            airline_value = ent_node.properties['航司代码']
            airline_ans = '，办理%s航司业务' % airline_value
        if res_type == 'list' or rel is None or len(rel) == 0:
            # 列举类，找出时间和地点
            time_prop, time_prop_value, loc_prop, loc_prop_value, price_prop, price_prop_value = '时间', '暂无', '地点', '暂无', '价格', '暂无'
            tel_prop, tel_prop_value = '电话', '暂无'
            for prop in ent_node.properties.keys():
                if '时间' in prop:
                    time_prop, time_prop_value = prop, ent_node.properties[prop]
                if '地点' == prop:
                    loc_prop, loc_prop_value = prop, ent_node.properties[prop]
                if '价格' in prop or '收费' in prop:
                    price_prop, price_prop_value = prop, ent_node.properties[prop]
                if '电话' in prop or '联系' in prop:
                    tel_prop, tel_prop_value = prop, ent_node.properties[prop]


            # ans = '您好，机场内有%s，办理%s航司业务，位置在%s，营业时间是%s，价格为%s，电话是%s' % (ans_name, airline_value, loc_prop_value, time_prop_value, price_prop_value, tel_prop_value)

            # 加话术
            ans = '您好，机场内有%s，位置在%s，营业时间是%s，价格为%s，电话是%s' % (ans_name, loc_prop_value, time_prop_value, price_prop_value, tel_prop_value)
            ans = ans + airline_ans
            # if (time_prop_value == '' or time_prop_value == '暂无') and (loc_prop_value == '' or loc_prop_value == '暂无'):
            #     ans = '您好，机场内有%s' % (ans_name)
            # elif time_prop_value == '' or time_prop_value == '暂无':
            #     ans = '您好，机场内有%s，位置在%s' % (ans_name, loc_prop_value)
            # elif loc_prop_value == '' or loc_prop_value == '暂无':
            #     ans = '您好，机场内有%s，营业时间是%s' % (ans_name, loc_prop_value)
            # else:
            #     ans = '您好，机场内有%s，位置在%s，营业时间是%s' % (ans_name, loc_prop_value, time_prop_value)
            return ans

        if rel is None or len(rel) == 0:
            # 无属性，暂时没有话术
            return ans

        prop = list(rel.keys())[0]
        prop_val = rel[prop]
        # 话术生成
        if '地点' in prop:
            ans = '您好，%s在%s' % (ans_name, prop_val)
        elif '时间' in prop:
            ans = '您好，%s的服务时间是%s' % (ans_name, prop_val)
        elif '客服电话' == prop or '联系电话' == prop or '联系方式' == prop:
            ans = '您好，%s的客服电话是%s' % (ans_name, prop_val)
        elif prop == '手续费' or '价格' in prop or '收费' in prop:
            ans = '您好，%s的收费标准是%s' % (ans_name, prop_val)
        else:
            ans = '您好，' + ans
        ans = ans + airline_ans
        return ans


def decorate(qa_res):
    res = []
    for r in qa_res:
        rest, neoId, name, rela, rst, score, score_entity = r
        rel = list(rela.keys())[0]
        val = rela[rel]
        source = {'name': name, 'label': 'NONE',
                  'neoId': neoId}
        if val:
            target = {'name': val, 'label': '属性值', 'neoId': None}
            path = [[source, rel, target]]
        else:
            target = None
            path = []
        res.append({'header': source, 'tailer': target,
                    'available_words': [], 'path': path, 'score': score, 'res': rest})
    return res


if __name__ == "__main__":
    driver = AsyncNeoDriver.AsyncNeoDriver()
    # server_address=r'http://10.1.1.30:7474',
    # entity_label='Instance1')
    linker = EntityLinking.Linker(driver=driver)
    qahelper = QAHelper(driver, linker)
    questions = ['姚明的女儿是谁?', '姚明出生在哪？', '你知道姚明是哪里人吗？', '北京航空航天大学在哪？']
    # questions = ['打火机可以携带吗？', '机场有婴儿车可以租用吗', '机场有轮椅可以租用吗', '停车场怎么收费','停车场费用怎么样？','停车场一个小时多少钱？','停车场多少钱？']
    for q in questions:
        res = qahelper.qa(q)
        print(res[:10])
    del driver
