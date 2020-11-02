import logging
import traceback
from functools import lru_cache
from numpy import mean
import pandas as pd
import time


from fuzzywuzzy import fuzz

from ..common import AsyncNeoDriver
from ..common import Segment as seg
from ..config import config
from ..linking import EntityLinking
from .ConstraintExtractor import ConstraintExtractor
from .ListQuestion import check_list_questions
from .RelExtractor import BertRelExtractor, MatchRelExtractor


def load_stopwords(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        stopwords = {line.strip() for line in f}
    return stopwords


stopwords = load_stopwords(config.STOP_WORD_PATH)
aerospace_lexicons = set()
with open(config.AEROSPACE_LEXICON_PATH, 'r') as f:
    for line in f:
        aerospace_lexicons.add(line.split()[0])
# 问题中包含民航高频关键实体词的阈值空值
max_sent_len_contain_freq = 12    # 句子最短长度
max_threshold_contain_freq = 1.8  # final_score 在[1, 1.8]之间
min_threshold_contain_freq = 1

logger = logging.getLogger('qa')


def replace_word(sent, stop_list, rpl_list, special_rules):
    # 过滤词汇
    for stop in stop_list:
        sent = sent.replace(stop, '')

    # 替换实体或属性名称
    for replace in rpl_list:
        sent = sent.replace(replace[0], replace[1])

    code_word = ['三字']  # IATA是航司的二字码，机场的三字码
    for word in code_word:
        if word in sent:
            en_name = 'IATA' if '机场' in sent else 'ICAO'
            sent = sent.replace(word,en_name)

    # 航司全称是公司名称，机场的全称是中文名
    if '全称' in sent:
        property = '中文名' if '机场' in sent else '公司名称'
        sent = sent.replace('全称', property)

    for rules in special_rules:
        keywords = rules[0]
        name = rules[1]
        flag = True
        for word in keywords:
            if flag and word not in sent:
                flag = False
                break
        if flag:
            sent = sent+ name
    return sent


# 取列表中不被其他字符串包含的字符串 如：['IATA','IA','AT','中国航空公司','国航']，结果是['IATA', '中国航空公司']
def remain_max_string(name_list,sent_cut):
    res = []
    all_list = name_list+sent_cut
    for i in all_list:
        item = all_list.copy()
        item.remove(i)
        flag = True
        for k in item:
            if i in k:
                flag = False
                break
        if flag:
            res.append(i)
    res = list(set(res) - set(sent_cut))
    return res


class QA():
    stop_list = ['吗', '里面']
    rpl_list = [('在哪', '的地点所在城市'),
                ('哪里', '地点所在城市'),
                ('在哪个城市', '所在城市'),
                ('航司', '航空公司'),
                ('属于', '在'),
                ('vip', '贵宾'),
                ('我国', '国内'),
                # ('国内', '中国'),
                ('哪个航空公司', '公司名称'),
                ('哪家航空公司', '公司名称'),
                ('是什么航空', '公司名称'),
                ('什么公司', '公司名称'),
                ('哪个公司', '公司名称'),
                ('每小时多少公里', '速度'),
                ('飞多少小时？', '飞行时间'),
                ('坐多少人', '容量'),
                ('失事的几率', '事故率'),
                ('载客', '客运量'),
                ('英文是', '外文名'),
                ('英文名是', '外文名'),
                ('英语', '外文名'),
                ('英文', '外文名'),
                ('飞多高', '高度'),
                ('有多大', '面积'),
                ('分为', '分类'),
                ('有多重', '重量'),
                ('烧什么油', '燃料'),
                ('个部分', '组成'),
                ('修好', '故障'),
                ('官方网站', '官网'),
                ('公司性质', '公司类型'),
                ('播音', '波音公司'),
                ('二字', 'IATA代码'),
                ('四字', 'ICAO代码'),
                ('功用', '作用'),
                ('什么用', '作用'),
                ('功能', '作用'),
                ('不足', '缺点'),
                ('优势', '优点'),
                ('哪年成立的', '成立时间'),
                ('什么时候建的', '成立时间'),
                ('什么时间建的', '成立时间'),
                ('建成', '成立时间'),
                ('什么时候成立', '成立时间'),
                ('时候', '时间'),
                ('机型', '飞机型号'),
                ('简写', '简称'),
                ('是怎么回事', '定义'),
                ('什么意思', '定义'),
                ('什么是', '定义'),
                ('什么叫', '定义'),
                ('是什么', '定义'),
                ('指什么', '定义'),
                ('什么叫做', '定义')]
    special_rules = [
        [("航班", "不正常"), "不正常航班"],
        [("机场", "ICAO"), "机场三字码"],
        [("公司", "价格"), "造价"],
    ]

    def __init__(self):
        self.driver = AsyncNeoDriver.get_driver()
        self.bert_linker = EntityLinking.BertLinker()
        self.commercial_linker = EntityLinking.CommercialLinker()
        self.rule_linker = EntityLinking.RuleLinker()
        self.match_extractor = MatchRelExtractor()
        self.bert_extractor = BertRelExtractor()
        self.constr_extractor = ConstraintExtractor()
        self.frontend = FrontendAdapter()


    def preprocess(self, sent):
        """
            返回(处理后问句sent_replaced, 分词结果sent_cut, 词性标注 pos_tag)
            问句预处理：
            1. 分词前替换规则
            2. 分词, pos tag
            3. 分词后替换规则：停用词，特殊词
        """
        # 分词前替换规则：过滤掉停用词，替换抽象词汇    如'在哪儿'表示'的地点'
        if "有机场吗" in sent:
            sent = sent.replace("有机场吗",'有哪些机场')
        sent_replaced = replace_word(sent, QA.stop_list, QA.rpl_list, QA.special_rules)
        # 分词
        sent_cut, pos_tag = seg.pos_cut(sent_replaced)  # pos_cut：list(zip(*pseg.cut(sent))) pos_tag：词性标注
        # 特殊词处理
        sent_cut_pro = []
        pos_tag_pro = []
        for word, tag in zip(sent_cut, pos_tag):
            if len(word) > 2 and word[-1] == '费':
                sent_cut_pro.append(word[:-1])  # 去除'费'，增加cut集合 费用
                sent_cut_pro.append('费用')
                pos_tag_pro.append(tag)
                pos_tag_pro.append(tag)  # 费用与word[-1] 同词性
            else:
                sent_cut_pro.append(word)
                pos_tag_pro.append(tag)

        if '机场' in sent and '面积' in sent_replaced and '航站楼面积' not in sent_cut_pro:
            sent_cut_pro.append('航站楼面积')
            pos_tag_pro.append('n')
        if '成立' in sent_replaced and '时间' in sent_replaced and '成立时间' not in sent_cut_pro:
            sent_cut_pro.append('成立时间')
            pos_tag_pro.append('n')

        ent_list = []
        for word in aerospace_lexicons:
            word = str(word)
            if (word in sent or word in sent_replaced) and word not in sent_cut_pro:
                ent_list.append(word)
        ent_list = remain_max_string(ent_list,sent_cut_pro)
        sent_cut_pro += ent_list
        pos_tag_pro += ['n' for i in range(len(ent_list))]

        # 去除英文分词错误
        # 如：分词中既有IATA，又有IA。如果删除句子中的IATA后，不存在IA字符串，说明IA是分词错误，需要删除
        for word, sup_word in config.SPECIAL_ENGLISH_IN_SEGMENT.items():  # eg：{'IATA':['AT', 'IA', 'ATA', 'TA']}
            if word in sent_cut_pro and any([sup in sent_cut_pro for sup in sup_word]):  # 分词中既有IATA，又有IA或ATA
                temp = sent.replace(word, "")
                if any([sup in temp for sup in sup_word]) is False:  # 删除IATA后，句子中不再有IA或ATA，删除IA和ATA
                    for sup in sup_word:
                        if sup in sent_cut_pro:
                            pos_tag_pro.pop(sent_cut_pro.index(sup))
                            sent_cut_pro.remove(sup)

        # 去除停用词(特殊词处理后)
        sent_cut = []
        pos_tag = []
        for word, tag in zip(sent_cut_pro, pos_tag_pro):
            if word not in stopwords and word != ' ' and word not in sent_cut:
                # 如何某个词 in 实体名字符串 中，则算作不精准分词，删除
                str_in_ent_name = False
                for ent in ent_list:
                    if word in ent and word != ent:
                        str_in_ent_name = True
                        break
                if str_in_ent_name:
                    continue

                sent_cut.append(word)
                pos_tag.append(tag)
        return sent_replaced, sent_cut, pos_tag

    # @lru_cache(maxsize=128)
    def link(self, sent, sent_cut, pos_tag, is_list):
        def merge_link_res(res, id2linked_ent):
            for linked_ent in res:
                neoId = linked_ent['id']
                if neoId in id2linked_ent:
                    if linked_ent['source'] not in id2linked_ent[neoId]['source']:
                        id2linked_ent[neoId]['score'] += linked_ent['score']
                        id2linked_ent[neoId]['source'] += ' ' + \
                            linked_ent['source']
                else:
                    id2linked_ent[neoId] = linked_ent
                    link_res.append(linked_ent)

        # bert_res = self.bert_linker.link(sent)  # zsh 航空尝试问答无训练数据，暂不用
        # print('bert链接: %s' % str(bert_res))

        # commercial_res = self.commercial_linker.link(sent, sent_cut)  # 注释掉 zsh
        # print('商业链接: %s' % str(commercial_res))

        rule_res = self.rule_linker.link(sent, sent_cut, pos_tag, is_list)  # bert embedding to fuzzy match entities
        logger.debug('规则链接: %s' % str(rule_res))
        link_res = rule_res

        # merge commecial res
        # merge_link_res(commercial_res, id2linked_ent) # 注释掉 zsh
        # 获取具体实体
        link_res_extend = []
        for linked_ent in link_res:
            ent = linked_ent['ent']
            if not is_list and ent['entity_label'] in ['SubGenre', 'SubGenre_child', 'Genre']:  #  and ent['name'] not in EntityLinking.exception_subgenre)\
                # if self.rule_linker.is_special_entity(ent):  # 注释掉 zsh
                #     continue
                instances = self.get_instances(
                    ent['name'], ent['entity_label'], 'Instance')
                if instances is not None:
                    link_res_extend.extend([{
                        'ent': e,
                        'mention': linked_ent.get('mention', e['name']),
                        'id':e['neoId'],
                        'score':linked_ent['score'],
                        'source':linked_ent['source']+' sub',  # source 是指是bert结果，还是规则匹配结果
                    } for e in instances])
            elif is_list or ent['entity_label'] == 'Instance':  # or ent['name'] in EntityLinking.exception_subgenre:
                link_res_extend.append(linked_ent)
        link_res_extend.sort(key=lambda x: x['score'], reverse=True)

        id2linked_ent = {}
        id2linked_ent = {linked_ent['id']: linked_ent
                         for linked_ent in link_res if linked_ent['id'] not in id2linked_ent or (linked_ent['id'] in id2linked_ent and id2linked_ent[linked_ent['id']]['score'] < linked_ent['score'])}

        link_res_extend = []
        for k,v in id2linked_ent.items():
            link_res_extend.append(v)

        return link_res_extend, id2linked_ent

    def get_instances(self, parent_name, parent_label, instance_label):
        instances = self.driver.get_genres_by_relation(
            parent_label, instance_label, parent_name, reverse=True).result()
        return instances

    def extract_rel(self, sent: str, sent_cut: list, link_res, is_list=False):
        # 抽取关系
        rel_match_res, properties = [], []
        for linked_ent in link_res:
            ent = linked_ent['ent']
            # 匹配关系
            rel_matched = self.match_extractor.extract_rel(
                sent_cut, linked_ent, is_list, limits=None,
                thresh=config.prop_ths)
            if is_list and rel_matched and not properties:
                properties = [item['rel_name'] for item in rel_match_res]
            rel_match_res += rel_matched
            # bert 提取关系
            """ 
            rel_bert_res = self.bert_extractor.extract_rel(sent, linked_ent)
            for bert_rel in rel_bert_res:
                for match_rel in rel_matched:
                    if bert_rel['rel_name'] == match_rel['rel_name']:
                        match_rel['rel_score'] += 0.3
                        match_rel['rel_source'] += ' bert'
                    else:
                        rel_match_res.append(bert_rel)
            """
        return rel_match_res, properties


    def match_constraint(self, qa_res, constraint, id2linked_ent):
        non_empty_constr = [(constr_name, constr_val) for constr_name, constr_val in constraint.items()
                            if constr_val != '' and constr_val is not None]
        is_matched = False
        for i,ans in enumerate(qa_res):
            linked_ent = id2linked_ent[ans['id']]
            match_res = None
            try:
                match_res = self.constr_extractor.match_constraint(
                    constraint, linked_ent)
            except:
                traceback.print_exc()

            ans['constr_score'] = 0
            if match_res is not None and len(match_res) != 0:
                logger.info('限制匹配结果: '+str(match_res))
                for constr, is_match in match_res.items():
                    if is_match:
                        ans['constr_score'] += 0.3
                        ans['constr_name'] = constr
                        ans['constr_val'] = linked_ent['ent'][constr]
                    else:
                        ans['constr_score'] += -0.2
            else:
                for constr_name, constr_val in non_empty_constr:
                    if fuzz.UQRatio(constr_val[0], ans['entity']) >= 60:
                        ans['link_score'] -= 0.3
                    else:
                        ans['constr_score'] += -0.2
            if ans['constr_score']>0:
                is_matched = True
            qa_res[i] = ans
        return qa_res,is_matched

    def generate_natural_ans(self, qa_res: dict, id2linked_ent, flag):
        linked_ent = id2linked_ent[qa_res['id']]
        ent = linked_ent['ent']
        ent_name = ent['name']
        cand_name = qa_res['mention']
        # ent_name = qa_res['mention']  # 答案术语用问题中的mention回答。【考虑到mention会提取错误的问题，暂时把该功能去除】
        true_entity = qa_res['entity']
        natural_ans = ''
        ans_name = ent_name
        airline_ans = ''
        if flag == 0:
            natural_ans = '您好，%s有%s' % (ans_name, true_entity)
            natural_ans += airline_ans
            return natural_ans
        else:
            if 'rel_score' not in qa_res:
                ans_list = []
                ans_list.append('您好，机场内有%s' % ans_name)
                # 列举类，找出时间和地点
                for rel, rel_val in ent.items():
                    if '时间' in rel:
                        ans_list.append('营业时间是%s' % rel_val)
                    if '地点' == rel:
                        ans_list.append('位置在%s' % rel_val)
                    # if '价格' in rel or '收费' in rel:
                    #     ans_list.append('价格为%s' % rel_val)
                    if '电话' in rel or '联系' in rel:
                        ans_list.append('电话是%s' % rel_val)

                # ans = '您好，机场内有%s，办理%s航司业务，位置在%s，营业时间是%s，价格为%s，电话是%s' % (ans_name, airline_value, loc_prop_value, time_prop_value, price_prop_value, tel_prop_value)

                # 加话术
                # if (time_prop_value == '' or time_prop_value == '暂无') and (loc_prop_value == '' or loc_prop_value == '暂无'):
                #     ans = '您好，机场内有%s' % (ans_name)
                # elif time_prop_value == '' or time_prop_value == '暂无':
                #     ans = '您好，机场内有%s，位置在%s' % (ans_name, loc_prop_value)
                # elif loc_prop_value == '' or loc_prop_value == '暂无':
                #     ans = '您好，机场内有%s，营业时间是%s' % (ans_name, loc_prop_value)
                # else:
                #     ans = '您好，机场内有%s，位置在%s，营业时间是%s' % (ans_name, loc_prop_value, time_prop_value)
                if not natural_ans:
                    prop_desc = []
                    for prop in ent.keys():
                        if prop in ["name","名称","neoId","entity_label"]:
                            continue
                        info = ent[prop]
                        if info[0] == '[' and info[-1] == ']':
                            info = eval(info)
                            info = str("、".join(info))
                        desc = "%s是%s" % (prop, info)
                        if desc not in prop_desc:
                            prop_desc.append(desc)
                    natural_ans = '您好，您询问的是%s，具体信息如下：%s' % (ent_name, "，".join(prop_desc))
                    return natural_ans
                else:
                    natural_ans = '，'.join(ans_list)
                    natural_ans += airline_ans
                    return natural_ans
            else:
                rel = qa_res['rel_name']
                rel_val = qa_res['rel_val']
                if rel_val == ans_name:  # 答案术语使用mention，如：FM是mention，答案为FM的公司名称是上海航空股份有限公司
                    ans_name = qa_res['mention']
                # 话术生成
                if '地点' in rel:
                    natural_ans = '您好，%s在%s' % (ans_name, rel_val)
                else:
                    natural_ans = '您好，%s的%s是%s' % (ans_name, rel, rel_val)
                natural_ans += airline_ans
                return natural_ans

    def rank_ans(self, qa_res, sent):
        for ans in qa_res:
            if ans.get('rel_score', 0) == 0 and ans['mention'] == sent:
                # 判断问题中是否只有实体，不包含属性。如果只包含实体，则不进行权值计算，只取实体链接得分
                ans['final_score'] = ans['link_score']
            else:
                # 各步骤分值相加
                ans['final_score'] = ans.get('link_score', 0) * 0.75 + ans.get('rel_score', 0) * 0.25  # ans.get('constr_score', 0) + \
            # # 各步骤分值相乘
            # ans['final_score'] = ans.get('constr_score', 1) * \
            #     ans['link_score'] * \
            #     ans.get('rel_score', 1)
        qa_res.sort(key=lambda ans: ans['final_score'], reverse=True)
        # score相同，mention是问题子串的排在前面
        for i in range(len(qa_res)):
            if qa_res[i]['mention'] in sent:
                if qa_res[i]['final_score'] == qa_res[i-1]['final_score']:
                    long_metion = len(qa_res[i]['mention']) > len(qa_res[i-1]['mention'])
                    long_relation = 'rel_name' in qa_res[i] and 'rel_name' in qa_res[i - 1] \
                                    and len(qa_res[i]['rel_name']) > len(qa_res[i - 1]['rel_name'])
                    pre_is_definition = 'rel_name' in qa_res[i - 1] and qa_res[i - 1]['rel_name'] == "定义"
                    if qa_res[i -1]['mention'] in sent and (long_metion or long_relation or pre_is_definition):
                        temp = qa_res[i].copy()
                        qa_res[i] = qa_res[i-1]
                        qa_res[i-1] = temp
        return qa_res

    def answer(self, sent):
        """
            问答流程：
             1. 使用替换规则修改问句并分词
             2. 提取限制条件：时间、地点、航空公司等
             3. 三种linker提取实体，合并排序：bert 商业 规则（航空常识只用到了规则）
             4. 判断是否列举型
             5. 非列举型匹配关系
             6. 如果没有匹配到的关系，且实体识别分值较高，算作列举
             7. 匹配限制
             8. 答案排序
             9. 生成自然语言答案
        """
        logger.info('question :%s' % sent)
        # 1. 使用替换规则(预处理)
        # sent = replace_word(sent, self.stop_list, self.rpl_list)
        # sent_cut = LTP.customed_jieba_cut(sent, cut_stop=True)
        sent, sent_cut, pos_tag = self.preprocess(sent)  # 处理后问句sent_replaced, 分词结果sent_cut, 词性标注 pos_tag
        logger.debug('cut :' + str(sent_cut))

        # 2. 提取限制条件:时间、地点、航空公司等
        constr_res = self.constr_extractor.extract(sent)
        logger.debug('限制: ' + str(constr_res))

        # 3. 处理列举类型：问句询问某实体是否存在或存在数量
        is_list = check_list_questions(sent)  # 提及具体的属性（如时间等）则False
        logger.debug('是否列举: ' + str(is_list))

        # 4. link: 使用多种linker，合并结果
        link_res, id2linked_ent = self.link(sent, sent_cut, pos_tag, is_list)
        logger.debug('链接结果: ' + str(link_res[:10]))

        # 5. 匹配关系 extract relations, match+bert
        rel_match_res,properties = self.extract_rel(sent, sent_cut, link_res, is_list)
        logger.debug('关系匹配: ' + str(rel_match_res[:10]))
        if is_list and rel_match_res:
            property_name = properties
        else:
            property_name = None

        # 6. 如果没有匹配到的关系，且实体识别分值较高，算作列举
        qa_res = []
        LINK_THRESH = 0.8
        REL_THRESH = 0.72
        match_rel_ent_ids = {e['id'] for e in rel_match_res}
        if is_list and len(link_res) > 0:
            rel_match_res_id = [item['id'] for item in rel_match_res]
            max_rel_score = 0
            for linked_ent in link_res:
                if linked_ent['id'] not in rel_match_res_id:
                    qa_res.append({
                        'id': linked_ent['id'],
                        'mention': linked_ent.get('mention', linked_ent['ent']['name']),
                        'entity': linked_ent['ent']['name'],
                        'link_score': linked_ent['score'],
                    })
                else:
                    for item in rel_match_res:
                        if item['entity'] != linked_ent['ent']['name']:
                            continue
                        if item['rel_score'] >= REL_THRESH and item['link_score'] >= LINK_THRESH:
                            qa_res.append(item)
                            if item['rel_score'] > max_rel_score:
                                max_rel_score = item['rel_score']
            # qa_res.extend([{
            #     'id': linked_ent['id'],
            #     'mention': linked_ent.get('mention', linked_ent['ent']['name']),
            #     'entity': linked_ent['ent']['name'],
            #     'link_score': linked_ent['score'],
            # } for linked_ent in link_res])
        else:
            # 删掉关系匹配的结果中，匹配结果不高的部分
            qa_res.extend([rel_res for rel_res in rel_match_res
                           if rel_res['rel_score'] >= REL_THRESH and
                           rel_res['link_score'] >= LINK_THRESH
                           ])
            for linked_ent in link_res:
                if linked_ent['id'] not in match_rel_ent_ids \
                        and linked_ent['score'] >= LINK_THRESH:

                    qa_res.append({
                        'id': linked_ent['id'],
                        'mention': linked_ent.get('mention', linked_ent['ent']['name']),
                        'entity': linked_ent['ent']['name'],
                        'link_score': linked_ent['score']
                    })

                    # if '定义' in linked_ent['ent'].keys():
                    #     qa_res.append({
                    #         'id': linked_ent['id'],
                    #         'mention': linked_ent.get('mention', linked_ent['ent']['name']),
                    #         'entity': linked_ent['ent']['name'],
                    #         'rel_name': '定义',
                    #         'rel_val': linked_ent['ent']['定义'],
                    #         'link_score': linked_ent['score'],
                    #         'rel_score': 0.5,
                    #         'rel_source': 'match'
                    #     })
                    # else:
                    #     qa_res.append({
                    #         'id': linked_ent['id'],
                    #         'mention': linked_ent.get('mention', linked_ent['ent']['name']),
                    #         'entity': linked_ent['ent']['name'],
                    #         'link_score': linked_ent['score']
                    #     })

        # 7. 匹配机场本身相关的问题
        """
        if len(qa_res) == 0 and '机场' in sent:
            airport_ent = self.driver.get_entities_only_by_name(
                config.airport.name)[0]
            linked_airport = {
                'ent': airport_ent,
                'mention': airport_ent['name'],
                'id': airport_ent['neoId'],
                'score': 1,
                'source': 'airport',
            }
            id2linked_ent[linked_airport['id']] = linked_airport
            airport_rel_match = self.extract_rel(
                sent, sent_cut, [linked_airport])
            qa_res.extend(airport_rel_match)
        """

        # 8. 匹配限制
        if any(map(lambda x: x != '', constr_res.values())):
            qa_res,is_matched = self.match_constraint(qa_res, constr_res, id2linked_ent)
            if is_matched:
                qa_res = [ent for ent in qa_res if ent['constr_score']>0]
            qa_res = sorted(qa_res,key=lambda x:x['constr_score'],reverse=True)
            logger.debug('限制结果: ' + str(qa_res))

        if is_list:
            # 10. 生成自然语言答案
            filtered_qa_res = []
            ids = {item['id']: 0 for item in qa_res}
            cnt = len(ids)

            ans_all = []
            if property_name:
                for i, res in enumerate(qa_res):
                    if 'rel_val' not in res:
                        res['rel_val'] = '暂无'
                        res['rel_name'] = "和".join(property_name)
                    if res['entity'] == qa_res[i - 1]['entity']:
                        ans_all[-1] += "  {}是{}".format(res['rel_name'], res['rel_val'])
                    else:
                        ans_all.append("{}的{}是{}".format(res['entity'], res['rel_name'], res['rel_val']))
            else:
                for res in qa_res:
                    ans_all.append(res['entity'])
            ans_all_head = []
            ans_all_behind = []
            for ans_str in ans_all:
                if "暂无" in ans_str:
                    ans_all_behind.append(ans_str)
                else:
                    ans_all_head.append(ans_str)
            ans_all = ans_all_head + ans_all_behind
            natural_ans = "、".join(ans_all)
            _res = dict()
            _res['natural_ans'] = natural_ans
            _res['final_score'] = qa_res[0]['link_score'] * 0.75 + max_rel_score * 0.25
            filtered_qa_res.append(_res)
            logger.info('自然语言答案: ' + str(natural_ans))
            return self.frontend.decorate(filtered_qa_res, cnt)
        else:
            # 9. 答案排序
            qa_res = self.rank_ans(qa_res,sent)
            qa_res = qa_res[:10]
            logger.debug('答案: ' + str(qa_res))

            # 10. 生成自然语言答案
            natural_ans = []
            filtered_qa_res = []
            for res in qa_res:
                n_ans = self.generate_natural_ans(res, id2linked_ent, 1)
                if n_ans not in natural_ans:
                    res['natural_ans'] = n_ans.replace("\n",'')
                    natural_ans.append(n_ans)
                    filtered_qa_res.append(res)
            logger.info('自然语言答案: ' + str(natural_ans))

            # # 11 对包含常见民航词语（飞机、飞行、航空）的问题进行阈值控制  阈值待调整 1101
            # aver_final_score = mean([item['final_score'] for item in filtered_qa_res[:3]])
            # if self.contain_freq_word(sent) \
            #     and min_threshold_contain_freq <= aver_final_score <= max_threshold_contain_freq \
            #     and len(sent) > max_sent_len_contain_freq:
            #     filtered_qa_res = []
            return self.frontend.decorate(filtered_qa_res[:3], -1)  # -1表示非列举类问题

    def contain_freq_word(self, question):
        for word in config.FREQUENT_WORDS:
            if word in question:
                return True
        return False


class FrontendAdapter():
    def __init__(self):
        self.driver = AsyncNeoDriver.get_driver()

    def decorate_one(self, qa_res,cnt):
        """
        Args:
            qa_res: {
                'id': '', 主题实体id
                'mention': '',
                'entity': '',
                'rel_name': '', 关系名
                'rel_val'： '', 关系值
                'constr_name': '', 限制名
                'constr_val': '', 限制值
                'natural_ans': '', 自然语言答案
           }

        Return:
            decoreated: {
                'answers': '',
                'nodes': [
                    {
                        'name':
                        'id':
                        'property':{},
                        'type':
                        'is_ans':
                    }
                ],
                'edges': [
                    {
                        'source': 111,
                        'target': 112,
                        'name': '',
                        'is_ans': ''
                    }
                ],
            }
        """
        answer = qa_res['natural_ans']
        score = qa_res['final_score']
        nodes = []
        edges = []
        global_id = 0

        def make_node(neoId, name, is_ans):
            nonlocal global_id
            if neoId >= 0:
                ent_type = self.driver.get_label_by_id(neoId).result()[0]
                ent_prop = self.driver.get_props_by_id(neoId, ent_type).result()
            else:
                ent_type = 'Property'
                ent_prop = {}
            res = {
                'name': name,
                'id': global_id,
                'neoId': neoId,
                'property': ent_prop,
                'type': ent_type,
                'is_ans': is_ans
            }
            global_id += 1
            return res

        def make_edge(src, tgt, name, is_ans):
            return {
                'source': src['id'],
                'target': tgt['id'],
                'name': name,
                'is_ans': is_ans,
            }

        # 列举类
        list_flag = False
        if cnt != -1:
            list_flag = True
        if list_flag:
            return {
                'answers': '搜寻到的结果数:{}  {} '.format(cnt,answer),
                'score': qa_res['final_score']
            }

        topic_ent = make_node(int(qa_res['id']), qa_res['entity'], 'answer_entity')
        # logger.debug('qa_res: ' + str(qa_res))
        if 'rel_val' in qa_res:
            ans_ent = make_node(-1, qa_res['rel_val'], 'answer_property_value')
            nodes.append(topic_ent)
            nodes.append(ans_ent)

            edges.append(make_edge(topic_ent, ans_ent, qa_res['rel_name'], 'answer_property'))
        else:
            nodes.append(topic_ent)

        if 'constr_name' in qa_res:
            constr_ent = make_node(-1, qa_res['constr_val'], 'answer_restriction_value')
            nodes.append(constr_ent)
            edges.append(make_edge(topic_ent, constr_ent, qa_res['constr_val'],'answer_restriction'))

        return {
            'answers': answer,
            'score': score,
            'nodes': nodes,
            'edges': edges
        }

    def decorate(self, all_qa_res, cnt):
        return [self.decorate_one(res,cnt) for res in all_qa_res]


def test_qa():
    qa = QA()
    # questions = ['昆明机场是什么时候建立的？', '长水机场在什么地方？', '东航的值机柜台在哪？', '有什么餐厅吗？', '机场哪里有吃饭的地方？', '机场有麦当劳吗？', '打火机可以携带吗？', '机场有婴儿车可以租用吗', '机场有轮椅可以租用吗',
    #              '停车场怎么收费', '停车场费用怎么样？', '停车场一个小时多少钱？', '停车场多少钱？']
    questions = ['过了安检里面有没有书吧？']
    for q in questions:
        qa.answer(q)


if __name__ == "__main__":
    test_qa()
