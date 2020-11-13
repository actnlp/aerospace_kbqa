import re, os
import pandas as pd
from fuzzywuzzy import fuzz
from ..config import config
from ..config.config import AIRPORT_LEXICON_PATH


def load_info(filename):
    words = []
    if os.path.isfile(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words.append(line.strip('\n'))
    return words


class ConstraintExtractor():
    remove_prop = {'name', 'subname', 'description',
                   'label', 'taglist', 'neoId', 'keyId', 'id', 'score', 'rel', 'hidden', 'textqa答案', 'entity_label'}

    def __init__(self):
        city_sets = load_info(config.CITY)
        province_sets = load_info(config.PROVINCE)
        loc_sets = load_info(config.HEADQUATERS_LOCATION)
        self.loc_words = city_sets + province_sets + loc_sets

    def preprocess(self, sent):
        digits = [('一', '1'), ('二', '2'), ('三', '3'), ('四', '4'),
                  ('五', '5'), ('六', '6'), ('七', '7'), ('八', '8'), ('九', '9')]
        if '十' in sent:
            pre_index = sent.index('十') - 1
            if pre_index >= 0 and sent[pre_index] in ''.join([x[0] for x in digits]):
                digits.append(('十', '0'))
        for digit in digits:
            sent = sent.replace(digit[0], digit[1])
        return sent

    def check_time(self, sent):
        flags = ["客服时间", "建造时间", "竣工", "发明时间", "成立", "停运", "立项", "改造", "服务时间", '服役', "通航时间", "运营时间", "撤销", "认证", "交付",
                 "立项", "首飞", "服役", "定型时间", "研发时间", "退役时间", "投入运营", "停产", "生产时间", "引进时间", "设计时间", "展出时间", "首航", "重组",
                 "破产"]
        ret = []
        for time_flag in flags:
            if time_flag in sent:
                ret.append(time_flag)
        return ret

    def check_location(self, sent):
        flags = ["城市", "省", "国家", "地区", "地点", "地方", '位置', '国外'] + self.loc_words
        flags = list(set(flags))
        ret = []
        for loc_flag in flags:
            if loc_flag in sent:
                ret.append(loc_flag)
        return ret

    def check_ffp(self, sent):
        flags = ['俱乐部'] + load_info(config.FFP)
        ret = []
        for flag in flags:
            if flag in sent:
                ret.append(flag)
        return ret

    def check_airline(self, sent):
        airlines = ['南航', '海南航空', '上海航空', '重庆航空', '深圳航空', '西部航空', '东方航空', '金鹿航空', '新华航空', '光大银行', '国际航空', '南方航空', '国航',
                    '海航', '奥凯航空', '祥鹏', '祥鹏航空', '昆明航空', '泰国航空', '山东航空', '东航',
                    '川航', '西藏航空', '春秋航空', '美国航空', '红土航空', '成都航空', '福州航空', '长龙航空', '老挝航空', '立荣航空', '瑞丽航空', '亚洲航空',
                    '四川航空', '吉祥航空', '厦门航空', '东海航空', '大韩航空', '联合航空', '首都航空', '天津航空', '百事特']
        ret = []
        for airline in airlines:
            if airline in sent:
                ret.append(airline)

        return ret

    # 低成本廉价航空判断
    def check_business_model(self, sent):
        prices = ['低成本', '廉价']
        ret = []
        for price in prices:
            if price in sent:
                ret.append(price)
        return ret

    # 航空联盟
    def check_alliances(self, sent):
        flags = ['星空联盟', '天合联盟', '寰宇一家', '海航联盟']
        ret = []
        for flag in flags:
            if flag in sent:
                ret.append(flag)
        return ret

    # 国际机场
    def check_INT(self, sent):
        flags = ['国际机场']
        ret = []
        for flag in flags:
            if flag in sent:
                ret.append(flag)
        return ret

    # 航司类型:国有，民营等
    def check_company_type(self, sent):
        flags = ['国营', '国有企业', '私营', '民营']
        ret = []
        for flag in flags:
            if flag in sent:
                ret.append(flag)
        return ret

    def extract(self, sent):
        sent = self.preprocess(sent)  # 一～十 替换成 阿拉伯数字
        constr = {'时间': '', '地点': '', '中文名': '', '常旅客计划': '', '运营模式': '', '航空联盟': '','类型': ''}
        time_check = self.check_time(sent)
        loc_check = self.check_location(sent)
        INT_check = self.check_INT(sent)  # 国际
        ffp_check = self.check_ffp(sent)  # 常旅客计划
        business_model_check = self.check_business_model(sent)  # 运营模式
        alliances_check = self.check_alliances(sent)  # 航空联盟
        company_type_check = self.check_company_type(sent)  # 航司类型:国有，民营等

        if time_check is not None and len(time_check) > 0:
            constr['时间'] = time_check
        if loc_check is not None and len(loc_check) > 0:
            constr['地点'] = loc_check
        if INT_check is not None and len(INT_check) > 0:
            constr['中文名'] = INT_check
        if ffp_check is not None and len(ffp_check) > 0:
            constr['常旅客计划'] = ffp_check
        if business_model_check is not None and len(business_model_check) > 0:
            constr['运营模式'] = business_model_check
        if alliances_check is not None and len(alliances_check) > 0:
            constr['航空联盟'] = alliances_check
        if company_type_check is not None and len(company_type_check) > 0:
            constr['类型'] = company_type_check

        for check in [time_check, loc_check, INT_check, ffp_check, business_model_check, alliances_check, company_type_check]:
            if check is None:
                return None
        # if time_check is None and loc_check is None and cur_check is None and bank_check is None and airline_check is None and price_check is None:
        #     return None

        return constr

    def match_constraint(self, constraint: dict, linked_ent):
        # check whether limit is available
        exist_constr = []
        res_constr = {}
        ent = linked_ent['ent']
        match_constr = False
        for constr_name, constr_val in constraint.items():
            if constr_val != '' and constr_val is not None:
                # and fuzz.UQRatio(constr_val[0], ent['name']) < 50:
                for rel, rel_val in ent.items():
                    # rel 实体属性名  rel_val 实体属性值
                    if rel in self.remove_prop:
                        continue
                    rel_val = str(rel_val)
                    if (constr_name in rel
                        # or constr_name in rel_val
                        # or constr_val[0] in rel
                        # or constr_val[0] in rel_val):
                        and constr_val[0] in rel_val):
                        match_constr = True
                        res_constr[rel] = True
                        exist_constr.append((rel, constr_name))
        if not match_constr:
            return None

        # filter result
        time_pattern = re.compile(r'\d+[:, ：]')
        # for rel, constr_name in exist_constr:
        #     rel_val = ent[rel].lower()
        #     if '地点' in constr_name:
        #         for item in constraint['地点']:
        #             if item not in rel_val:
        #                 res_constr[rel] = False
        #     elif '时间' in constr_name:
        #         for item in constraint['时间']:
        #             #  ''' or item == '最早' or item == '最晚'''''
        #             if (item == '24小时' or item == '最早' or item == '最晚') and item not in rel_val:
        #                 res_constr[rel] = False
        #                 continue
        #             bg_ed = time_pattern.findall(rel_val)
        #             bg_ed = [int(x[:-1]) for x in bg_ed]
        #             if '时' not in item:
        #                 if len(bg_ed) == 2:
        #                     if not (bg_ed[0] < int(item) < bg_ed[1]):
        #                         res_constr[rel] = False
        #
        #     elif '币种' in constr_name or '银行' in constr_name or '航空公司' in constr_name or '价格' in constr_name:
        #         for item in constraint[constr_name]:
        #             if item not in rel_val:
        #                 res_constr[rel] = False

        return res_constr
