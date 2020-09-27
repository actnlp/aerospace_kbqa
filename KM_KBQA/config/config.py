import logging
import os

project_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_dir, 'data')
model_path = os.path.join(project_dir, 'models')

LEX_PATH = os.path.join(project_dir, 'lexicons')
STOP_WORD_PATH = os.path.join(LEX_PATH, 'stopwords.txt')
AEROSPACE_LEXICON_PATH = os.path.join(LEX_PATH, 'aerospace_lexicon.txt')
AIRPORT_LEXICON_PATH = os.path.join(LEX_PATH, 'air_lexicon.txt')
ENT_ALIAS_PATH = os.path.join(project_dir, 'res', 'ent_alias.txt')

####################################总部地点、省份、所在城市
HEADQUATERS_LOCATION = os.path.join(project_dir, 'res/list_txt', 'Headquaters_location.txt')
PROVINCE = os.path.join(project_dir, 'res/list_txt', 'province.txt')
CITY = os.path.join(project_dir, 'res/list_txt', 'city.txt')
####################################

ABSTRACT_DICT = {'丢': '失物招领', '取款机': 'atm', '存款机': 'atm', '存取款机': 'atm', '提款机': 'atm', '取钱': '银行', 'kfc': '肯德基', '加油': '加油站', '乘机手续': '值机柜台',
                 '住': '酒店', '住宿': '酒店', '吃': '餐厅', '吃饭': '餐厅', '餐饮': '餐厅', '开水': '饮水处', '剪发': '理发店', '剪头发': '理发店', '喂奶': '母婴室',
                 '热水': '饮水处', '插电': '充电', '换': ['货币兑换', '母婴室'], '兑换': '货币兑换', '厕所': '卫生间', '胶囊': '睡眠', '打包': '行李打包', '寄存': '行李寄存',
                 '换外币': '货币兑换', '打的士': '出租车', '打的': '出租车', '打车': '出租车', '的士': '出租车', '巴士': '机场公交', '大巴': '公交', '停': '停车场', '速运': '快递',
                 '速递': '快递', '配送': '快递', '上网': '无线网', '商场': '免税店', '登机牌': '值机柜台', '休息': '休息区', '行李车': '手推车', '喝咖啡': '咖啡吧',
                 '出国wifi': 'wi-fi租赁', 'wifi购买': 'wi-fi租赁', '境外wifi': 'wi-fi租赁', '车停': '停车场', '抽烟': '吸烟室', '轮椅': '机场轮椅', '开会': '会议室',
                 '复印': '打印', '航意险': '航空意外保险', '柜台': '值机柜台'}

ABS_REL_DICT = {'几点': '时间', '地址': '地点', '地方': '地点', '几楼': '地点', '位置': '地点', '怎么走': '地点', '地点': '地点', '时间': '时间', '价格': '价格', '收费': '价格', '收费标准': '价格',
                '联系电话': '联系电话', '电话': '电话', '联系方式': '联系方式'}

LIST_WORD_LIST = ['有', '哪些', '几个', '可以', '介绍', '列举','哪几大']

# 可回答的列举类问题标志词
LIST_CONTENT_LIST = ['航空', '航司', '机场','机型','系列','飞机']

# CQA问题关键词
CQA_QUESTION_WORDS = ["怎么做", "怎么办", "怎样", "如何", "办理", "手续", "流程", "条件", "要做什么", "有哪些要求", "目的", "为什么", "靠什么", "原理", "原因",
                      "为何", "排名", "历史", "发展史", "趋势", "怎么来的", "来源", "优点", "缺点", "优缺点", "特点", "区别", "相比", "比较", "前者", "后者",
                      "关系", "联系", "影响", "差值", "注意", "要求", "须知", "规则", "过程", "上飞机", "开飞机", "坐飞机", "需要", "可提供", "使用",
                      "积分", "补偿", "故障", "携带", "排名", "发展", "起源", "计算", "怎么算", "额度"]
# 非民航问题关键词
NO_AIR_QUESTION_WORDS = ["直升机", "直升飞机", "招聘", "招飞行员", "驾驶", "考", "选拔", "故事", "神话", "无人机", "无人飞机", "空军", "兵", "军官"]

# 民航问题中频繁出现的词语
FREQUENT_WORDS = ["飞机", "飞行", "客票", "航班", "行李"]  # "航空"容易影响"航空公司"问题，需要单独处理

# 易被错匹配的术语类实体名称
FREQUENT_MATCHED_ENT = ["机场", "飞机", "飞行（轮档）小时", "航空器（大中型）", "航空器（小型）", "航空", "客票", "航班", "行李", "飞行员"]

# 特殊英语分词处理（如ca是国航的代码，如果把ICAO继续ngram，会在指称中出现ca，最终造成实体链接错误）
SPECIAL_ENGLISH_IN_SEGMENT = {'IATA': ['AT', 'IA', 'ATA', 'TA', 'at', 'ia', 'ata', 'ta'], "ICAO": ['ca', 'CA']}

# 航空公司公司名称中对实体相似度匹配无意义的词（在实体链接相似度匹配前需要删除）
AIR_COMPANY_TAG = ['航空股份有限公司', '航空有限责任公司', '航空有限公司', '航空投资有限公司', '航空公司']


class Airport:
    def __init__(self):
        self.name = '昆明长水机场'
        self.alias = ['昆明机场', '长水机场', '昆明长水机场']
        self.filter_words = ['昆明机场', '长水机场', '昆明长水', '昆明国际', '国际机场']
        self.remove_words = ['机场', '长水']

airport = Airport()

# ths
check_kbqa_ths = 0.5
simi_ths = 0.80
prop_ths = 0.76
exact_ent_ths = 0.83

'''multitask training'''
RAW_SOURCE_DATA = os.path.join(
    project_dir, 'KM_KBQA/data_processing/seq_label_raw.txt')
TARGET_SOURCE_DATA = os.path.join(
    project_dir, 'KM_KBQA/data_processing/seq_label_target.txt')

'''KBQA config'''
# 问题分类模型地址
QCLS_PATH = os.path.join(model_path, 'check_kbqa_model.pt')

# config logging

logger = logging.getLogger('qa')
formatter = logging.Formatter(
    '%(asctime)s %(filename)s:%(lineno)d %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# ch.setLevel(logging.INFO)
fh = logging.FileHandler('run.log', 'a', encoding='utf-8')
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
debug_h = logging.FileHandler('debug.log', 'a', encoding='utf-8')
debug_h.setFormatter(formatter)
debug_h.setLevel(logging.DEBUG)
# logger.addHandler(ch)
logger.addHandler(fh)
logger.addHandler(debug_h)
