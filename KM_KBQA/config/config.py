import logging
import os

project_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_dir, 'data')
model_path = os.path.join(project_dir, 'models')

LEX_PATH = os.path.join(project_dir, 'lexicons')
STOP_WORD_PATH = os.path.join(LEX_PATH, 'stopwords.txt')
AIR_LEXICON_PATH = os.path.join(LEX_PATH, 'air_lexicon.txt')

ABSTRACT_DICT = {'丢': '失物招领', '取款机': 'atm', '存款机': 'atm', '存取款机': 'atm', '提款机': 'atm', '取钱': '银行', 'kfc': '肯德基', '加油': '加油站', '乘机手续': '值机柜台',
                 '住': '酒店', '住宿': '酒店', '吃': '餐厅', '吃饭': '餐厅', '餐饮': '餐厅', '开水': '饮水处', '剪发': '理发店', '剪头发': '理发店', '喂奶': '母婴室',
                 '热水': '饮水处', '插电': '充电', '换': ['货币兑换', '母婴室'], '兑换': '货币兑换', '厕所': '卫生间', '胶囊': '睡眠', '打包': '行李打包', '寄存': '行李寄存',
                 '换外币': '货币兑换', '打的士': '出租车', '打的': '出租车', '打车': '出租车', '的士': '出租车', '巴士': '机场公交', '大巴': '公交', '停': '停车场', '速运': '快递',
                 '速递': '快递', '配送': '快递', '上网': '无线网', '商场': '免税店', '登机牌': '值机柜台', '休息': '休息区', '行李车': '手推车', '喝咖啡': '咖啡吧',
                 '出国wifi': 'wi-fi租赁', 'wifi购买': 'wi-fi租赁', '境外wifi': 'wi-fi租赁', '车停': '停车场', '抽烟': '吸烟室', '轮椅': '机场轮椅', '开会': '会议室',
                 '复印': '打印', '航意险': '航空意外保险', '柜台': '值机柜台'}

ABS_REL_DICT = {'几点': '时间', '地址': '地点', '地方': '地点', '几楼': '地点', '位置': '地点', '怎么走': '地点', '地点': '地点', '时间': '时间', '价格': '价格', '收费': '价格', '收费标准': '价格',
                '联系电话': '联系电话', '电话': '电话', '联系方式': '联系方式'}

LIST_WORD_LIST = ['有', '哪些', '几个', '可以']
FILTER_NODE_LABELS = ['行李安检', '普通行李', '航空公司',
                      '电池', '剧毒物品', '非易燃物品', '放射性物品', '易燃易爆物品']


class Airport:
    def __init__(self):
        self.name = '昆明长水机场'
        self.alias = ['昆明机场', '长水机场', '昆明长水机场']
        self.filter_words = ['昆明机场', '长水机场', '昆明长水', '昆明国际', '国际机场']
        self.remove_words = ['机场', '长水']


airport = Airport()

# ths
check_kbqa_ths = 0.3
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
logger.setLevel(logging.INFO)
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
