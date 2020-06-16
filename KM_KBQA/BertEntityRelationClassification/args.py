import argparse
import json
import os

USE_PARSER = False
# init argparse
if USE_PARSER:
    parser = argparse.ArgumentParser()
    parser.add_argument('-blr', '--bert_lr', type=float, default=2e-5,
                        help='learning rate for bert-related network')
    parser.add_argument('-nblr', '--non_bert_lr', type=float, default=1e-3,
                        help='learning rate for bert-unrelated network')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=32, help='batch size for training and eval')
    parser.add_argument('-lhs', '--lstm_hidden_size', type=int,
                        default=512, help='bilstm hidden size')
    parser.add_argument('-mhs', '--mlp_hidden_size', type=int,
                        default=512, help='mlp hidden size')
    parser.add_argument('-dv', '--device', type=str,
                        default='cuda:0', help='used training device')
    parser.add_argument('-ep', '--epoch', type=int,
                        default=15, help='training epochs')
    p_args = parser.parse_args()


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)



# -----------ARGS---------------------
VOCAB_FILE = "KM_KBQA/models/vocab.txt"

bert_model = "KM_KBQA/models/"  # BERT 预训练模型种类 bert-base-chinese
bert_config_path = "KM_KBQA/models/bert_config.json"
label_method = 'BIO'

flag_words = ["[PAD]", "[CLP]", "[SEP]", "[UNK]"]
max_seq_length = 100
do_lower_case = True
train_batch_size = 32
eval_batch_size = 32
learning_rate = 2e-5
non_bert_learning_rate = 1e-3
num_train_epochs = 1
warmup_proportion = 0.1
no_cuda = False
seed = 2020
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
# labels = ["B_PER", "I_PER", "B_T", "I_T", "B_ORG", "I_ORG", "B_LOC", "I_LOC", "O"]
if label_method == 'BIO':
    # labels = ["B-E", "I-E", "B-RL", "I-RL", "B-RT", "I-RT", "O"]
    labels = ["B-E", "I-E", "B-RL", "I-RL",
              "B-RT", "I-RT", "B-ABS", "I-ABS", "O"]
elif label_method == 'BIEOS':
    labels = ["B-E", "I-E", "E-E", "S-E", "B-RL", "I-RL",
              "E-RL", "S-RL", "B-RT", "I-RT", "E-RT", "S-RT", "O"]
num_tag = len(labels)
device = "cpu"


# ----------- MULTITASKING ---------------------
# data loader
K = 5
RAW_SEQ_DATA = 'KM_KBQA/res/all_single_seq_label_raw.txt'
TARGET_SEQ_DATA = 'KM_KBQA/res/all_single_seq_label_target.txt'
CLS_RAW_SOURCE_DATA = 'KM_KBQA/res/all_single_rel_cls_raw.txt'
ENT_CLS_DATA = 'KM_KBQA/res/all_single_ent_cls.txt'
REL_CLS_DATA = 'KM_KBQA/res/all_single_rel_cls.txt'
RESTR_CLS_DATA = 'KM_KBQA/res/all_single_restr_cls.txt'
SCHEMA_DATA = 'KM_KBQA/res/existed_schema_4.30.json'
ENT_SEQ_CLS_DATA = 'KM_KBQA/res/all_single_ent_seq_cls.txt'
REL_SEQ_CLS_DATA = 'KM_KBQA/res/all_single_rel_seq_cls.txt'

# slot filling 同上
slot_filling_loss_ratio = 0.01
ent_loss_ratio = 1.0
rel_loss_ratio = 0.1

# classification
ENT_LABELS = ['', '航空公司', '机场接送机服务', '行李查询', '母婴室', '基础设施', '儿童娱乐区', '货币兑换处', '摆渡车', '扫描', '值机柜台', '安检通道', '取票处',
              '行李安检', '钟点房', '医疗服务', '中转柜台', '汽车维修', '足浴店', '昆明长水机场', '行李打包', '临时身份证办理', '行李搬运', '登机过检服务', '过夜区',
              '商场', '边防检查', '商务型酒店', '政要通道', '航空售票', '派出所', '休闲吧', '空港物流', '睡眠舱', '充电服务', '行李转盘', '回民餐厅', '电瓶车',
              '离境退税', '快速安检通道', '保险', 'ATM机', '失物招领处', '飞机维修', '轮椅服务', '普通行李', '旅游咨询', '银行', '机场交规', '金融服务区', '租车服务',
              '国内乘机有效证件', '酒店', '免费手推车', '机场招聘', '机场大巴', '到达区', '问询台', '快餐', '机场航站楼', '贵宾服务', '办公室', '商店', '广播寻人', '更衣室',
              '贵宾厅', '其他柜台', '航站楼', '茶室', '售货机', '贵宾通道', '电话卡销售', '小吃', '冬衣寄存', '机场WIFI', '引领服务', '药店', '行李携带', '消费维权', '货运站',
              '饮水处', '打印', '手机营业厅', '商业', '无陪伴旅客服务', '卫生间', '电源插座', '按摩处', '行李寄存', '汽车站', '照相机', '出租车', '长水机场出口', '电池', '商务中心',
              '吸烟室', '机场投诉电话', '邮局', '停车服务', '电视机', '中餐厅', '空港快线', '休息室', '国际航班', '公共电话亭', '候机室', '加油站', '宠物运输', '书店', '爱心服务',
              '特殊旅客服务', '手机租赁', '出国宝', '机上手机充电', '奶茶店', '会议室', '复印', '传真', '快递', '理发店', '空调', '车管所', '经济连锁型酒店', '航空安全护卫部',
              '国际到达', '绿色通道', '老人旅客服务', '餐厅', '旅游退换货', '头等舱休息室', '免税店', '咖啡厅', '行李托运', 'WIFI租赁', '落地签自拍', '超市', '海关过境', '地铁',
              '签证办理', '奶粉']

REL_LABELS = ['', '时间', '电话', '密码', '推荐菜', '主营菜系', '价格', '停车场类型', '服务内容', '限制', '地点', '航站楼面积', '时长', '邮编', '介绍', '航司代码', '货币种类',
              '购买条件', '经营范围', '使用方法', '限制金额', '工资']

REL_CONVERT_DICT = {'手续费': '价格', '收费标准': '价格', '服务介绍': '介绍', '携带限制': '限制', '托运限制': '限制', '携带限额': '限制', '托运限额': '限制', '携带尺寸限制': '限制',
                    '联系电话': '电话', '客服电话': '电话', '维修电话': '电话', '联系方式': '电话', '使用时间': '时间', '服务时间': '时间', '银行类型': '所属银行类型', '所属地区': '地点'}

REVERT_REL_CONVERT_DICT = {'价格': ['手续费', '收费标准'], '介绍': '服务介绍', '限制': ['携带限制', '托运限制', '携带限额', '托运限额', '携带尺寸限制'], '电话': [
    '联系电话', '客服电话', '维修电话', '联系方式'], '时间': ['使用时间', '服务时间'], '所属银行类型': ['银行类型'], '地点': ['所属地区']}

# multitask network
crf_hidden_size = 768
hidden_dropout_prob = 0.4
bilstm_input_size = 768
bilstm_hidden_size = 256
bilstm_num_layers = 2
bilstm_dropout = 0.4
mlp_input_size = 2 * bilstm_hidden_size
mlp_hidden_size = 256
mlp_ent_output_size = len(ENT_LABELS)
mlp_rel_output_size = len(REL_LABELS)
mlp_num_layers = 2

interaction = 0
use_soft_attention = True
attention_T = 10

if USE_PARSER:
    # init args
    learning_rate = p_args.bert_lr
    non_bert_learning_rate = p_args.non_bert_lr
    train_batch_size = p_args.batch_size
    eval_batch_size = p_args.batch_size
    bilstm_hidden_size = p_args.lstm_hidden_size
    mlp_hidden_size = p_args.mlp_hidden_size
    mlp_input_size = 2 * bilstm_hidden_size
    device = p_args.device
    num_train_epochs = p_args.epoch

# save configs
task_name = '%s_blr_%f_nlr_%f_bs_%d_lhs_%d_mhs_%d_epoch_%d_all_data' % (
    'multitask_atten_cnn', learning_rate, non_bert_learning_rate, train_batch_size, bilstm_hidden_size, mlp_hidden_size,
    num_train_epochs)
# task_name = "bert_ner_data_neraug_BIO"                      # 训练任务名称
log_path = "KM_KBQA/output/logs/" + task_name
create_path(log_path)
k_fold_report_path = os.path.join(log_path, 'k_fold_report.txt')
eval_res_path = os.path.join(log_path, 'eval_res.csv')
plot_path = "KM_KBQA/output/images/" + task_name + "/loss_acc.png"
create_path(plot_path.rsplit('/', 1)[0])
data_dir = "KM_KBQA/res/"  # 原始数据文件夹，应包括tsv文件
cache_dir = "KM_KBQA/model/"
output_dir = os.path.join("KM_KBQA/output/checkpoint",
                          task_name)  # checkpoint和预测输出文件夹
trained_dir = os.path.join("KM_KBQA/output/trained_model")
create_path(output_dir)
create_path(trained_dir)

tensorboard_path = os.path.join('KM_KBQA/output', 'tensorboardX', task_name)
create_path(tensorboard_path)


def load_existing_schema(schema_path, ent2id, rel2id):
    # load json
    with open(schema_path, 'r', encoding='utf-8') as f:
        # line = f.readlines()
        schema_data = json.load(f)
    schema_data = schema_data['ans']

    ent_data = [x['name'] for x in schema_data]
    schema_dict = {}
    for i in range(len(ENT_LABELS)):
        schema_dict[i] = [0, 1, 2, 6, 8, 10]
    for i, ent_item in enumerate(ent_data):
        ent_item = ent_item.strip().replace('类服务', '')
        if ent_item not in ENT_LABELS:
            continue
        cand_rel = schema_data[i]['prop_list']
        schema_dict[ent2id(ent_item)] = [0]
        for rel_item in cand_rel:
            rel_item = rel_item.strip()
            if rel_item not in REL_LABELS and rel_item not in REL_CONVERT_DICT.keys():
                continue
            if rel_item in REL_CONVERT_DICT.keys():
                rel_item = REL_CONVERT_DICT[rel_item]
            if rel_item not in REL_LABELS:
                continue
            if rel2id(rel_item) not in schema_dict[ent2id(ent_item)]:
                schema_dict[ent2id(ent_item)].append(rel2id(rel_item))

    return schema_dict


def cls_ent_to_id(ent_name):
    return ENT_LABELS.index(ent_name)


def cls_rel_to_id(rel_name):
    return REL_LABELS.index(rel_name)


schema_file = SCHEMA_DATA
cls_schema_dict = load_existing_schema(
    schema_file, cls_ent_to_id, cls_rel_to_id)

# restrictions
airline_list = os.path.join(data_dir, 'airline.txt')
currency_list = os.path.join(data_dir, 'currency.txt')
bank_list = os.path.join(data_dir, 'banks.txt')


def load_restr_list(file):
    ret = []
    with open(file, 'r') as f:
        for line in f:
            ret.append(line.strip())
    return ret


airline_list = load_restr_list(airline_list)
currency_list = load_restr_list(currency_list)
bank_list = load_restr_list(bank_list)
comp_list = ['贵宾厅', '贵宾', '头等舱', '营业厅', '手机厅']

cws_model_path = os.path.join(cache_dir, 'ltp_data', 'cws.model')
pos_model_path = os.path.join(cache_dir, 'ltp_data', 'pos.model')
par_model_path = os.path.join(cache_dir, 'ltp_data', 'parser.model')

print('task begin: ', task_name)
