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
ENT_LABELS = ['', "“国王”系列", "其他型号", "707系列", "717系列", "737系列", "757系列", "A300系列", "A320系列", "A340系列", "A380系列", "ERJ系列",
              "挑战者系列", "Falcon 10/100系列", "Falcon 2000系列", "Dash7", "“喷气流”31型客机", "125系列", "“S.E.P”系列", "“雷神”系列",
              "“公爵”系列", "C919大型客机", "CR929宽体客机", "727系列", "747系列", "767系列", "777系列", "787系列", "A310系列", "A330系列",
              "A350系列", "EMB系列", "E190系列", "CRJ系列", "Falcon 20/200系列", "Falcon 50系列", "Falcon 7X系列", "Dash8",
              "“喷气流”41型客机", "146系列（RJ系列）", "“湾流”系列", "“奖状”系列", "“凯旋”系列", "“豪客”系列", "“富豪”系列", "ARJ21飞机", "国内航空公司",
              "国内机场", "民航企业", "行政机构", "科学家", "国外航空公司", "国外机场", "民航院校", "飞行员", "企业家", "市场", "运输凭证", "运价", "行李", "运输服务",
              "质量指标", "设施、设备", "“喷气流”61型客机", "通用术语", "旅客", "定座管理", "售票", "乘机", "载重与平衡", "运输", "包机", "其他", "代号共享",
              "航空运输", "代理人", "民用航空", "航空运输企业", "运输飞机", "航线", "客运市场", "市场分析", "市场管理", "常旅客", "计算机定座系统",
              "座位控制", "客票", "航程", "退票", "比例分摊", "航班动态信息", "办理乘机手续", "座位管理", "定座", "客票构成", "旅费证", "出票", "票价计算", "安全检查",
              "行李", "行李标识", "分摊赔偿", "飞机重量", "旅客票价", "费用", "行李收运", "行李交付", "行李不正常运输", "行李运输事故记录", "行李赔偿", "载重平衡", "飞机调配",
              "售票服务", "机场", "航班", "运输规则", "生产指标", "客源", "市场营销", "普通旅客", "始发旅客", "特殊旅客", "定座服务", "地面服务", "空中服务", "广播服务",
              "餐饮服务", "包机业务", "包机费", "质量指标", "行李运输差错率", "设备完好率", "配载", "运力", "运输", "航班", "特殊服务", "引导服务", "问讯服务", "包机合同"]  # ,"包机合同变更", "包机", "专机", "旅客满意率", "出票差错率", "有效投诉率", "运输事故", "航站楼", "特种车辆", "运输事故等级", "客机坪", "残疾人服务专用设备","成本","收益", "收款方式", "客票更改", "航空运价", "航空客运收益管理", "航空公司联盟"

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
PREFIX = os.path.dirname(__file__)
log_path = PREFIX+"/output/logs/" + task_name
create_path(log_path)
k_fold_report_path = os.path.join(log_path, 'k_fold_report.txt')
eval_res_path = os.path.join(log_path, 'eval_res.csv')
plot_path = PREFIX+"/output/images/" + task_name + "/loss_acc.png"
create_path(plot_path.rsplit('/', 1)[0])
data_dir = "KM_KBQA/res/"  # 原始数据文件夹，应包括tsv文件
cache_dir = "KM_KBQA/model/"
output_dir = os.path.join(PREFIX+"/output/checkpoint",
                          task_name)  # checkpoint和预测输出文件夹
trained_dir = os.path.join(PREFIX+"/output/trained_model")
create_path(output_dir)
create_path(trained_dir)

tensorboard_path = os.path.join(PREFIX, 'tensorboardX', task_name)
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
# airline_list = os.path.join(data_dir, 'airline.txt')
airport_list = os.path.join(data_dir, 'airport.txt')
# currency_list = os.path.join(data_dir, 'currency.txt')
# bank_list = os.path.join(data_dir, 'banks.txt')


def load_restr_list(file):
    ret = []
    with open(file, 'r') as f:
        for line in f:
            ret.append(line.strip())
    return ret


# airline_list = load_restr_list(airline_list)
airport_list = load_restr_list(airport_list)
# currency_list = load_restr_list(currency_list)
# bank_list = load_restr_list(bank_list)
comp_list = ['贵宾厅', '贵宾', '头等舱', '营业厅', '手机厅']

cws_model_path = os.path.join(cache_dir, 'ltp_data', 'cws.model')
pos_model_path = os.path.join(cache_dir, 'ltp_data', 'pos.model')
par_model_path = os.path.join(cache_dir, 'ltp_data', 'parser.model')

