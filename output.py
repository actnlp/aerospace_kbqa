import os

import pandas as pd
from tqdm import tqdm

from KM_KBQA.config import config
from KM_KBQA.qa.QAFull import QAFull
from KM_KBQA.qa.QA import QA

TEST_FILE_DIR = 'test/'
TEST_RESULT_DIR = os.path.join(TEST_FILE_DIR, 'result')
if not os.path.isdir(TEST_RESULT_DIR):
    os.mkdir(TEST_RESULT_DIR)

def create_format(a):
    if "rel_name" not in a:
        return "{}-无-{}".format(a['entity'], a['mention'])
    else:
        return "{}-{}-{}".format(a['entity'], a['rel_name'], a['mention'])

def test_qa(fname, use_full=True):  # use_full ???  包括拒识？
    if not os.path.isfile(fname):
        fname = os.path.join(TEST_FILE_DIR, fname)
    with open(fname, 'r', encoding='utf-8') as f:
        quesitons = [line.strip() for line in f]
    #quesitons = ['CPA是哪个航空公司的代码？']
    if use_full:
        qa = QAFull()
    else:
        qa = QA()
    ans = [qa.answer(q) for q in tqdm(quesitons)]
    #print(ans[1])
    #ans_text = ['\n'.join(map(lambda x:x['natural_ans'], a[:3])) for a in ans]
    ans_text_1 = []
    ans_text_2 = []
    ans_text_3 = []
    ans_ent_1 = []
    ans_ent_2 = []
    ans_ent_3 = []
    for a in ans:
        if (len(a) >= 3):
            ans_text_1.append(a[0]['natural_ans'])
            ans_text_2.append(a[1]['natural_ans'])
            ans_text_3.append(a[2]['natural_ans'])
            ans_ent_1.append(create_format(a[0]))
            ans_ent_2.append(create_format(a[1]))
            ans_ent_3.append(create_format(a[2]))
        elif (len(a) == 2):
            ans_text_1.append(a[0]['natural_ans'])
            ans_text_2.append(a[1]['natural_ans'])
            ans_text_3.append("无")
            ans_ent_1.append(create_format(a[0]))
            ans_ent_2.append(create_format(a[1]))
            ans_ent_3.append("无")            
        elif (len(a) == 1):
            ans_text_1.append(a[0]['natural_ans'])
            ans_text_2.append("无")
            ans_text_3.append("无")
            ans_ent_1.append(create_format(a[0]))
            ans_ent_2.append("无")
            ans_ent_3.append("无")
        elif (len(a) == 0):
            ans_text_1.append("无")
            ans_text_2.append("无")
            ans_text_3.append("无")
            ans_ent_1.append("无")
            ans_ent_2.append("无")
            ans_ent_3.append("无")
    ans_df = pd.DataFrame({'question': quesitons, \
                           '答案1': ans_text_1, '实体-属性-mention_1':ans_ent_1, \
                           '答案2': ans_text_2, '实体-属性-mention_2':ans_ent_2, \
                           '答案3': ans_text_3, '实体-属性-mention_3':ans_ent_3  \
                           })
    basename = os.path.basename(fname).split('.')[0]
    output_fname = os.path.join(TEST_RESULT_DIR, basename + '_res.xlsx')
    # 与上一次结果对比
#    if os.path.isfile(output_fname):
#        old_df = pd.read_excel(output_fname)
#        if 'answer top3' in old_df:
#            idx = old_df['answer top3'] != ans_df['answer top3']
#            diff_df = pd.merge(old_df[idx], ans_df[idx],
#                               left_index=True, right_index=True, how='left')
#            diff_fname = os.path.join(TEST_RESULT_DIR, basename + '_diff.xlsx')
#            diff_df.to_excel(diff_fname)

    ans_df.to_excel(output_fname)


if __name__ == "__main__":
    test_files = [
        'aerospace.txt',
        # 'test_refuse.txt',
        # 'refuse.txt'
        # 'wrong_refuse.txt'
    ]
    for test_file in test_files:
        test_qa(test_file, use_full=True)
    # QA.test_qa()
