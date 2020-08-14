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


def test_qa(fname, use_full=True):  # use_full ???  包括拒识？
    # if not os.path.isfile(fname):
    #     fname = os.path.join(TEST_FILE_DIR, fname)
    # with open(fname, 'r', encoding='utf-8') as f:
    #     quesitons = [line.strip() for line in f][0:1]
    quesitons = ['航线：']
    if use_full:
        qa = QAFull()
    else:
        qa = QA()
    ans = [qa.answer(q) for q in tqdm(quesitons)]

    ans_text = ['\n'.join(map(lambda x:x['natural_ans'], a[:3])) for a in ans]
    ans_df = pd.DataFrame({'question': quesitons, 'answer top3': ans_text})
    basename = os.path.basename(fname).split('.')[0]
    output_fname = os.path.join(TEST_RESULT_DIR, basename + '_res.xlsx')
    # 与上一次结果对比
    if os.path.isfile(output_fname):
        old_df = pd.read_excel(output_fname)
        if 'answer top3' in old_df:
            idx = old_df['answer top3'] != ans_df['answer top3']
            diff_df = pd.merge(old_df[idx], ans_df[idx],
                               left_index=True, right_index=True, how='left')
            diff_fname = os.path.join(TEST_RESULT_DIR, basename + '_diff.xlsx')
            diff_df.to_excel(diff_fname)

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
