import pandas as pd
from tqdm import tqdm

from KM_KBQA.qa.QAFull import QAFull


def test_qa(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        quesitons = [line.strip() for line in f]
    qa = QAFull()
    ans = [qa.answer(q) for q in tqdm(quesitons)]
    ans_text = ['\n'.join(map(lambda x:x['natural_ans'], a[:3])) for a in ans]
    ans_df = pd.DataFrame({'question': quesitons, 'answer top3': ans_text})
    ans_df.to_excel('qa_res.xlsx')


if __name__ == "__main__":
    test_file = 'KM_KBQA/res/all_single_rel_cls_raw.txt'
    test_qa(test_file)
