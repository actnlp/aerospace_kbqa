import csv
import pdb
from .. import args

def write_eval_res(path, ids, ents, rels, is_trues_top1, is_trues_top3, k):
    # c0ntents: {'id': , 'text', 'entity':, 'relation':, 'true':}
    # k-fold path
    k_path = '%s_%d.csv' % (path.split('.csv')[0], k)
    with open(k_path, 'w', encoding='utf-8') as fcsv:
        texts = []
        with open(args.CLS_RAW_SOURCE_DATA, 'r', encoding='utf-8') as ftext:
            for line in ftext:
                texts.append(line.strip())
        writer = csv.writer(fcsv)
        writer.writerow(['id', 'text', 'entity', 'relation', 'is_true'])
        for i in range(len(ids)):
            try:
                writer.writerow([ids[i], texts[ids[i]], args.ENT_LABELS[ents[i]], args.REL_LABELS[rels[i]], is_trues_top1[i], is_trues_top3[i]])
            except:
                pdb.set_trace()

    return

def aeas_write_eval_res(path, ids, ents, rels, is_trues, k):
    # c0ntents: {'id': , 'text', 'entity':, 'relation':, 'true':}
    # k-fold path
    k_path = '%s_%d.csv' % (path.split('.csv')[0], k)
    with open(k_path, 'w', encoding='utf-8') as fcsv:
        texts = []
        with open(args.CLS_RAW_SOURCE_DATA, 'r', encoding='utf-8') as ftext:
            for line in ftext:
                texts.append(line.strip())
        writer = csv.writer(fcsv)
        writer.writerow(['id', 'text', 'entity', 'relation', 'is_true'])
        for i in range(len(ids)):
            try:
                writer.writerow([ids[i], texts[ids[i]], args.SEQ_ENT_CLS_LABELS[ents[i][0]], args.SEQ_REL_CLS_LABELS[rels[i][0]], is_trues[i]])
            except:
                pdb.set_trace()

    return