# coding: utf-8
import json
import pdb
import random

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from . import args


def produce_data(shuffle):
    # load data
    raw_seq, tar_seq, ent_cls, rel_cls, ent_seq_cls, rel_seq_cls = [], [], [], [], [], []
    with open(args.RAW_SEQ_DATA, 'r', encoding='utf-8') as fraw_seq:
        for line in fraw_seq:
            raw_seq.append(line.lower().strip().replace(' ', '.').split('|'))
    with open(args.TARGET_SEQ_DATA, 'r', encoding='utf-8') as ftar_seq:
        for line in ftar_seq:
            tar_seq.append(line.strip().split('|'))
    with open(args.ENT_CLS_DATA, 'r', encoding='utf-8') as fent_cls:
        for line in fent_cls:
            ent_cls.append(line.strip())
    with open(args.REL_CLS_DATA, 'r', encoding='utf-8') as frel_cls:
        for line in frel_cls:
            rel_cls.append(line.strip())
    with open(args.ENT_SEQ_CLS_DATA, 'r', encoding='utf-8') as fent_seq_cls:
        for line in fent_seq_cls:
            ent_seq_cls.append([int(x) for x in (line.strip().split('|'))])
    with open(args.REL_SEQ_CLS_DATA, 'r', encoding='utf-8') as frel_seq_cls:
        for line in frel_seq_cls:
            rel_seq_cls.append([int(x) for x in (line.strip().split('|'))])

    assert len(raw_seq) == len(tar_seq) and len(raw_seq) == len(ent_cls) and len(raw_seq) == len(tar_seq) and len(raw_seq) == len(ent_seq_cls) and len(raw_seq) == len(rel_seq_cls)
    # shuffle
    idx = [i for i in range(len(raw_seq))]
    if shuffle:
        random.seed(args.seed)
        random.shuffle(idx)
    sf_raw_seq, sf_tar_seq, sf_ent_cls, sf_rel_cls, sf_ent_seq_cls, sf_rel_seq_cls = [], [], [], [], [], []
    for i in idx:
        sf_raw_seq.append(raw_seq[i])
        sf_tar_seq.append(tar_seq[i])
        sf_ent_cls.append(ent_cls[i])
        sf_rel_cls.append(rel_cls[i])
        sf_ent_seq_cls.append(ent_seq_cls[i])
        sf_rel_seq_cls.append(rel_seq_cls[i])

    return idx, sf_raw_seq, sf_tar_seq, sf_ent_cls, sf_rel_cls, sf_ent_seq_cls, sf_rel_seq_cls

def get_k_fold_data(k, i, *data):
    # pre-check and data extraction
    assert k > 1
    data = data[0]
    fold_size = len(data[0]) // k

    k_fold_train_data = [[] for _ in range(len(data))]
    k_fold_valid_data = [[] for _ in range(len(data))]
    num_j_part = [[] for _ in range(len(data))]

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        for m in range(len(data)):
            num_j_part[m] = data[m][idx]
        if j == i:
            for n in range(len(data)):
                k_fold_valid_data[n] = num_j_part[n]
        elif len(k_fold_train_data[0]) == 0:
            for n in range(len(data)):
                k_fold_train_data[n] = num_j_part[n]
        else:
            for n in range(len(data)):
                k_fold_train_data[n] += num_j_part[n]

    return k_fold_train_data, k_fold_valid_data

def convert_seq_feature(seq_raw_data, seq_tar_data, tokenizer):
    # convert sequence raw data to bert feature
    input_ids, input_masks, segment_ids, label_ids, output_masks = [], [], [], [], []
    for i, token in enumerate(seq_raw_data):
        tokens = ["[CLS]"] + token + ["[SEP]"]
        output_mask = [0] + [1] * len(token) + [0]

        segment_id = [0] * len(tokens)
        ## 词转换成数字
        try:
            input_id = tokenizer.convert_tokens_to_ids(tokens)
        except:
            pdb.set_trace()

        input_mask = [1] * len(input_id)

        padding = [0] * (args.max_seq_length - len(input_id))

        input_id += padding
        input_mask += padding
        segment_id += padding
        output_mask += padding

        label_id = [args.labels.index(x) for x in seq_tar_data[i]]
        label_padding = [-1] * (args.max_seq_length - len(label_id))
        label_id += label_padding

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        label_ids.append(label_id)
        output_masks.append(output_mask)

    return input_ids, input_masks, segment_ids, output_masks, label_ids

def convert_cls_seq_feature(ent_seq_cls, rel_seq_cls):
    ret_ent_seq_cls, ret_rel_seq_cls = [], []
    for i, _ in enumerate(ent_seq_cls):
        cur_ent_len = len(ent_seq_cls[i])
        cur_rel_len = len(rel_seq_cls[i])
        ret_ent_seq_cls.append(ent_seq_cls[i] + [0] * (args.max_seq_length - cur_ent_len))
        ret_rel_seq_cls.append(rel_seq_cls[i] + [0] * (args.max_seq_length - cur_rel_len))
    return ret_ent_seq_cls, ret_rel_seq_cls

def convert_cls_feature(ent_cls, rel_cls):
    ent_labels, rel_labels = [], []
    for i, _ in enumerate(ent_cls):
        ent_label = args.ENT_LABELS.index(ent_cls[i])
        rel_label = args.REL_LABELS.index(rel_cls[i])
        ent_labels.append(ent_label)
        rel_labels.append(rel_label)

    return ent_labels, rel_labels

def create_dataset_batch(idx, input_ids, input_masks, segment_ids, output_masks, label_ids, ent_labels, rel_labels, mode):
    idx = torch.tensor(idx, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    output_masks = torch.tensor(output_masks, dtype=torch.long)

    label_ids = torch.tensor(label_ids, dtype=torch.long)
    ent_labels = torch.tensor(ent_labels, dtype=torch.long)
    rel_labels = torch.tensor(rel_labels, dtype=torch.long)

    # 数据集
    data = TensorDataset(idx, input_ids, input_masks, segment_ids, output_masks, label_ids, ent_labels, rel_labels)

    if mode == "train":
        sampler = RandomSampler(data)
        batch_size = args.train_batch_size
    elif mode == "dev":
        sampler = SequentialSampler(data)
        batch_size = args.eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator
    elif mode == "dev":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)
