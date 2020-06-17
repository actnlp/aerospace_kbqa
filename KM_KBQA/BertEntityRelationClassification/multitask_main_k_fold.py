# encoding: utf-8
import os

import torch
from transformers import BertConfig, BertModel, BertTokenizer

from . import args
from .multitask_data_processor import (convert_cls_feature,
                                       convert_seq_feature,
                                       create_dataset_batch, get_k_fold_data,
                                       produce_data)
from .MultiTaskKBQA import MultiTaskKBQA
from .train import multitask_fit
from .utils.progress_util import ProgressBar


def k_fold_start(k, shuffle=True):
    # load data
    idx, raw_seq, tar_seq, ent_cls, rel_cls, _, _ = produce_data(
        shuffle=shuffle)

    # k-fold parameters
    losses, slot_seq_f1, ent_cls_f1, rel_cls_f1, overall_top1_f1, overall_top3_f1 = [
    ], [], [], [], [], []

    # split dataset to k fold
    tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)
    for i in range(k):
        k_fold_train_data, k_fold_valid_data = \
            get_k_fold_data(k, i, [raw_seq, tar_seq, ent_cls, rel_cls, idx])

        train_raw_seq, train_tar_seq, train_ent_cls, train_rel_cls, train_idx = k_fold_train_data
        valid_raw_seq, valid_tar_seq, valid_ent_cls, valid_rel_cls, valid_idx = k_fold_valid_data

        tr_input_ids, tr_input_masks, tr_segment_ids, tr_output_masks, tr_label_ids = convert_seq_feature(
            train_raw_seq, train_tar_seq, tokenizer)
        vl_input_ids, vl_input_masks, vl_segment_ids, vl_output_masks, vl_label_ids = convert_seq_feature(
            valid_raw_seq, valid_tar_seq, tokenizer)
        tr_ent_label_ids, tr_rel_label_ids = convert_cls_feature(
            train_ent_cls, train_rel_cls)
        vl_ent_label_ids, vl_rel_label_ids = convert_cls_feature(
            valid_ent_cls, valid_rel_cls)

        num_train_steps = int(
            len(tr_input_ids) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        epoch_size = num_train_steps * args.train_batch_size * \
            args.gradient_accumulation_steps / args.num_train_epochs

        train_iter = create_dataset_batch(train_idx, tr_input_ids, tr_input_masks, tr_segment_ids,
                                          tr_output_masks, tr_label_ids, tr_ent_label_ids, tr_rel_label_ids, 'train')
        eval_iter = create_dataset_batch(valid_idx, vl_input_ids, vl_input_masks, vl_segment_ids,
                                         vl_output_masks, vl_label_ids, vl_ent_label_ids, vl_rel_label_ids, 'dev')
        pbar = ProgressBar(epoch_size=epoch_size,
                           batch_size=args.train_batch_size)

        model = MultiTaskKBQA(args).to(args.device)

        print(model)

        eval_loss, slot_f1, ent_f1, rel_f1, slot_acc, ent_acc, rel_acc, top1_f1, top3_f1 = multitask_fit(model=model,
                                                                                                         training_iter=train_iter,
                                                                                                         eval_iter=eval_iter,
                                                                                                         num_epoch=args.num_train_epochs,
                                                                                                         pbar=pbar,
                                                                                                         num_train_steps=num_train_steps,
                                                                                                         k=i,
                                                                                                         verbose=1)

        losses.append(eval_loss)
        slot_seq_f1.append(slot_f1)
        ent_cls_f1.append(ent_f1)
        rel_cls_f1.append(rel_f1)
        overall_top1_f1.append(top1_f1)
        overall_top3_f1.append(top3_f1)

    # calculate k-fold result
    with open(args.k_fold_report_path, 'w') as f:
        print('%d-fold result:\n' % k, file=f)
        print('eval_loss: ', losses, file=f)
        print('slot_f1: ', slot_seq_f1, file=f)
        print('ent_f1: ', ent_cls_f1, file=f)
        print('rel_f1: ', rel_cls_f1, file=f)
        print('top1_acc: ', overall_top1_f1, file=f)
        print('top3_acc: ', overall_top3_f1, file=f)
        print('aver top1: ', sum(overall_top1_f1) /
              len(overall_top1_f1), file=f)
