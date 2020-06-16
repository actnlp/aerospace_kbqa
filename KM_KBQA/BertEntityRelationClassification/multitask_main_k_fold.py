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


global_tokenizer = None
global_model = None


def get_tokenizer():
    global global_tokenizer
    if global_tokenizer is None:
        global_tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)
    return global_tokenizer


def get_model():
    global global_model
    if global_model is None:
        global_model = MultiTaskKBQA(args).to(args.device)
        global_model.load_state_dict(torch.load(os.path.join(
            args.trained_dir, "pytorch_model.bin"), map_location='cpu'))
    return global_model


def multitask_predict(sent):
    # get input_ids, input_mask and segment_ids from sent
    tokenizer = get_tokenizer()
    token = list(sent.lower())
    tokens = ["[CLS]"] + token + ["[SEP]"]

    segment_ids = [0] * len(tokens)
    # 词转换成数字
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    padding = [0] * (args.max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    # convert to torch
    input_ids, input_mask, segment_ids = torch.tensor([input_ids], device=args.device), torch.tensor([input_mask],
                                                                                                     device=args.device), torch.tensor(
        [segment_ids], device=args.device)
    # load model
    model = get_model()

    with torch.no_grad():
        model.eval()
        # [1, 200, 7]
        _, ent_output, rel_output = model(
            input_ids, segment_ids, input_mask, batch_idx=0)
        ent_top1, ent_top3 = model.predict_ent_classify(ent_output)
        rel_top1, rel_top3 = model.predict_rel_classify(rel_output, ent_top3)

    ent_string = args.ENT_LABELS[ent_top1.item()]
    ent_top3_string = [args.ENT_LABELS[ent_top3[0][0].item(
    )], args.ENT_LABELS[ent_top3[0][1].item()], args.ENT_LABELS[ent_top3[0][2].item()]]
    rel_string = args.REL_LABELS[rel_top1.item()]

    return ent_string, rel_string, ent_top3_string
