import os

import torch
from transformers import BertConfig, BertModel, BertTokenizer

from ..config import config
from . import args
from .MultiTaskKBQA import MultiTaskKBQA

pretrained_bert_path = 'KM_KBQA/models/bert-base-chinese.bin'
pretrained_vocab_path = 'KM_KBQA/models/vocab.txt'
pretrained_config_path = 'KM_KBQA/models/bert_config.json'

BertModel.pretrained_model_archive_map['bert-base-chinese'] = pretrained_bert_path
BertTokenizer.pretrained_vocab_files_map['vocab_file']['bert-base-chinese'] = pretrained_vocab_path
BertConfig.pretrained_config_archive_map['bert-base-chinese'] = pretrained_config_path

global_tokenizer = None
global_model = None


def get_tokenizer():
    global global_tokenizer
    if global_tokenizer is None:
        # global_tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)
        global_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    return global_tokenizer


def get_model():
    global global_model
    if global_model is None:
        global_model = MultiTaskKBQA(args).to(args.device)
        global_model.load_state_dict(torch.load(os.path.join(
            config.model_path, "BertERCls.bin"), map_location='cpu'))
    return global_model


def predict(sent):
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
