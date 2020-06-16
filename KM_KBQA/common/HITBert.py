import pdb

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

pretrained_bert_path = 'KM_KBQA/models/pytorch_model.bin'
pretrained_vocab_path = 'KM_KBQA/models/vocab.txt'
pretrained_config_path = 'KM_KBQA/models/bert_config.json'

BertModel.pretrained_model_archive_map['bert-base-chinese'] = pretrained_bert_path
BertTokenizer.pretrained_vocab_files_map['vocab_file']['bert-base-chinese'] = pretrained_vocab_path
BertConfig.pretrained_config_archive_map['bert-base-chinese'] = pretrained_config_path

hit_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
hit_model = BertModel.from_pretrained('bert-base-chinese')
hit_model.cuda()
hit_model.eval()

def encode(sent):
    input_ids = torch.tensor(hit_tokenizer.encode(sent)).cuda().unsqueeze(0)
    # with torch.no_grad():
    outputs = hit_model(input_ids)[0][0]

    outputs = torch.mean(outputs[1:-1], dim=0)

    return [outputs.tolist()]

if __name__ == '__main__':
    encode('黄花机场的卫生间位置')
