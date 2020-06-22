from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer

pretrained_bert_path = 'KM_KBQA/models/bert-base-chinese.bin'
pretrained_vocab_path = 'KM_KBQA/models/vocab.txt'
pretrained_config_path = 'KM_KBQA/models/bert_config.json'

BertModel.pretrained_model_archive_map['bert-base-chinese'] = pretrained_bert_path
BertTokenizer.pretrained_vocab_files_map['vocab_file']['bert-base-chinese'] = pretrained_vocab_path
BertConfig.pretrained_config_archive_map['bert-base-chinese'] = pretrained_config_path

hit_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
hit_model = BertModel.from_pretrained('bert-base-chinese')
device = 'cpu'
hit_model.to(device)
hit_model.eval()


def get_padding_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.ones(lengths.size(0), max_len,
                      dtype=torch.int, device=lengths.device)
    m = torch.arange(0, max_len) >= lengths[:, None]
    mask[m] = 0
    return mask


@lru_cache(maxsize=4096)
def encode(sent):
    # input_ids = torch.tensor(hit_tokenizer.encode(sent)
    #                          ).unsqueeze(0).to(device)
    # # with torch.no_grad():
    # outputs = hit_model(input_ids)[0][0]
    # outputs = torch.mean(outputs[1:-1], dim=0)
    # outputs = [outputs.tolist()]
    # print(outputs)
    # return outputs
    return encode_batch(sent)[0]


def encode_batch(sents):
    encode_tensors = hit_tokenizer.batch_encode_plus(sents,
                                                     add_special_tokens=True,
                                                     return_tensors='pt',
                                                     return_input_lengths=True)
    input_ids, lengths = encode_tensors['input_ids'], encode_tensors['input_len']
    padding_mask = get_padding_mask(lengths)
    with torch.no_grad():
        outputs = hit_model(input_ids, attention_mask=padding_mask)[0]
        outputs.masked_fill_(padding_mask[:, :, None] == 0, 0)
        outputs = outputs[:, 1:-1, :].sum(dim=1)/(lengths[:, None]-2)
    return outputs


@lru_cache(maxsize=4096)
def cosine_word_similarity(w1, w2):
    vec1, vec2 = encode(w1), encode(w2)
    return F.cosine_similarity(vec1, vec2, dim=0).item()


if __name__ == '__main__':
    # encode('黄花机场的卫生间位置')
    res = encode_batch(['黄花机场的卫生间位置', '机场的卫生间位置'])
    print(res)
