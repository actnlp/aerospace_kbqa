from functools import lru_cache

import pandas as pd
import numpy as np
import torch,time,json
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer

all_entity_encode_path = 'KM_KBQA/res/entity_encode_1029.txt'
pretrained_bert_path = 'KM_KBQA/models'

hit_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_path)
hit_model = BertModel.from_pretrained(pretrained_bert_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
hit_model.to(device)
hit_model.eval()

all_entity_encode = json.loads(open(all_entity_encode_path).read())
all_entity_encode = {k:torch.Tensor(v).to(device) for k,v in all_entity_encode.items()}
mention_encode = {}

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
    # # print(outputs)
    # return outputs
    return encode_batch(sent)[0]


def encode_batch(sents):
    encode_tensors = hit_tokenizer.encode_plus(sents,
                                               add_special_tokens=True,
                                               return_tensors='pt',
                                               pad_to_max_length=True)
    input_ids = encode_tensors['input_ids']
    lengths = torch.from_numpy(np.array([input_ids.shape[1] for _ in range(input_ids.shape[0])])) 
# padding_mask = get_padding_mask(lengths)  # transformers==2.3.0
    padding_mask = encode_tensors['attention_mask']  # transformers>3.0
    with torch.no_grad():
        outputs = hit_model(input_ids.to(device), attention_mask=padding_mask.to(device))[0]
        outputs.masked_fill_(padding_mask[:, :, None].to(device) == 0, 0)
        outputs = outputs[:, 1:-1, :].sum(dim=1).to(device)/(lengths[:, None]-2).to(device)
    return outputs


@lru_cache(maxsize=4096)
def cosine_word_similarity(w1, w2):
    if w1 not in mention_encode:
        vec1 = encode(w1)
        mention_encode[w1] = vec1
    else:
        vec1 = mention_encode[w1]

    if w2 not in all_entity_encode:
        vec2 = encode(w2)
        all_entity_encode[w2] = vec2
    else:
        vec2 = all_entity_encode[w2]

    return F.cosine_similarity(vec1, vec2, dim=0).item()


if __name__ == '__main__':
    # df = pd.read_csv("KM_KBQA/lexicons/aerospace_lexicon.txt",sep="\t",header=None)
    # all_entity_encode_f = open(all_entity_encode_path,"w")
    # all_entity_encode_f = open(all_entity_encode_path,"a")
    #
    # dict = {}
    # for tmp in df[0]:
    #     entity = tmp.split(" ")[0]
    #     encode = encode_batch(entity)[0]
    #     if entity not in dict:
    #         dict[entity] = encode.numpy().tolist()
    # all_entity_encode_f.write(json.dumps(dict,ensure_ascii=True))
    print("OK")


