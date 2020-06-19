
from fuzzywuzzy import fuzz

from ..config import config
from ..common import LTP


def generate_ngram(seq, max_size=6, min_size=1, ignore_w=['服务']):
    for n in range(min_size, max_size + 1):
        for j in range(len(seq) - n + 1):
            # delete word consists of two single token or in ignore_w
            cur_w = ''.join(seq[j:j + n])
            if len(cur_w) <= 1:
                if cur_w in ['丢', '住', '吃', '喝', '换', '停']:
                    yield (cur_w)
                continue
            if '服务' == cur_w:
                continue
            for ig_w in ignore_w:
                if ig_w in cur_w:
                    yield (''.join(cur_w.split(ig_w)))
            # if len(list(filter(lambda x: x in cur_w, ignore_w))) > 0:
            #     continue
            # if ''.join(seq[j:j + n]) in ignore_w:
            #     continue
            if n == 2 and len(seq[j]) == 1 and len(seq[j + 1]) == 1 and seq[j] != '几':
                continue
            yield (cur_w)


def merge_ngram(cut_tokens):
    # obtain all air_lexicon
    air_lexicons = []
    with open(config.AIR_LEXICON_PATH, 'r') as f:
        for line in f:
            air_lexicons.append(line.split()[0])
    # match and merge tokens
    pos_list = LTP.pos_tag_tokens(cut_tokens)
    merged_list = []
    for air_lexicon in air_lexicons:
        for i, token_i in enumerate(cut_tokens):
            for j, token_j in enumerate(cut_tokens[i + 1:]):
                if len(token_i) <= 1 or len(token_j) <= 1 or pos_list[i] == '' or pos_list[
                        j + i + 1] == '' or '机场' in token_i or '机场' in token_j:
                    continue
                new_token_ij, new_token_ji = token_i + token_j, token_j + token_i
                for new_token in [new_token_ij, new_token_ji]:
                    score = fuzz.ratio(air_lexicon, new_token)
                    if score > 70:
                        if new_token not in merged_list and token_i not in air_lexicons and token_j not in air_lexicons:
                            merged_list.append(new_token)
    return merged_list


def retrieve_ngram(sent):
    # n-gram extraction
    cut_tokens = LTP.customed_jieba_cut(sent, config.STOP_WORD_PATH)
    n_gram_list = list(generate_ngram(cut_tokens, 1))
    # merge two token if the combined one is similar to the entity in the air_lexicon
    merged_list = merge_ngram(n_gram_list)
    return n_gram_list + merged_list


retrieve_funcs = [retrieve_ngram]


def recognize_entity(sent, merge=True):
    retrieve_res = [func(sent) for func in retrieve_funcs]
    if merge:
        mention_set = set()
        mention_set.update(*retrieve_res)
        mention_list = list(mention_set)
        return mention_list
    else:
        return retrieve_res
