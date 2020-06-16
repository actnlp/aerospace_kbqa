import pdb
import re

from fuzzywuzzy import fuzz

from ..config import config
from ..common import LTP, AsyncNeoDriver

# logging.basicConfig(level=logging.DEBUG)
entity_pattern = r'《(.+)》|<(.+)>|“(.+)”|"(.+)"|\'(.+)\''
entity_set = None


def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a & b) / len(a | b)


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
            if n == 2 and len(seq[j]) == 1 and len(seq[j + 1]) == 1 and seq[j] != '几': continue
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


def retrieve_rule(sent):
    # retrieve special tokens
    mentions = map(lambda x: next(filter(lambda l: l != '', x)),
                   re.findall(entity_pattern, sent))
    return mentions


retrieve_funcs = [retrieve_rule, retrieve_ngram]


def recognize_entity(sent, merge=True):
    retrieve_res = [func(sent) for func in retrieve_funcs]
    if merge:
        mention_set = set()
        mention_set.update(*retrieve_res)
        mention_list = list(mention_set)
        return mention_list
    else:
        return retrieve_res


class FeatureVectorizer():
    def __init__(self, bert_ranker=None):
        self.bert_ranker = bert_ranker

    def vectorize(self, sent, mention_list, retrieve_res):
        vectors = []
        if self.bert_ranker:
            _, bert_res = self.bert_ranker.rank(sent, mention_list)
            for m, s in zip(mention_list, bert_res):
                # mention source
                vec = [1 if m in rtr_set else 0 for rtr_set in retrieve_res]
                vec.append(jaccard(m[0], sent))  # jaccard distance
                vec.append(m[1])  # mention location
                vec.append(s[0])
                vectors.append(vec)
        else:
            for m in mention_list:
                # mention source
                vec = [1 if m in rtr_set else 0 for rtr_set in retrieve_res]
                vec.append(jaccard(m[0], sent))  # jaccard distance
                vec.append(m[1])  # mention location
                vectors.append(vec)
        return vectors


keep_keys = ['name', 'label', 'neoId', 'taglist']


class Linker():
    def __init__(self, hangxin=False, driver=None):
        if driver:
            self.driver = driver
        else:
            if not hangxin:
                self.driver = AsyncNeoDriver.AsyncNeoDriver()
            else:
                self.driver = AsyncNeoDriver.AsyncNeoDriver(
                    server_address=r'http://10.1.1.30:7474',
                    entity_label='Instance1')

    def rank(self, sent):
        pass

    def exist_mention(self, cand_mention):
        if_exist = list(map(lambda x: x.result(),
                            map(self.driver.exist_name,
                                cand_mention)))
        # logging.info(if_exist)
        mention_list = list(map(lambda x: x[1],
                                filter(lambda x: x[0],
                                       zip(if_exist, cand_mention))))
        return mention_list


if __name__ == '__main__':
    s = ['告诉我姚明的女儿是谁？', '请问北航的校长是谁？', '陈赫和王传君共同主演的电视剧是什么？', '王传君和陈赫共同主演的电视剧是什么？',
         '城关镇在哪？', '《线性代数》这本书的出版时间是什么？', '告诉我高等数学的出版时间是什么时候？', '告诉我中国人民大学的校长是谁？', '北京大学出了哪些哲学家？']
    bert_model_name = 'm.25+pos+data'
    retrieve_res = [func(sent) for func in retrieve_funcs]
    print(retrieve_res[0])
    print(retrieve_res[1])
    print(retrieve_res[2])
    print('hello')
