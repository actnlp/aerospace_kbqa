from functools import lru_cache

from fuzzywuzzy import fuzz

from ..config import config

air_lexicons = set()
with open(config.AIR_LEXICON_PATH, 'r') as f:
    for line in f:
        air_lexicons.add(line.split()[0])


@lru_cache(maxsize=4096)
def cal_ratio(w1, w2):
    return fuzz.ratio(w1, w2)


def generate_unigram(sent_cut, ignore_words=['服务']):
    unigram = []
    for w in sent_cut:
        if len(w) <= 1:
            if w in {'丢', '住', '吃', '喝', '换', '停'}:
                unigram.append(w)
            continue
        if w == '服务':
            continue
        for i_w in ignore_words:
            if i_w in w:
                unigram.append(w.replace(i_w, ''))
                break
        else:
            unigram.append(w)
    return unigram


def generate_bigram(sent_cut, pos_tag):
    n = len(sent_cut)
    bigram = [''.join(sent_cut[i:i+2]) for i in range(n-1)]
    pos_tag = [tag if 'n' in tag or 'v' in tag
               else ''
               for tag in pos_tag]
    used_words = [sent_cut[i] for i in range(n)
                  if len(sent_cut[i]) > 1 and
                  pos_tag[i] != '' and
                  '有' not in sent_cut[i] and
                  '机场' not in sent_cut[i]
                  and sent_cut[i] not in air_lexicons  # TODO 这个是为啥？？
                  ]
    cand = []
    for w1 in used_words:
        for w2 in used_words:
            if w1 == w2:
                continue
            cand.append(w1+w2)
    # bigram = []
    for w in used_words:
        for air_lexicon in air_lexicons:
            score = cal_ratio(w, air_lexicon)
            if score > 70:
                bigram.append(w)
                break
    return bigram


def retrieve_mention(sent_cut, pos_tag):
    # unigram
    unigram = generate_unigram(sent_cut)
    # bigram
    bigram = generate_bigram(sent_cut, pos_tag)
    mentions = list(set(unigram+bigram))
    return mentions
