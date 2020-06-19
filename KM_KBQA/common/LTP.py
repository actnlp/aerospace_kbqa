import os
import pdb

import jieba
import jieba.posseg as pseg

from ..config.config import LEX_PATH, STOP_WORD_PATH

# cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
# par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
# ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')

jieba.load_userdict(os.path.join(LEX_PATH, 'air_lexicon.txt'))
jieba.enable_paddle()

# segmentor = Segmentor()
# segmentor.load_with_lexicon(cws_model_path, os.path.join(LEX_PATH, 'air_lexicon.txt'))

# postagger = Postagger()
# postagger.load_with_lexicon(pos_model_path, os.path.join(LEX_PATH, 'air_lexicon.txt'))

# parser = Parser()
# parser.load(par_model_path)

# recognizer = NamedEntityRecognizer()
# # recognizer.load_with_lexicon(ner_model_path, os.path.join(LEX_PATH, 'air_lexicon.txt'))
# recognizer.load(ner_model_path)


'''def parse(sent):
    words = list(segmentor.segment(sent))
    postags = list(postagger.postag(words))
    parses = list(parser.parse(words, postags))

    tree_str = '\n'.join(
        ['%s\t%s\t%d\t%s' % (word, tag, parse.head, parse.relation)
         for word, tag, parse in zip(words, postags, parses)]
    )
    graph = DependencyGraph(tree_str, top_relation_label='HED')
    return graph'''


def cut(sent):
    return list(jieba.cut(sent, cut_all=False))


def load_stopwords(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        stopwords = {line.strip() for line in f}
    return stopwords


stopwords = load_stopwords(STOP_WORD_PATH)


def customed_jieba_cut(sent, path=None, cut_stop=True):
    # with open(path, 'r', encoding='utf-8') as f:
        # stopwords = [line.replace('\n', '') for line in f]
    cut_sent = list(jieba.cut(sent, cut_all=False))

    for i, wd in enumerate(cut_sent):
        if len(wd) > 2 and wd[-1] == '费':
            cut_sent[i] = wd[:-1]
            cut_sent.append('费用')

    if not cut_stop:
        return cut_sent
    ret_cut_sent = []
    for wd in cut_sent:
        if wd not in stopwords and wd != ' ':
            ret_cut_sent.append(wd)

    return ret_cut_sent


def pos_tag_tokens(tokens):
    # add pos tag to the splited word
    pos = []
    for token in tokens:
        tag = [list(x) for x in pseg.cut(token)]
        # 如果分割出来多个词或者不是 n./v.
        if len(tag) > 1 or ('n' not in tag[0][1] and 'v' not in tag[0][1]) or '有' in tag[0][0]:
            pos.append('')
        else:
            pos.append(tag[0][1])
    return pos


# def customed_ltp_cut(sent, path, cut_stop=True):
#     with open(path, 'r', encoding='utf-8') as f:
#         stopwords = [line.replace('\n', '') for line in f]
#     cut_sent = cut(sent)
#     print('customed_ltp: ', cut_sent)
#     merge_sent = []
#     i = 0
#     while i < len(cut_sent) - 1:
#         merge_w = cut_sent[i] + cut_sent[i + 1]
#         '''if ('长沙' in cut_sent[i] or '黄花' in cut_sent[i]) and '机场' in cut_sent[i + 1]:
#             merge_sent.append(merge_w)
#             i += 2
#         else:'''
#         merge_sent.append(cut_sent[i])
#         i += 1
#     merge_sent.append(cut_sent[-1])
#     if not cut_stop:
#         return merge_sent
#     ret_cut_sent = []
#     for wd in merge_sent:
#         if wd not in stopwords:
#             ret_cut_sent.append(wd)

#     return ret_cut_sent


# def ner(sent):
#     words = list(segmentor.segment(sent))
#     postags = list(postagger.postag(words))
#     netags = list(recognizer.recognize(words, postags))
#     return list(zip(words, netags))

# def customed_ner(sent, path):
#     words = customed_jieba_cut(sent, path)

#     postags = list(postagger.postag(words))
#     netags = list(recognizer.recognize(words, postags))
#     return list(zip(words, netags))


# def postag(sent):
#     words = list(segmentor.segment(sent))
#     postags = list(postagger.postag(words))
#     return list(zip(words, postags))

# def plain_postag(words):
#     return list(postagger.postag(words))

# def filter_postag(word_list, tag):
#     tag_list = postagger.postag(word_list)
#     word_tag = list(zip(word_list, tag_list))
#     tmp = list(filter(lambda x: x[1] == tag, word_tag))
#     filtered_word = list(map(lambda y: y[0], tmp))
#     return filtered_word
