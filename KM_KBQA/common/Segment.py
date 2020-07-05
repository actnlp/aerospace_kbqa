import os

import jieba
import jieba.posseg as pseg

from ..config.config import LEX_PATH

jieba.load_userdict(os.path.join(LEX_PATH, 'air_lexicon.txt'))
jieba.enable_paddle()


def cut(sent):
    return list(jieba.cut(sent, cut_all=False))


def pos_cut(sent):
    return list(zip(*pseg.cut(sent)))
