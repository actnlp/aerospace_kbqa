
import os
from ltp import LTP
from ..config.config import LEX_PATH
ltp = LTP(path = "base")

ltp.init_dict(path=os.path.join(LEX_PATH, 'aerospace_lexicon.txt'))

def cut(sent):
    segment, _ = ltp.seg([sent])
    return segment[0]

def pos_cut(sent):
    segment, hidden = ltp.seg([sent])
    pos = ltp.pos(hidden)
    return [tuple(segment[0]), tuple(pos[0])]