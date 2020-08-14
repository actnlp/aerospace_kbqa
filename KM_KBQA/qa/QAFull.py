import logging

from ..common.QCLS import QCLSWrapper
from ..config import config
from .QA import QA

logger = logging.getLogger('qa')


class QAFull():
    def __init__(self):
        self.kbqa = QA()
        self.qcls = QCLSWrapper.from_pretrained(config.QCLS_PATH)

    def answer(self, sent):
        # kbqa_prob = self.qcls.eval([sent])[0]
        # if kbqa_prob < config.check_kbqa_ths:
        #     ans = [{'natural_ans': '非KBQA问题'}]
        #     logger.info('非KBQA问题: %s %f' % (sent, kbqa_prob))
        # else:
        #     ans = self.kbqa.answer(sent)
        ans = self.kbqa.answer(sent)
        return ans
