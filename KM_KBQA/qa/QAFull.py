import logging

from ..common.QCLS import QCLSWrapper
from ..config import config
from .QA import QA

logger = logging.getLogger('qa')


class QAFull():
    def __init__(self):
        self.kbqa = QA()
        # self.qcls = QCLSWrapper.from_pretrained(config.QCLS_PATH)

    def not_kbqa_question(self,question):
        for word in config.NO_AIR_QUESTION_WORDS:
            if word in question:
                return "非民航问题"
        for word in config.CQA_QUESTION_WORDS:
            if word in question:
                return "非KBQA问题"
        return False

    def contain_freq_word(self, question):
        for word in config.FREQUENT_WORDS:
            if word in question:
                return True
        if "航空" in question and "公司" not in question:
            return True
        return False

    def answer(self, sent):
        # 拒识
        try:
            no_kbqa_flag = self.not_kbqa_question(sent)
            if no_kbqa_flag:
                # ans = [{'natural_ans': not_kbqa_flag}]
                ans = [no_kbqa_flag]
                ans = [{'answers':no_kbqa_flag}]
                logger.info('%s: %s' % (no_kbqa_flag,sent))
            else:
                ans = self.kbqa.answer(sent)
            return ans
        except Exception as e:
            return [e]