import logging
import pandas as pd
# from ..common.QCLS import QCLSWrapper
from ..config import config
from .QA import QA
# from ..ERNIEReject.torch_utils import load_config
# from ..ERNIEReject.loader_foronetext import DataLoader
# from ..ERNIEReject.trainer import MyTrainer
# import requests, os
from ..ERNIEReject import kbqa_reject

logger = logging.getLogger('qa')


class QAFull():
    def __init__(self):
        self.kbqa = QA()
        # self.qcls = QCLSWrapper.from_pretrained(config.QCLS_PATH)

    def question_type(self, question):
        res = kbqa_reject.query_type_infer(question)
        return res

    def not_kbqa_question(self, question):
        for word in config.NO_AIR_QUESTION_WORDS:
            if word in question:
                if word == '驾驶' and '仪' in question:
                    continue
                return "非民航问题"
        if self.question_type(question) == 1:
            return "非KBQA问题"
        return False

    def answer(self, sent):
        # 拒识
        try:
            no_kbqa_flag = self.not_kbqa_question(sent)
            if no_kbqa_flag:
                ans = [{'answers': no_kbqa_flag}]
                logger.info('%s: %s' % (no_kbqa_flag, sent))
            else:
                ans = self.kbqa.answer(sent)
            return ans
        except Exception as e:
            logger.info('%s: %s' % (str(e.args), sent))
            return []
