from ..linking import NewLinking
from ..common import LTP
from .Limiter import Limiter
import logging
from ..common import AsyncNeoDriver
from .RelExtractor import BertRelExtractor, MatchRelExtractor


logger = logging.getLogger()

# 过滤词汇


def filter_sentence(sent, stop_list, rpl_list):
    for stop in stop_list:
        sent = sent.replace(stop, '')
    for replace in rpl_list:
        sent = sent.replace(replace[0], replace[1])
    return sent


class QA():
    stop_list = ['吗', '里面']
    rpl_list = [('在哪儿', '的地点'), ('在哪里', '的地点'), ('在哪', '的地点'), ('哪里',
                                                                '地点'), ('哪里', '地点'), ('哪有', '地点'), ('属于', '在'), ('vip', '贵宾')]

    def __init__(self):
        self.driver = AsyncNeoDriver.get_driver()
        self.bert_linker = NewLinking.BertLinker()
        self.commercial_linker = NewLinking.CommercialLinker()
        self.rule_linker = NewLinking.RuleLinker()
        self.match_extractor = MatchRelExtractor()
        self.bert_extractor = BertRelExtractor()

    def link(self, sent):
        bert_res = self.bert_linker.link(sent)
        commercial_res = self.commercial_linker.link(sent)
        rule_res = self.rule_linker.link(sent)
        link_res = rule_res
        id2linked_ent = {linked_ent['id']: linked_ent for linked_ent in link_res}
        # merge bert res
        for linked_ent in commercial_res:
            neoId = linked_ent['id']
            if neoId in id2linked_ent:
                id2linked_ent[neoId]['score'] += linked_ent['score']
                id2linked_ent[neoId]['source'] += ' '+linked_ent['source']
            else:
                id2linked_ent[neoId] = linked_ent
                link_res.append(linked_ent)

        # merge commecial res
        for linked_ent in bert_res:
            neoId = linked_ent['id']
            if neoId in id2linked_ent:
                id2linked_ent[neoId]['score'] += linked_ent['score']
                id2linked_ent[neoId]['source'] += ' '+linked_ent['source']
            else:
                id2linked_ent[neoId] = linked_ent
                link_res.append(linked_ent)
        link_res.sort(key=lambda x:x['score'], reverse=True)
        return link_res

    def get_instances(self, parent_name, parent_label, instance_label):
        instances = self.driver.get_genres_by_relation(
            parent_label, instance_label, parent_name, reverse=True).result()
        return instances

    def extract_rel(self, sent: str, sent_cut: list, link_res):
        # 获取具体实体
        link_res_extend = []
        for linked_ent in link_res:
            ent = linked_ent['ent']
            if ent['label'] == 'SubGenre' or ent['label'] == 'Genre':
                instances = self.get_instances(
                    ent['name'], ent['label'], 'Instance')
                if instances is not None:
                    link_res_extend.extend([{
                        'ent': e,
                        'mention': linked_ent['mention'],
                        'id':e['neoId'],
                        'score':linked_ent['score'],
                        'source':linked_ent['source']+' sub',
                    } for e in instances])
            elif ent['label'] == 'Instance':
                link_res_extend.append(linked_ent)
        # 抽取关系
        rel_match_res = []
        for linked_ent in link_res_extend:
            ent = linked_ent['ent']
            # 匹配关系
            rel_matched = self.match_extractor.extract_rel(
                sent_cut, linked_ent)
            if rel_matched is not None:
                rel_match_res += rel_matched
            # bert 提取关系
            rel_bert_res = self.bert_extractor.extract_rel(sent, linked_ent)
            for bert_rel in rel_bert_res:
                for match_rel in rel_match_res:
                    if bert_rel['rel_name'] == match_rel['rel_name']:
                        match_rel['rel_score'] += 0.3
                    else:
                        rel_match_res.append(bert_rel)
        return rel_match_res

    def answer(self, sent):
        '''
            问答流程：
             1. 使用替换规则修改问句并分词
             2. 规则提取限制
             3. 三种linker提取实体，合并排序
             4. 判断是否列举型
             5. 非列举型匹配关系
             6. 如果没有匹配到的关系，且实体识别分值较高，算作列举
             7. 匹配限制
             8. 答案排序
             9. 匹配机场本身相关问题
             10. 返回答案
        '''
        logger.info('question :%s' % sent)
        # 1. 使用替换规则
        sent = filter_sentence(sent, self.stop_list, self.rpl_list)
        sent_cut = LTP.customed_jieba_cut(sent, cut_stop=True)
        logger.info('cut :'+str(sent_cut))

        # 2. extract constaints
        limits = Limiter(sent).check()
        logger.info('限制: '+str(limits))

        # 3. link: 使用多种linker，合并结果
        # TODO 多linker合并
        # link_res = self.rule_linker.link(sent, limits)
        link_res = self.link(sent)
        logger.info('链接结果: '+str(link_res))
        # TODO 4. 处理列举类型

        # 5. 非列举型匹配关系 extract relations, match+bert
        rel_match_res = self.extract_rel(sent, sent_cut, link_res)
        logger.info('关系匹配: '+str(rel_match_res))

        # TODO 6. 如果没有匹配到的关系，且实体识别分值较高，算作列举
        # TODO 7. 匹配限制
        # TODO 8. 答案排序
        # TODO 9. 匹配机场本身相关的问题
        # TODO 10. select answer


if __name__ == "__main__":
    qa = QA()
    questions = ['打火机可以携带吗？', '机场有婴儿车可以租用吗', '机场有轮椅可以租用吗',
                 '停车场怎么收费', '停车场费用怎么样？', '停车场一个小时多少钱？', '停车场多少钱？']
    for q in questions:
        qa.answer(q)
