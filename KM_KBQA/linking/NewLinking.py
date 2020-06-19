'''
    实体链接
    link
    返回列表：[dict({info_dict, mention, 抽取方式, 分值})]
'''
import jieba
from fuzzywuzzy import fuzz, process

from ..BertEntityRelationClassification.BertERClsPredict import \
    predict as BertERCls
from ..common import AsyncNeoDriver, LTP


class RuleLinker():
    def link(self, sent):
        pass


class BertLinker():
    def __init__(self, driver=None):
        if driver is None:
            self.driver = AsyncNeoDriver.get_driver(name='default')
        else:
            self.driver = driver

    def link(self, sent):
        _, _, ent_type_top3 = BertERCls(sent)
        print(ent_type_top3)
        instances_top3 = [self.driver.get_instance_of_genre(ent_type)
                          for ent_type in ent_type_top3]
        res = []
        for rank, instances in enumerate(instances_top3):
            for e in instances:
                ent = {
                    'ent': e,
                    'id': e['neoId'],
                    'rank': rank+1,
                    'source': 'bert'
                }
                res.append(ent)
        return res


class CommercialLinker():
    def __init__(self, driver=None):
        if driver is None:
            self.driver = AsyncNeoDriver.get_driver(name='default')
        else:
            self.driver = driver
        self.content2entId = self.build_revert_index()

    def link(self, sent):
        content_keys = self.retrieve_content_keys(sent)
        res = []
        for content, score in content_keys:
            ent_ids = self.content2entId[content]
            for ent_id in ent_ids:
                e = self.driver.get_entity_by_id(ent_id).result()[0]
                ent = {
                    'ent': e,
                    'id': ent_id,
                    'score': score,
                    'source': 'commercial',
                    'content': content
                }
                res.append(ent)
        return res

    def build_revert_index(self):
        entities = self.driver.get_entities_by_genre('Instance').result()
        # entities += self.driver.get_entities_by_genre('SubGenre').result()
        content2entId = {}
        for ent in entities:
            ent_id = ent['neoId']
            content_str = ent.get('服务内容', '')
            content = content_str.split(';')
            for c in content:
                if c == '':
                    continue
                if c in content2entId:
                    content2entId[c].append(ent_id)
                else:
                    content2entId[c] = [ent_id]
        return content2entId

    def retrieve_content_keys(self, sent):
        words = LTP.customed_jieba_cut(sent)
        sent = ''.join(words)
        res = process.extract(sent, self.content2entId.keys(),
                            scorer=fuzz.UQRatio,
                               limit=2)

        print(words)
        print(res)
        return res


def test_bert_linker():
    bert_linker = BertLinker()
    sent = '有可以玩游戏的地方吗？'
    # sent = '东航的值机柜台在哪？'
    res = bert_linker.link(sent)
    print(res)


def test_commercial_linker():
    commercial_linker = CommercialLinker()
    # print(commercial_linker.content2entId)
    sent = '有可以打游戏的地方吗？'
    # sent = '东航的值机柜台在哪？'
    res = commercial_linker.link(sent)
    print(res)


if __name__ == '__main__':
    test_bert_linker()
    # test_commercial_linker()
