# coding:utf-8
import asyncio
import json
import logging
import pdb
import pickle
import threading
from functools import lru_cache

import aiohttp
import torch


driver_pool = {}


def get_driver(**kwargs):
    global driver_pool
    name = kwargs.get('name', 'default')
    if name in driver_pool:
        return driver_pool[name]
    else:
        driver = AsyncNeoDriver(**kwargs)
        driver_pool[name] = driver
        return driver


# logging.basicConfig(level=logging.INFO)
headers = {'Authorization': 'bmVvNGo6MTIzNDU2',
           'Accept': 'application/json; charset=UTF-8',
           'Content-Type': 'application/json'}

default_address = r'http://10.1.1.28:7979'

first_genre_label = 'Genre'
second_genre_label = 'SubGenre'
entity_label = 'Instance'


class AsyncNeoDriver():
    def __init__(self,
                 server_address=default_address,
                 entity_label='Instance',
                 fuzzy_index=None,
                 name='default'):
        self.server_address = server_address
        self.entity_label = entity_label
        self.fuzzy_index = fuzzy_index
        self.query_url = '%s/db/data/transaction/commit' % self.server_address
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True)   # daemon: 守护线程，如果剩下的线程只有守护线程，整个程序就会退出
        self.loop_thread.start()
        timeout = aiohttp.ClientTimeout(total=None)
        conn = aiohttp.TCPConnector(
            limit=256, keepalive_timeout=4096, loop=self.loop)
        self.session = aiohttp.ClientSession(            # 如果loop参数为None，则在connector参数中获取
            connector=conn, timeout=timeout, headers=headers)
        self.entity_id_cache = {}
        self.entity_name_cache = {}
        self.relation_cache = {}
        self.all_entity_embeddings = {}

    async def execute_async(self, query):
        payload = {'statements': [{'statement': query}]}
        try_time = 0
        while try_time < 3:
            try_time += 1
            try:
                async with self.session.post(self.query_url, json=payload) as response:
                    return json.loads(await response.text())['results'][0]
            except:
                await asyncio.sleep(0.1)
        return None

    def execute(self, query):
        '''
        the query running on neo4j server's command line
        '''
        return asyncio.run_coroutine_threadsafe(self.execute_async(query), self.loop)

    @staticmethod
    def process_result(result):
        try:
            res = []
            for a in result['data']:
                a['row'][0]['neoId'] = str(a['row'][1])
                res.append(a['row'][0])
            assert len(result['data']) == len(res)
            return res
        except:
            # print('error at %s val %s\n result : %s'%(property,value,result))
            return None

    def get_entities_by_matching_property(self, property, value):
        return asyncio.run_coroutine_threadsafe(
            self.get_entities_by_matching_property_async(property, value), self.loop)

    async def get_entities_by_matching_property_async(self, property, value):
        result = await self.execute_async('match (n:%s) where n.%s contains "%s" return n,id(n)' %
                                          (self.entity_label, property, value))
        return self.process_result(result)

    async def get_entities_by_property_async(self, property, value):
        # print('\n\nget_entity\n\n: ', property, value)
        result = await self.execute_async('match (n) where n.%s="%s" return n,id(n)' %
                                          (property, value))
        return self.process_result(result)

    async def get_all_entities_async(self, entity_label, filter_labels):
        # print('\n\nget_entity\n\n: ', property, value)
        # print('match (n:%s) return n' % (entity_label))
        processed_filters = ''
        for filter_label in filter_labels:
            processed_filters += 'not n.label contains \'%s\' and ' % (
                filter_label)
        if processed_filters != '':
            # 删去最后一个and 并在前面加上where语句
            processed_filters = processed_filters[:-5]
            processed_filters = 'where ' + processed_filters
        result = await self.execute_async('match (n:%s) %s return n, id(n)' %
                                          (entity_label, processed_filters))
        # print('entities number:', len(result['data']))
        return self.process_result(result)

    def get_all_entities(self, entity_label, filter_label=[]):
        return asyncio.run_coroutine_threadsafe(
            self.get_all_entities_async(entity_label, filter_label), self.loop)

    async def get_entities_by_name_async(self, name):
        '''
            return : list of entities
        '''
        if self.fuzzy_index:
            '''cur_name_embedding = self.bc.encode([name])
            res = []
            for key in self.all_entity_embeddings.keys():
                res.append((key, cal_word_simi(self.all_entity_embeddings[key], cur_name_embedding)))

            res = sorted(res, key=lambda x: x[1], reversed=True)'''

            print("CALL db.index.fulltext.queryNodes('%s','%s') yield node,score return node,id(node),score" % (
                self.fuzzy_index, name))
            result = await self.execute_async("CALL db.index.fulltext.queryNodes('%s','%s') yield node,score return node,id(node),score" % (self.fuzzy_index, name))
            # print('\nfuzzy_result: ', result)
            try:
                res = []
                thresh = 1.5
                max_score = 0
                for a in result['data']:
                    name = a['row'][0]['name']
                    if '黄花机场' in name or name == '机场':
                        continue
                    a['row'][0]['neoId'] = str(a['row'][1])
                    score = a['row'][2]
                    max_score = max(max_score, score)
                    a['row'][0]['score'] = score
                    if score > thresh:
                        res.append(a['row'][0])
                res = list(filter(lambda x: x['score'] == max_score, res))
                return res
            except:
                return None
        else:
            if name in self.entity_name_cache:  # and name != '机场' and '黄花机场' not in name:
                return self.entity_name_cache[name]
            else:
                r = await self.get_entities_by_property_async('name', name)
                r = list(filter(lambda x: x['name']
                                != '黄花机场' and x['name'] != '机场', r))
                # print('\n\nget_entitie_by_name_exact: ', r)
                self.entity_name_cache[name] = r
                for entity in r:
                    neoId = entity['neoId']
                    self.entity_id_cache[neoId] = [entity]
                return r

    def get_entities_by_name(self, name):
        '''
            return : list of entities
        '''
        return asyncio.run_coroutine_threadsafe(
            self.get_entities_by_name_async(name), self.loop)

    async def get_entity_by_id_async(self, neoId):
        '''
            return : list of entities
        '''
        if neoId in self.entity_id_cache:
            return self.entity_id_cache[neoId]
        else:
            r = await self.execute_async('match (n:%s) where id(n)=%s return n,id(n)' % (self.entity_label, neoId))
            r = self.process_result(r)
            self.entity_id_cache[neoId] = r
            return r

    def get_entity_by_id(self, neoId):
        '''
            return : list of entities
        '''
        return asyncio.run_coroutine_threadsafe(
            self.get_entity_by_id_async(neoId), self.loop)

    async def get_entity_information_by_label_async(self, label):
        result = await self.execute_async('match (n:%s) return id(n), n.name, properties(n)' % (label))
        res = list(map(lambda x: x['row'], result['data']))
        return res

    def get_entity_information_by_label(self, label):
        return asyncio.run_coroutine_threadsafe(
            self.get_entity_information_by_label_async(label), self.loop
        )

    def exist_name(self, name):
        async def _exist_name(name):
            r = await self.get_entities_by_name_async(name)
            return r is not None and r != []
        return asyncio.run_coroutine_threadsafe(_exist_name(name), self.loop)

    async def get_genres_by_relation_async(self, f_genre, c_genre, f_name, reverse=False):
        '''
            return : list of triples (relation, entity name, entity neoId)
        '''
        arrow1 = ''
        arrow2 = ''
        if reverse:
            arrow1 = '<'
        else:
            arrow2 = '>'
        res = await self.execute_async('match (n:%s)%s-[r]-%s(m:%s) where n.name =\'%s\' return distinct m, id(m)'
                                       % (f_genre, arrow1, arrow2, c_genre, f_name))
        # print('match (n:%s)%s-[r]-%s(m:%s) where n.name = \'%s\' return m, id(m)'% (f_genre, arrow1, arrow2, c_genre, f_name))
        # res = [x['name'] for x in result]
        # res = list(map(lambda x: tuple(x['row']), result['data']))
        # self.relation_cache[query_tuple] = res
        return self.process_result(res)

    def get_genres_by_relation(self, f_genre, c_genre, f_name, reverse=False):
        '''
            return : list of triples (relation, entity name, entity neoId)
        '''
        return asyncio.run_coroutine_threadsafe(
            self.get_genres_by_relation_async(f_genre, c_genre, f_name, reverse), self.loop)

    async def get_relations_by_id_async(self, id, l_label, r_label, direction=''):
        '''
            return : list of triples (relation, entity name, entity neoId)
        '''
        query_tuple = (id, direction, l_label, r_label)
        if query_tuple in self.relation_cache:
            return self.relation_cache[query_tuple]
        else:
            if l_label != '':
                l_label = ':' + l_label
            if r_label != '':
                r_label = ':' + r_label
            arrow1 = ''
            arrow2 = ''
            if direction == '<':
                arrow1 = '<'
            elif direction == '>':
                arrow2 = '>'
            result = await self.execute_async('match (n%s)%s-[r]-%s(m%s) where id(n)=%s return type(r), m.name, id(m)'
                                              % (l_label, arrow1, arrow2, r_label, id))
            res = list(map(lambda x: list(x['row']), result['data']))
            self.relation_cache[query_tuple] = res
            return res

    def get_relations_by_id(self, id, l_label, r_label, direction=''):
        '''
            return : list of triples (relation, entity name, entity neoId)
        '''
        return asyncio.run_coroutine_threadsafe(
            self.get_relations_by_id_async(id, l_label, r_label, direction), self.loop)

    async def get_labels_by_name_async(self, name):
        result = await self.execute_async('match (n) where n.name = \'%s\' return labels(n), id(n)' % (name))
        res = list(map(lambda x: x['row'], result['data']))

        return res

    def get_labels_by_name(self, name):
        return asyncio.run_coroutine_threadsafe(
            self.get_labels_by_name_async(name), self.loop)

    async def get_label_by_id_async(self, id):
        result = await self.execute_async('match (n) where id(n) = %d return labels(n)' % (id))
        res = list(map(lambda x: x['row'][0][0], result['data']))

        return res

    def get_label_by_id(self, id):
        return asyncio.run_coroutine_threadsafe(
            self.get_label_by_id_async(id), self.loop)

    async def get_name_by_id_async(self, id):
        result = await self.execute_async('match (n) where id(n) = %d return n.name' % (id))
        res = list(map(lambda x: x['row'][0], result['data']))

        return res

    def get_name_by_id(self, id):
        return asyncio.run_coroutine_threadsafe(
            self.get_name_by_id_async(id), self.loop)

    async def get_props_in_entity_async(self, name, label):
        result = await self.execute_async('match (n:%s) where n.name = \'%s\' return properties(n)' % (label, name))
        res = list(map(lambda x: x['row'][0], result['data']))[0]
        return res

    def get_props_in_entity(self, name, label):
        res = asyncio.run_coroutine_threadsafe(
            self.get_props_in_entity_async(name, label), self.loop)

        return res

    async def get_props_by_id_async(self, neoId, label):
        result = await self.execute_async('match (n:%s) where id(n) = %d return properties(n)' % (label, neoId))
        res = list(map(lambda x: x['row'][0], result['data']))[0]
        return res

    def get_props_by_id(self, neoId, label):
        res = asyncio.run_coroutine_threadsafe(
            self.get_props_by_id_async(neoId, label), self.loop)

        return res

    # DPQA
    async def get_props_by_id_dpqa_async(self, neoId):
        result = await self.execute_async('match (n) where id(n) = %d return properties(n)' % (neoId))
        res = list(map(lambda x: x['row'][0], result['data']))[0]
        return res

    def get_props_by_id_dpqa(self, neoId):
        res = asyncio.run_coroutine_threadsafe(
            self.get_props_by_id_dpqa_async(neoId), self.loop)

        return res

    # Merchant
    async def get_entities_by_genre_async(self, genre):
        '''
            return : list of entities
        '''

        result = await self.execute_async('match (n:%s) return n,id(n)' % (genre))
        res = self.process_result(result)
        return res

    def get_entities_by_genre(self, genre):
        '''
            return : list of entities
        '''
        return asyncio.run_coroutine_threadsafe(
            self.get_entities_by_genre_async(genre), self.loop)

    async def get_entities_by_genre_and_name_async(self, genre, name):
        '''
            return : list of entities
        '''

        result = await self.execute_async('match (n:%s) where n.name = \'%s\' return n,id(n), properties(n)' % (genre, name))
        res = list(map(lambda x: x['row'], result['data']))
        return res

    def get_entities_by_genre_and_name(self, genre, name):
        '''
            return : list of entities
        '''
        return asyncio.run_coroutine_threadsafe(
            self.get_entities_by_genre_and_name_async(genre, name), self.loop)

    async def get_merchandise_entities_by_genre_async(self, genre):
        '''
            return : list of entities
        '''

        result = await self.execute_async('match (n:%s) where exists(n.服务内容) return properties(n)' % (genre))
        res = list(map(lambda x: x['row'], result['data']))
        return res

    def get_merchandise_entities_by_genre(self, genre):
        '''
            return : list of entities
        '''
        return asyncio.run_coroutine_threadsafe(
            self.get_merchandise_entities_by_genre_async(genre), self.loop)

    @lru_cache(maxsize=256)
    def get_instance_of_genre(self, genre_name, genre='SubGenre'):
        result = self.execute(
            'match (n:Instance)-[:属于]->(m:%s {name:"%s"}) return distinct n,id(n)' % (genre, genre_name)).result()
        result = self.process_result(result)
        return result

    @lru_cache(maxsize=128)
    def get_entities_only_by_name(self, name):
        result = self.execute(
            'match (n {name:"%s"}) return n, id(n)' % name
        ).result()
        result = self.process_result(result)
        return result

    def __del__(self):
        asyncio.run_coroutine_threadsafe(self.session.close(), self.loop)


if __name__ == "__main__":
    pass
