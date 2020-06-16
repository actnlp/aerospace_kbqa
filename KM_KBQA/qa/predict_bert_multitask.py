# coding: utf-8
import argparse
import asyncio
import csv
import json
import logging
import pdb
import subprocess

import jieba
from aiohttp import web

from ..BertEntityRelationClassification import args
from ..BertEntityRelationClassification.multitask_main_k_fold import \
    multitask_predict
from ..BertEntityRelationClassification.predict import merchandise_predict
from ..common.link_kb import prepare_server
from . import Limiter

logger = logging.getLogger('commercial')
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler('run.log', 'a', encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

parser = argparse.ArgumentParser()
parser.add_argument('-sa', '--server_addr',
                    required=True, help='remote kb addr')
parser.add_argument('-sp', '--server_port',
                    required=True, help='remote kb port')
parser.add_argument('-lp', '--local_port', required=True,
                    help='local port which runs the service for kb')
parser.add_argument('-stc', '--sentence', default='机场T1航站楼有哪些可以吃东西的地方？',
                    help='question example to be asked')
# parser.add_argument('-ap', '--all_predict', required=True)
p_args = parser.parse_args()
driver = prepare_server(p_args)

routes = web.RouteTableDef()


def dump(data):
    return json.dumps(data, ensure_ascii=False)


def match_entities_content(sent, entities):
    '''
    Args:
        sent :str
        entities : list of dict
    Return:
        matched : list of dict
    '''
    sent_cut = set(jieba.lcut(sent))
    matched = []
    for ent in entities:
        if '服务内容' in ent:
            # do match
            content = set(ent['服务内容'].split(';'))
            if len(sent_cut & content) > 0:
                matched.append(ent)
    return matched


def predictV2(sentence):
    '''
    商业实体服务内容匹配流程1：
        1. 找出top1实体类别, top1关系, top3实体类别 （multitask_predict
        2. 获取top3类别下属具体实体
        3. 将句子和具体实体服务内容匹配
            a. 如果有匹配
                返回匹配实体
            b. 如果无匹配
                返回top1类别实体
    '''
    # 1. top1 top3 predict
    m_ent_top1, m_rel_top1, m_ent_top3 = multitask_predict(sentence.strip())
    mer_ent_top3 = merchandise_predict(sentence, driver)
    if len(mer_ent_top3) > 0:
        m_ent_top3 = mer_ent_top3
    # 2. 获取下属实体
    genre2entites = {}
    cand_entities = []
    for genre in m_ent_top3:
        sub_entities = driver.get_instance_of_genre(genre)
        genre2entites[genre] = sub_entities
        cand_entities.extend(sub_entities)
    # 3. 匹配内容
    matched_entities = match_entities_content(sentence, cand_entities)

    if len(matched_entities) > 0:
        return {
            'type_top1': m_ent_top3[0],
            'rel_top1': m_rel_top1,
            'type_top3': m_ent_top3,
            'entities': list(map(lambda x: x['name'], matched_entities)),
        }
    else:
        return {
            'type_top1': m_ent_top1,
            'rel_top1': m_rel_top1,
            'type_top3': m_ent_top3,
            'entities': list(map(lambda x: x['name'], genre2entites[m_ent_top1])),
        }


def predictV3(sentence):
    '''
    商业实体服务内容匹配流程1：
        1. 找出top1实体类别, top1关系, top3实体类别 （multitask_predict
        2. 获取top3类别下属具体实体
        3. 将句子和具体实体服务内容匹配
            a. 如果有匹配
                返回匹配实体
            b. 如果无匹配
                返回top1类别实体
        4. 找出句子中的各限制条件并返回
    '''
    # 1. top1 top3 predict
    m_ent_top1, m_rel_top1, m_ent_top3 = multitask_predict(sentence.strip())
    mer_ent_top3 = merchandise_predict(sentence, driver)
    if len(mer_ent_top3) > 0:
        m_ent_top3 = mer_ent_top3
    # 2. 获取下属实体
    genre2entites = {}
    cand_entities = []
    for genre in m_ent_top3:
        sub_entities = driver.get_instance_of_genre(genre)
        genre2entites[genre] = sub_entities
        cand_entities.extend(sub_entities)
    # 3. 匹配内容
    matched_entities = match_entities_content(sentence, cand_entities)
    # compute restrictions
    limiter = Limiter(sentence)
    restrictions = limiter.check()
    logger.info(restrictions)

    if len(matched_entities) > 0:
        return {
            'type_top1': m_ent_top3[0],
            'rel_top1': m_rel_top1,
            'type_top3': m_ent_top3,
            'entities': list(map(lambda x: x['name'], matched_entities)),
            'time': restrictions['时间'],
            'location': restrictions['地点'],
            'price': restrictions['价格'],
            'bank': restrictions['银行'],
            'currency': restrictions['币种'],
            'airline': restrictions['航空公司'],
            'region': restrictions['地区'],
            'company': restrictions['公司']
        }
    else:
        return {
            'type_top1': m_ent_top1,
            'rel_top1': m_rel_top1,
            'type_top3': m_ent_top3,
            'entities': list(map(lambda x: x['name'], genre2entites[m_ent_top1])),
            'time': restrictions['时间'],
            'location': restrictions['地点'],
            'price': restrictions['价格'],
            'bank': restrictions['银行'],
            'currency': restrictions['币种'],
            'airline': restrictions['航空公司'],
            'region': restrictions['地区'],
            'company': restrictions['公司']
        }


def predict(sentence):
    mer_ent_top3 = merchandise_predict(sentence, driver)
    m_ent_top1, m_rel_top1, m_top3_ent = multitask_predict(sentence.strip())
    if len(mer_ent_top3) > 0:
        return {
            'entity_top1': mer_ent_top3[0],
            'rel_top1': m_rel_top1,
            'entity_top3': mer_ent_top3
        }
    else:
        return {
            'entity_top1': m_ent_top1,
            'rel_top1': m_rel_top1,
            'entity_top3': m_top3_ent
        }


@routes.get('/commercial/')
async def commercial_predict(request):
    logger.info(request.path_qs)
    try:
        if 'q' in request.query:
            q = request.query['q']
            logger.info(q)
            res = predict(q)
            logger.info(res)
            return web.json_response(res, dumps=dump)
    except Exception as e:
        logger.error(e)
    return web.json_response({})


@routes.get('/commercialV2/')
async def commercial_predict(request):
    logger.info(request.path_qs)
    try:
        if 'q' in request.query:
            q = request.query['q']
            logger.info(q)
            res = predictV2(q)
            logger.info(res)
            return web.json_response(res, dumps=dump)
    except Exception as e:
        logger.error(e)
    return web.json_response({})


@routes.get('/commercialV3/')
async def commercial_predict(request):
    logger.info(request.path_qs)
    try:
        if 'q' in request.query:
            q = request.query['q']
            logger.info(q)
            res = predictV3(q)
            logger.info(res)
            return web.json_response(res, dumps=dump)
    except Exception as e:
        logger.error(e)
    return web.json_response({})


def run_server():
    server = web.Application()
    server.add_routes(routes)
    web.run_app(server, port=p_args.local_port)


def run_test():
    while True:
        sent = input()
        if sent == 'q':
            return
        else:
            print(predictV2(sent))


def jaccard_rank(entities):
    def jaccard(a, b):
        a = set(a)
        b = set(b)
        return len(a & b) / len(a | b)
    pass


def predict_defense(sentence, graph_driver):
    mer_ent_top3 = merchandise_predict(sentence, driver)
    m_ent_top1, m_rel_top1, m_top3_ent = multitask_predict(sentence.strip())
    logger.info('mer_ent_top3'+str(mer_ent_top3))
    logger.info('mtop3'+str(m_top3_ent))
    # calculate limits
    limiter = Limiter(sentence.strip())
    limits = limiter.check()
    # fetch entity and relation
    for m in mer_ent_top3:
        if m in m_top3_ent:
            m_top3_ent.remove(m)

    entities = mer_ent_top3[:2] + m_top3_ent
    return fetch_answer_from_graph(entities[:3], m_rel_top1, graph_driver, limits)


def fetch_answer_from_graph(ent_top3, rel_top1, driver, limits):
    '''
    fetch answers from knowledge graph
    :param ent_top3: the top3 entities returned by multitask-learning model [list]
    :param rel_top3: the top3 rels returned by the multitask-learning modle [list]
    :return: list [ans_str, cand, ent, {'prop':'prop_item'}, {'restr':'restr_item'}, ent_val, rel_val]
    '''
    if len(ent_top3) == 0:
        return {'type': 'not recognized', 'genre': ''}, [
            ['系统没有检测到对应的实体或属性，请重新输入！', None, None, None, None, None, None, None]]
    is_ok = [True, True, True]
    all_results = []
    res_type = ''
    for i, ent in enumerate(ent_top3):
        # first try SubGenre
        fetch_res = driver.get_entities_by_genre_and_name(
            'SubGenre', ent).result()
        if len(fetch_res) == 0:
            continue
        type_name = fetch_res[0][0]['name']
        if len(fetch_res) == 0:
            continue
        instances = driver.get_genres_by_relation(
            'SubGenre', 'Instance', type_name, reverse=True).result()
        # check the prop of each instance
        satisfy_restr = [1 for instance in instances]
        props, restrs = [], []
        for j, instance in enumerate(instances):
            prop = ''
            cur_prop, cur_restr = {}, {}
            if len(rel_top1) > 0:
                if rel_top1 not in instance.keys():
                    if rel_top1 in args.REVERT_REL_CONVERT_DICT:
                        for suspect_prop in args.REVERT_REL_CONVERT_DICT[rel_top1]:
                            if suspect_prop in instance.keys():
                                prop = suspect_prop
                                break
                else:
                    prop = rel_top1
            if prop == '':
                is_ok[i] = False
            else:
                cur_prop[prop] = instance[prop]

            # check restr
            for limit in limits:
                if len(limits[limit]) > 0:
                    if limit in ['时间', '地点', '币种', '价格']:
                        for restr_prop in instance:
                            if limit in restr_prop:
                                prop_limiter = Limiter(instance[restr_prop])
                                restrs = prop_limiter.check()
                                for item in limits[limit]:
                                    if item not in restrs[limit]:
                                        satisfy_restr[j] /= 2
                                        cur_restr[restr_prop] = item
                                    else:
                                        cur_restr[restr_prop] = item

                    elif limit in ['银行', '航空公司', '公司']:
                        for item in limits[limit]:
                            if item not in instance['name']:
                                if limit == '航空公司' and '航司代码' in instance and item in instance['航司代码']:
                                    cur_restr['航司代码'] = item
                                else:
                                    cur_restr[limit] = item
                                    satisfy_restr[j] /= 2
                            else:
                                cur_restr['name'] = item

            # merge answers :return: list [ans_str, cand, ent, {'prop':'prop_item'}, {'restr':'restr_item'}, ent_val, rel_val]
            if len(cur_prop) == 0 and res_type == '':
                res_type = 'list'
            elif len(cur_prop) > 0 and res_type == '':
                res_type = 'exact'
            if res_type == 'exact' and len(cur_prop) > 0:
                cur_ans = '%s的%s是%s' % (instance['name'], list(cur_prop.keys())[
                                        0], cur_prop[list(cur_prop.keys())[0]])
            else:
                cur_ans = '您好，机场内有%s' % instance['name']

            cur_result = [cur_ans, int(instance['neoId']), instance['name'], instance['name'],
                          cur_prop, cur_restr, satisfy_restr[j], satisfy_restr[j]]
            all_results.append(cur_result)
    all_results = list(sorted(all_results, key=lambda x: x[6], reverse=True))
    if len(all_results) == 0:
        return {'type': 'not recognized', 'genre': ''}, [
            ['系统没有检测到对应的实体或属性，请重新输入！', None, None, None, None, None, None, None]]
    return {'type': res_type, 'genre': ''}, all_results
    # if len(fetch_res) == 0:
    #     fetch_res = driver.get_entities_by_genre_and_name('Genre', ent).result()


if __name__ == '__main__':
    run_server()
    # run_test()
