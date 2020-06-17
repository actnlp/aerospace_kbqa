import asyncio
import json
import logging
import os
import pdb
import pickle
import subprocess
import traceback

import aiohttp_cors
from aiohttp import web

from ..common import AsyncNeoDriver, HITBert
from ..common.QCLS import QCLSWrapper
from ..config.config import check_kbqa_ths, data_path, model_path
from ..linking import EntityLinking
from . import SingleRelation
from .predict_bert_multitask import predict_defense

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler('run.log', encoding='utf-8', mode='a')
fh.setFormatter(formatter)
logger.addHandler(fh)


def prepare_server(args):
    server_address = args.server_addr
    server_port = args.server_port
    local_port = args.local_port

    kb_status = subprocess.getoutput('netstat -tnlp | grep ' + str(local_port))
    kb_status = kb_status.strip().split(' ')[-1].split('/')[0]
    if kb_status.isdigit():
        raise Exception('端口 %s 被占用，请换一个端口' % local_port)
    driver = AsyncNeoDriver.AsyncNeoDriver(
        server_address='http://' + server_address + ':' + server_port,
        entity_label='Instance',
        fuzzy_index=None)
    linker = EntityLinking.Linker(driver=driver)
    qa_helper = SingleRelation.QAHelper(driver, linker)
    return driver, linker, qa_helper


def dump(data):
    # cross domain enabled
    return json.dumps(data, ensure_ascii=False)


def update_entity_embeddings(encode, entity_labels, save_path, driver):
    all_entities = []
    for entity_label in entity_labels:
        tmp_entities = driver.get_all_entities(entity_label).result()
        if entity_label == 'Genre':
            tmp_entities = list(filter(
                lambda x: '类' not in x['name'] and '航空公司' not in x['name'] and '行李安检' not in x['name'], tmp_entities))
        all_entities += tmp_entities
    all_entities = [x['name'] for x in all_entities]

    all_ent_emb = {}
    for entity in all_entities:
        emb = encode(entity)
        if not isinstance(emb, list):
            all_ent_emb[entity] = emb.tolist()[0]
        else:
            all_ent_emb[entity] = emb[0]

    with open(save_path, 'wb') as f:
        pickle.dump(all_ent_emb, f)

    return all_ent_emb


def warm_up(qa_helper):
    warm_up_qs = ['机场能寄存衣服吗？', '昆明国际机场有花旗银行的ATM机吗', '麻烦问一下昆明机场的胶囊舱提供吃的吗']
    for q in warm_up_qs:
        _, res = qa_helper.qa(q)
    return


routes = web.RouteTableDef()


def run(args):
    driver, linker, qa_helper = prepare_server(args)

    @asyncio.coroutine
    def answer(request):
        logger.info(request.path_qs)
        try:
            if 'q' in request.query:
                q = request.query['q']
                # 无效输入
                if q is None or q == '':
                    ret = {
                        'qa': {'answers': [{'answer': '请输入问题！', 'nodes': [], 'edges': []}]}}
                    logger.info(ret)
                    return web.json_response(ret, dumps=dump)

                # 拒识模块
                model = QCLSWrapper.from_pretrained(
                    os.path.join(model_path, 'check_kbqa_model.pt'))
                kbqa_prob = model.eval([q])
                if kbqa_prob[0] < check_kbqa_ths or len(q) > 25:
                    ret = {
                        'qa': {'answers': [{'answer': '非KBQA问题！', 'nodes': [], 'edges': []}]}}
                    logger.info(ret)
                    return web.json_response(ret, dumps=dump)

                # kbqa
                q = q.lower()
                logger.info(q)
                # type_res, res = qa_helper.qa(q)
                type_res, res = predict_defense(q.strip(), driver)
                logger.info(res)
                res = qa_helper.decorate(res, type_res['type'], 3)
                res = {'qa': res}
                return web.json_response(res, dumps=dump)
            return web.json_response({})

        except Exception as e:
            logger.error(traceback.format_exc())

    # 载入实体字典
    all_ent_emb = update_entity_embeddings(HITBert.encode, ['Instance', 'SubGenre', 'Genre'],
                                           os.path.join(data_path, 'entity_embeddings.pkl'), driver)
    qa_helper.load_dictionary(all_ent_emb)

    # 预热系统
    warm_up(qa_helper)

    # 开启问答服务
    try:
        server = web.Application()
        cors = aiohttp_cors.setup(server, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
        resource = cors.add(server.router.add_resource('/qa/'))
        cors.add(resource.add_route("GET", answer))
        # server.add_routes(routes)
        web.run_app(server, port=args.local_port)
    finally:
        asyncio.run_coroutine_threadsafe(driver.session.close(), driver.loop)
