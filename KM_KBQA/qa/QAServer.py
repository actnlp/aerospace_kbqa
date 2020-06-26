import argparse
import json
import logging
import traceback

from aiohttp import web

from .QAFull import QAFull

parser = argparse.ArgumentParser()
# parser.add_argument('-sa', '--server_addr',
#                     required=True, help='remote kb addr')
# parser.add_argument('-sp', '--server_port',
#                     required=True, help='remote kb port')
parser.add_argument('-lp', '--local_port', required=True,
                    help='local port which runs the service for kb')
args = parser.parse_args()
logger = logging.getLogger('qa')
qa = QAFull()
routes = web.RouteTableDef()


def dump(data):
    return json.dumps(data, ensure_ascii=False)


@routes.get('/')
async def answer(request):
    try:
        if 'q' in request.query:
            q = request.query['q']
            res = qa.answer(q)
            res = [r['natural_ans'] for r in res[:3]]
            return web.json_response(res, dumps=dump)
    except Exception as e:
        logger.warning(traceback.format_exc())
    return web.json_response({})


def run_server():
    server = web.Application()
    server.add_routes(routes)
    web.run_app(server, port=args.local_port)


if __name__ == "__main__":
    run_server()
