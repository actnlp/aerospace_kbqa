import argparse
import json
import logging
import traceback

from aiohttp import web
import aiohttp_cors
from .QAFull import QAFull

parser = argparse.ArgumentParser()
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
            if q.strip():
                res = qa.answer(q)
                res = [r['natural_ans'] for r in res[:3]]
            else:
                res = ["输入问题q参数不能为空"]
        else:
            res = ["输入问题q参数不能为空"]
        return web.json_response(res, dumps=dump)

    except Exception as e:
        logger.warning(msg=traceback.format_exc())
    return web.json_response({})


def run_server():
    server = web.Application()

    cors = aiohttp_cors.setup(server, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    resource = cors.add(server.router.add_resource("/"))
    # cors.add(server.router.add_resource("/"))
    cors.add(resource.add_route("GET", answer))
    # app.router.add_route('GET', '/', getIndex)

    # server.add_routes(routes)
    web.run_app(server, port=args.local_port)

if __name__ == "__main__":
    run_server()
