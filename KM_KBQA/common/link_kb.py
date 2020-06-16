# coding: utf-8
import asyncio
import json
import logging
import os
import pdb
import pickle
import subprocess

import aiohttp_cors
from aiohttp import web

from .async_neo_driver import AsyncNeoDriver


def prepare_server(args):
    server_address = args.server_addr
    server_port = args.server_port
    local_port = args.local_port

    kb_status = subprocess.getoutput('netstat -tnlp | grep ' + str(local_port))
    kb_status = kb_status.strip().split(' ')[-1].split('/')[0]
    if kb_status.isdigit():
        raise Exception('端口 %s 被占用，请换一个端口' % local_port)
    driver = AsyncNeoDriver(
        server_address='http://' + server_address + ':' + server_port,
        entity_label='Instance',
        fuzzy_index=None)

    return driver
