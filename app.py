"""
程序的入口，可以在这里启动服务
"""
import os

import cv2
from flask import Flask, jsonify, request, json
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

import requests
import numpy as np
import random
import copy

from detect import detect
import config

app = Flask(__name__)
app.config.from_object(config)

db = SQLAlchemy(app)


@app.route('/detect/operating_cabinet', methods=['POST'])
def operating_cabinet():
    data = request.json
    # 从表单中拿到视频链接
    source = data.get('source')
    # 下载视频
    source_file = requests.get(source)
    with open(config.upload_path + '/' + source.split('/')[-1], 'wb') as f:
        f.write(source_file.content)
    # 从表单中拿到设备配置
    device_type_conf = str(data.get('device_type_conf')).replace('\'', '\"')
    print('device_type_conf', device_type_conf)
    res = detect(source=config.upload_path + '/' + source.split('/')[-1], recognize_type='operating-cabinet', operating_device_conf=device_type_conf, nosave=False)
    json_res = {}
    json_res['code'] = 0
    json_res['result'] = res

    return json_res


@app.route('/detect/helmet', methods=['POST'])
def helmet():
    data = request.json
    # 从表单中拿到视频链接
    source = data.get('source')
    # 下载视频
    source_file = requests.get(source)
    with open(config.upload_path + '/' + source.split('/')[-1], 'wb') as f:
        f.write(source_file.content)
    res = detect(source=config.upload_path + '/' + source.split('/')[-1], recognize_type='helmet',
                 nosave=False)

    json_res = {}
    json_res['code'] = 0
    json_res['result'] = res

    return json_res


# allow cross-origin resource sharing
CORS(app, origins='*', supports_credentials=True)


if __name__ == '__main__':
    app.run(port=5001, debug=True)
