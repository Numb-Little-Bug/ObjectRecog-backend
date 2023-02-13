'''
定义所有的接口
'''

from flask import Blueprint
from flask import request

import config
from detect import detect

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/')
def hello_world():
    return 'Hello World!'


@api_bp.route('/switch_light_strap', methods=['POST'])
def switch_light_strap():
    form = request.form
    # 从表单中拿到图片
    source = request.files.get('source')
    # 保存图片
    source.save(config.upload_path + '/' + source.filename)
    recognize_type = form.get('recognize_type')
    device_type = form.get('device_type')
    if device_type == '1':
        operating_device_conf = config.device1_type_config_path
    elif device_type == '2':
        operating_device_conf = config.device2_type_config_path
    else:
        operating_device_conf = config.device3_type_config_path

    detect(source=config.upload_path + '/' + source.filename, recognize_type=recognize_type, operating_device_conf=operating_device_conf)

    return 'success'
