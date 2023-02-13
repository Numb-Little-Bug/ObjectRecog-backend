'''
该服务的配置文件，定义了服务的端口号，以及其他的配置信息
'''

# 服务端口号
port = 5000

# 使用模型检测的相关配置
switch_light_strap_model_path = './weights/switch-light-strap-300epoch-default_lr.pt'
helmet_detection_model_path = './weights/helmet_detection.pt'
device1_type_config_path = './cfg/device/device_1.json'
device2_type_config_path = './cfg/device/device_2.json'
device3_type_config_path = './cfg/device/device_3.json'
source = ''
conf_thres = 0.65

# 文件上传的路径
upload_path = './upload'
