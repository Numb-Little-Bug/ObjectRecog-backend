'''
该服务的配置文件，定义了服务的端口号，以及其他的配置信息
'''

# 服务端口号
port = 5000

# 使用模型检测的相关配置
helmet_detection_model_path = './weights/helmet_detection.pt'
operating_cabinet_model_path = './weights/operating_cabinet.pt'
source = ''
conf_thres = 0.75

switch_light_strap_labels = ['strap-in', 'strap-out', 'switch-left', 'switch-middle', 'light-green', 'light-off', 'switch-right', 'light-red']

# 文件上传的路径
upload_path = './upload'
