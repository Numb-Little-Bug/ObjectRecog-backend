'''
定义一些工具函数
'''
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import json
from sklearn.cluster import KMeans

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import config


def detect(save_img=False, recognize_type=None, source=None, device='cpu', conf_thres=None, operating_device_conf=None, nosave=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./checkout/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--operating-device-conf', type=str, default=None,
                        help='path of config file for operating device conf')
    opt = parser.parse_args()
    if recognize_type == 'operating-cabinet':
        opt.weights = config.operating_cabinet_model_path
    elif recognize_type == 'helmet':
        opt.weights = config.helmet_detection_model_path
    else:
        raise ValueError('recognize_type must be operating-cabinet or helmet')
    opt.source = source
    opt.device = device
    opt.nosave = nosave
    if conf_thres is not None:
        opt.conf_thres = conf_thres
    else:
        opt.conf_thres = config.conf_thres
    opt.operating_device_conf = operating_device_conf

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Operating device configuration
    if opt.operating_device_conf is not None:
        operating_device_conf_dict = json.loads(opt.operating_device_conf)
    else:
        operating_device_conf_dict = None

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    detect_result = {}
    s_s = ''
    num_operations = -1
    detect_result_list = []

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                lights_x = []
                lights_y = []
                straps_x = []
                straps_y = []
                switches_x = []
                switches_y = []
                num_nohelmet = 0
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    num_nohelmet += int(cls.item())
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    if save_txt:  # Write to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    if operating_device_conf_dict is not None:
                        cls_int = int(cls.item())
                        if cls_int == 4 or cls_int == 5 or cls_int == 7:
                            lights_y.append(xywh[1])  # y coordinate of the top of the bounding box
                            lights_x.append((xywh[0], cls_int))  # x coordinate of the top of the bounding box
                        elif cls_int == 2 or cls_int == 3 or cls_int == 6:
                            switches_y.append(xywh[1])  # y coordinate of the top of the bounding box
                            switches_x.append((xywh[0], cls_int))  # x coordinate of the left of the bounding box
                        elif cls_int == 0 or cls_int == 1:
                            straps_y.append(xywh[1])  # y coordinate of the top of the bounding box
                            straps_x.append((xywh[0], cls_int))  # x coordinate of the left of the bounding box

                print('lights_x: ', lights_x)
                print('lights_y: ', lights_y)
                print('switches_x: ', switches_x)
                print('switches_y: ', switches_y)
                print('straps_x: ', straps_x)
                print('straps_y: ', straps_y)
                print('operating_device_conf_dict: ', operating_device_conf_dict)
                if operating_device_conf_dict is not None:
                    try:
                        # lights
                        # k-means clustering to find the lights with the y coordinate
                        num_clusters = len(operating_device_conf_dict.get('lights'))
                        lights_y_reshaped = np.array(lights_y).reshape(-1, 1)
                        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(lights_y_reshaped)
                        labels = kmeans.labels_
                        lights = [[] for i in range(num_clusters)]
                        for i in range(len(labels)):
                            lights[labels[i]].append((lights_x[i], lights_y[i]))
                        for i in range(len(lights)):
                            lights[i].sort(key=lambda x: x[0][0])
                        lights.sort(key=lambda x: x[0][1])
                        lights_str = ""
                        lights_lst = []
                        for i in range(len(operating_device_conf_dict.get('lights'))):
                            for j in range(len(lights[i])):
                                lights_lst.append({operating_device_conf_dict.get('lights')[i][j].get(
                                    'name'): config.switch_light_strap_labels[lights[i][j][0][1]]})
                                lights_str += operating_device_conf_dict.get('lights')[i][j].get(
                                    'name') + ": " + config.switch_light_strap_labels[lights[i][j][0][1]] + "\n"
                                # lights_str += operating_device_conf_dict.get('lights').get('line_' + str(i + 1)).get(
                                #     'light_' + str(j + 1)) + ": " + str(lights[i][j][0][1]) + "\n"

                        # switches
                        # k-means clustering to find the switches with the y coordinate
                        num_clusters = len(operating_device_conf_dict.get('switches'))
                        switches_y_reshaped = np.array(switches_y).reshape(-1, 1)
                        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(switches_y_reshaped)
                        labels = kmeans.labels_
                        switches = [[] for i in range(num_clusters)]
                        for i in range(len(labels)):
                            switches[labels[i]].append((switches_x[i], switches_y[i]))
                        for i in range(len(switches)):
                            switches[i].sort(key=lambda x: x[0][0])
                        switches.sort(key=lambda x: x[0][1])
                        switches_str = ""
                        switches_lst = []
                        for i in range(len(operating_device_conf_dict.get('switches'))):
                            for j in range(len(switches[i])):
                                switches_lst.append({operating_device_conf_dict.get('switches')[i][j].get(
                                    'name'): config.switch_light_strap_labels[switches[i][j][0][1]]})
                                switches_str += operating_device_conf_dict.get('switches')[i][j].get(
                                    'name') + ": " + config.switch_light_strap_labels[
                                    switches[i][j][0][1]] + "\n"
                                # switches_str += operating_device_conf_dict.get('switches').get(
                                #     'line_' + str(i + 1)).get('switch_' + str(j + 1)) + ": " + str(
                                #     switches[i][j][0][1]) + "\n"

                        # straps
                        # k-means clustering to find the strap with the y coordinate
                        num_clusters = len(operating_device_conf_dict.get('straps'))
                        straps_y_reshaped = np.array(straps_y).reshape(-1, 1)
                        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(straps_y_reshaped)
                        labels = kmeans.labels_
                        straps = [[] for i in range(num_clusters)]
                        for i in range(len(labels)):
                            straps[labels[i]].append((straps_x[i], straps_y[i]))
                        for i in range(len(straps)):
                            straps[i].sort(key=lambda x: x[0][0])
                        straps.sort(key=lambda x: x[0][1])
                        straps_str = ""
                        straps_lst = []
                        for i in range(len(operating_device_conf_dict.get('straps'))):
                            for j in range(len(straps[i])):
                                straps_lst.append({operating_device_conf_dict.get('straps')[i][j].get('name'): config.switch_light_strap_labels[
                                                   straps[i][j][0][1]]})
                                # straps_str += operating_device_conf_dict.get('straps').get('line_' + str(i + 1)).get(
                                #     'strap_' + str(j + 1)) + ": " + str(straps[i][j][0][1]) + "\n"
                                straps_str += operating_device_conf_dict.get('straps')[i][j].get('name') + ": " + config.switch_light_strap_labels[
                                    straps[i][j][0][1]] + "\n"

                        detect_result = {"lights": lights_lst, "switches": switches_lst, "straps": straps_lst}
                        if s_s != s:
                            detect_result_list.append(detect_result)
                            num_operations += 1
                            s_s = s
                            print('enter')
                        print('s_s', s_s)
                        print("------------------------------ DETECT RESULT ------------------------------")
                        print(detect_result)
                        print("-------------------------------------------------------------------")
                        print("---------------------------------\n")
                        print(lights_str)
                        print("---------------------------------\n")
                        print(switches_str)
                        print("---------------------------------\n")
                        print(straps_str)
                        print("---------------------------------")

                        if save_txt:  # Write to file again
                            with open(txt_path + '.txt', 'a') as f:
                                f.write('\n')
                                f.write(lights_str)
                                f.write(switches_str)
                                f.write(straps_str)
                    except Exception as e:
                        print("------------------------------ ERROR ------------------------------")
                        print(
                            "Perhaps something went wrong with the operating device detection. \n\nPlease check the operating device configuration file.")
                        print("-------------------------------------------------------------------")

                if recognize_type == 'helmet':
                    detect_result = {"未佩戴安全帽人数": num_nohelmet}

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            print('num_operations: ', num_operations)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    if recognize_type == 'helmet':
        return detect_result
    else:
        return detect_result_list
