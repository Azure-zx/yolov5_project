import sys, os
from untitled import  Ui_MainWindow
from PyQt5 import QtCore,QtGui,uic,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel,QFileDialog
import argparse
import cv2
from detect import run
import subprocess
import argparse
import os
import platform
import sys
from pathlib import Path
import requests
import torch
import time
import threading

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        # 初始化各控件
        super(MyApp,self).__init__(parent)
        self.setupUi(self)
        self.count = 0
        self.path = '0'
        self.running = True
        self.timer_video = QtCore.QTimer()
        self.pushButton.clicked.connect(self.open_camera)
        self.pushButton_2.clicked.connect(self.open_image)
        self.pushButton_3.clicked.connect(self.open_vedio)
        bg_image = QPixmap()
        bg_image.load('background.png')

        # 创建画刷
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(bg_image))

        # 应用画刷到窗口
        self.setPalette(palette)
        #
    # 初始化参数
    def run(self,
            weights=ROOT / 'best.pt',  # model path or triton URL
            source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.55,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            if self.running:
                for i, det in enumerate(pred):
                    # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                target_cls = 0  # 获取标签为0的目标框(down)
                                target_boxes = det[det[:, -1] == target_cls]

                                # 获取目标框置信度列表
                                confidences = target_boxes[:, 4]
                                # 如果没有摔倒人员 将count标记清0
                                if confidences.nelement() == 0:
                                    self.count = 0

                                if names[c] == 'down' and self.count == 0:
                                    self.count = self.count + 1
                                if self.count >= 1 and names[c] == 'down':
                                    target_cls = 0  # 获取标签为0的目标框(down)
                                    target_boxes = det[det[:, -1] == target_cls]

                                    # 获取目标框置信度列表
                                    confidences = target_boxes[:, 4]
                                    confidences_list = confidences.tolist()
                                    value = confidences_list[0]
                                    # 判断置信度是否超过期望阈值
                                    if value >= 0.85 and self.count <= 2:
                                        self.warning()
                                        self.count = self.count + 1
                                        mark = time.localtime()
                                        d = str(mark.tm_year) + '.' + str(mark.tm_mon) + '.' + str(
                                            mark.tm_mday) + ' ' + str(
                                            mark.tm_hour) + '：' + str(mark.tm_min) + '：' + str(mark.tm_sec)
                                        cv2.imwrite('D:/work/yolov5-master/data_saved/' + d + '.jpg', im0)

                    # Stream results
                    im0 = annotator.result()
                    height, width, channel = im0.shape
                    bytes_per_line = 3 * width
                    q = QImage(im0.data, width, height, bytes_per_line, QImage.Format_BGR888)
                    p = QPixmap.fromImage(q)
                    self.label.setScaledContents(True)
                    # print(type(p))
                    self.label.repaint()
                    # 将 QImage 设置为 QLabel 的背景
                    self.label.setPixmap(p.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))
        return 0




    def run_2(self,
            weights=ROOT / 'best.pt',  # model path or triton URL
            source=ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
            data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            vid_stride=1,  # video frame-rate stride
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):
                # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                            target_cls = 0  # 获取标签为0的目标框(down)
                            target_boxes = det[det[:, -1] == target_cls]

                            # 获取目标框置信度列表
                            confidences = target_boxes[:, 4]
                            # 如果没有摔倒人员 将count标记清0
                            if confidences.nelement() == 0:
                                self.count = 0

                            if names[c] == 'down' and self.count == 0:
                                self.count = self.count + 1
                            if self.count >= 1 and names[c] == 'down':
                                target_cls = 0  # 获取标签为0的目标框(down)
                                target_boxes = det[det[:, -1] == target_cls]

                                # 获取目标框置信度列表
                                confidences = target_boxes[:, 4]
                                confidences_list = confidences.tolist()
                                value = confidences_list[0]
                                # 判断置信度是否超过期望阈值
                                if value >= 0.85 and self.count <= 2:
                                    self.warning()
                                    self.count = self.count+1
                                    mark = time.localtime()
                                    d = str(mark.tm_year) + '.' + str(mark.tm_mon) + '.' + str(
                                        mark.tm_mday) + ' ' + str(
                                        mark.tm_hour) + '：' + str(mark.tm_min) + '：' + str(mark.tm_sec)
                                    cv2.imwrite('D:/work/yolov5-master/data_saved/' + d + '.jpg', im0)

                # Stream results
                im0 = annotator.result()
                height, width, channel = im0.shape
                bytes_per_line = 3 * width
                q = QImage(im0.data, width, height, bytes_per_line, QImage.Format_BGR888)
                p = QPixmap.fromImage(q)
                self.label.setScaledContents(True)
                # print(type(p))
                self.label.repaint()
                # 将 QImage 设置为 QLabel 的背景
                self.label.setPixmap(p.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))

    def warning(self):
        url = ' http://miaotixing.com/trigger?id=tej14SS'
        requests.get(url)

    def open_camera(self):
        self.running = True
        self.run(source='0')

    def open_image(self):
        self.running = False
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "data", "所有文件 (*);;文本文件 (*.txt)")
        # 如果用户选择了文件，则在控制台中输出文件路径
        if file_path:
            print("选择的文件是:", file_path)
            self.path = file_path
            self.run_2(source=self.path)

        if not file_path:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    def open_vedio(self):
        self.running = False
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "data", "所有文件 (*);;文本文件 (*.txt)")
        # 如果用户选择了文件，则在控制台中输出文件路径
        if file_path:
            print("选择的文件是:", file_path)
            self.path = file_path
            self.run_2(source=self.path)

        if not file_path:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)

    def warning(self):
        url = ' http://miaotixing.com/trigger?id=tej14SS'
        requests.get(url)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    # 创建一个进程 click_button时启动该线程然后进行一个window.run
    my_thread = threading.Thread(target=window.warning)
    # window.pushButton.clicked.connect(my_thread.start)

    # my_thread2 = threading.Thread(target=window.open_image)
    # window.pushButton_2.clicked.connect(my_thread2.start)
    sys.exit(app.exec_())
