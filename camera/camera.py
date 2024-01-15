# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/10/22
# @File:          camera.py
# @Software:      PyCharm
# @Project:       face_contrast

import cv2
import dlib
import numpy as np
import imutils

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from imutils import face_utils
from keras.models import load_model

from config import config
from scipy.spatial import distance
from algorithms.face import deal_frame
from algorithms.image_feature import face_feature_list
from algorithms.coordinate2points import coordinate2points
from algorithms.face import get_face
from algorithms.face import cate_frame
from algorithms.face_contrast import face_distance


class Camera(object):
    def __init__(self, camera_id, camera_ip, faces_folder_path, current_speed=50, fps=30, eye_ar_thresh=0.25,
                 eye_ar_consec_frames=3, mar_thresh=0.8, mouth_arconsec_frames=2, har_thresh=0.3,
                 nod_ar_consec_frames=2, par_thresh=25, par_duration=3, time_stamp=600, frame_list=None,
                 face_list=None, camera_log=None, blink_threshold=5, yawn_threshold=5, nod_threshold=2, par_threshold=1,
                 speed_threshold=30, face_threashold=0.08, time_interval=0, frame_threshold=5):

        """
        封装摄像机类
        :param camera_id: 摄像机编号
        :param camera_ip: 摄像机ip
        :param faces_folder_path: 本地人脸数据库地址
        :param current_speed: 车速阈值
        :param fps: 摄像机fps
        :param eye_ar_thresh: 眼睛长宽比
        :param eye_ar_consec_frames: 眼睛闪烁阈值(连续多少帧表示一次眨眼)
        :param mar_thresh: 打哈欠嘴巴长宽比
        :param mouth_arconsec_frames: 连续哈欠帧数
        :param har_thresh: 瞌睡点头
        :param nod_ar_consec_frames: 瞌睡点头阈值(连续多少帧标识一次瞌睡点头)
        :param par_thresh: 目不正视前方阈值
        :param par_duration: 持续时间
        :param time_stamp: 时间戳(规定时间之内如果没有达到报警阈值则当前计数器清零)
        :param frame_list: 存放视频帧list
        :param face_list: 存放单个人脸list
        :param camera_log: 摄像机日志对象
        :param blink_threshold: 判断眨眼频率报警阈值
        :param yawn_threshold: 判断打哈欠报警阈值
        :param nod_threshold: 判断点头报警阈值
        :param par_threshold: 判断偏头/目不正视前方报警阈值
        :param speed_threshold: 速度阈值(当车速大于某个设定的速度阈值时程序有效)
        :param face_threashold: 满足确定为同一张人脸的阈值
        :param time_interval:
        :param frame_threshold:
        """

        self.camera_id = camera_id
        self.camera_ip = camera_ip
        self.faces_folder_path = faces_folder_path
        self.camera_log = camera_log
        self.current_speed = current_speed
        self.fps = fps
        self.eye_ar_thresh = eye_ar_thresh
        self.eye_ar_consec_frames = eye_ar_consec_frames
        self.mar_thresh = mar_thresh
        self.mouth_arconsec_frames = mouth_arconsec_frames
        self.har_thresh = har_thresh
        self.nod_ar_consec_frames = nod_ar_consec_frames
        self.par_thresh = par_thresh
        self.par_duration = par_duration
        self.par_ar_consec_frames = self.fps * self.par_duration
        self.time_stamp = time_stamp
        self.blink_threshold = blink_threshold
        self.yawn_threshold = yawn_threshold
        self.nod_threshold = nod_threshold
        self.par_threshold = par_threshold
        self.speed_threshold = speed_threshold
        self.face_threashold = face_threashold
        self.time_interval = time_interval
        self.frame_threshold = frame_threshold

        # 定义提前训练好的人脸分类器，用于提取人脸的128个特诊点
        self.facerec = dlib.face_recognition_model_v1(config.face_recognition_resnet_model_path)

        # dlib.shape_predictor 获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor(config.landmarkd_model_path)

        # 使用detector(gray, 0) 进行脸部位置检测
        self.detector = dlib.get_frontal_face_detector()

        self.emotion_classifier = load_model(config.new_emotion_classifier_model_path, compile=False)

        # 7种表情列表
        self.emotions = ['生气', '厌恶', '害怕', '高兴', '悲伤', '惊讶', '正常']

        self.model_bin = config.model_bin
        self.config_text = config.config_text

        # load caffe model
        self.net = cv2.dnn.readNetFromCaffe(self.config_text, self.model_bin)
        # set back-end
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 加载字体
        self.font = config.font

        self.face_name_list = face_feature_list(faces_folder_path=self.faces_folder_path,
                                           detector=self.detector, predictor=self.predictor,
                                           facerec=self.facerec)

        # 分别获得左右眼面部标志的索引
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

        if frame_list is None:
            self.frame_list = []

        if face_list is None:
            self.face_list = []

        self.data = {
            'cameraId': self.camera_id,
            'order': 'Alarm',
            'videoPath': None,
            'secretToken': None,
            'frame': None,  # 当前帧视频
            'characteristic_value': None,  # 人脸特征值
            'face_coordinates': None  # 人脸坐标
        }

    def get_frame(self, frame_queue):
        cap = cv2.VideoCapture(self.camera_ip)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_queue.put(frame)

    def draw_name(self, frame_queue):
        """
        在数据库中存在的人脸框上添加中文名
        :return:
        """
        # cap = cv2.VideoCapture(0)
        while True:
            frame = frame_queue.get()
            if frame is not None:
                # cap = cv2.VideoCapture(self.camera_ip)
                # ret, frame = cap.read()
                frame = imutils.resize(frame, width=720)
                # if not ret:
                #     break

                # 人脸检测
                h, w = frame.shape[:2]
                # 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等，返回一个4通道的blob(blob可以简单理解为一个N维的数组，用于神经网络的输入)
                blobImage = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
                self.net.setInput(blobImage)
                cvout = self.net.forward()

                data_list = deal_frame(frame=frame, cvout=cvout, h=h, w=w, face_name_list=self.face_name_list,
                                       predictor=self.predictor, facerec=self.facerec)

                imagePIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                font = ImageFont.truetype(r'../model/NotoSansCJK-Black.otf', 20)
                draw = ImageDraw.Draw(imagePIL)
                for i in range(0, len(data_list)):
                    x, y, w, h = data_list[i]['coordinate']
                    name = data_list[i]['name']
                    text = '姓名： ' + name
                    draw.text((x, int(y + h) / 2), text=text, font=font, fill=(255, 0, 0))

                imgopencv = cv2.cvtColor(np.asarray(imagePIL), cv2.COLOR_RGB2BGR)

                cv2.imshow('', imgopencv)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def draw_emotion(self, frame_queue):
        """
        显示中文情绪
        :return:
        """
        # cap= cv2.VideoCapture(0)
        while True:
            frame = frame_queue.get()

            # cap = cv2.VideoCapture(self.camera_ip)
            # ret, frame = cap.read()
            # if not ret:
            #     break
            if frame is not None:
                frame = imutils.resize(frame, width=720)
                h, w = frame.shape[: 2]
                # 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道，返回一个4通道的blob(blob 可以简单的理解为一个N为数组，
                # 用于神经网络的输入)
                blob_img = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
                self.net.setInput(blob_img)
                cvout = self.net.forward()
                face_list, face_number = get_face(frame, cvout, h, w, emotions=self.emotions,
                                                  emotion_classifier=self.emotion_classifier,
                                                  predictor=self.predictor, facerec=self.facerec)

                image_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                font = ImageFont.truetype(self.font, 20)
                draw = ImageDraw.Draw(image_PIL)

                for face in face_list:
                    x, y, w, h = face['face']
                    emotion = face['emotion']
                    faces = 'faces: ' + str(face_number)
                    draw.text((x, int(y + h) / 2), text=emotion, font=font, fill=(255, 0, 0))
                    draw.text((10, 10), text=faces, font=font, fill=(255, 0, 0))

                img2opencv = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)

                cv2.imshow('emotion', img2opencv)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def show_attitude(self, frame_queue):
        """
        头部姿态识别
        :param frame_queue:
        :return:
        """
        # cap = cv2.VideoCapture(0)
        while True:
            frame = frame_queue.get()
            # cap = cv2.VideoCapture(self.camera_ip)
            # ret, frame = cap.read()
            # if not ret:
            #     break
            if frame is not None:
                frame = imutils.resize(frame, width=720)
                h, w = frame.shape[:2]
                # 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道，返回一个4通道的blob(blob可以简单理解为一个N维的数组，
                # 用于神经网络的输入)
                blobImage = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
                self.net.setInput(blobImage)
                cvout = self.net.forward()
                image_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                font = ImageFont.truetype(self.font, 20)
                draw = ImageDraw.Draw(image_PIL)
                frame_data_list, face_number = cate_frame(frame=frame, cvout=cvout, h=h, w=w, predictor=self.predictor,
                                                          facerec=self.facerec, lStart=self.lStart, lEnd=self.lEnd,
                                                          rStart=self.rStart, rEnd=self.rEnd, mStart=self.mStart,
                                                          mEnd=self.mEnd)

                if len(self.frame_list) == 0:
                    self.frame_list = [i for i in frame_data_list]

                for i in range(0, len(frame_data_list)):
                    for j in range(0, len(self.frame_list)):
                        new_idface = frame_data_list[i]['idface']
                        old_idface = self.frame_list[j]['idface']
                        ret = face_distance(idface1=new_idface, idface2=old_idface)
                        faces = 'faces: ' + str(face_number)
                        draw.text((10, 10), text=faces, font=font, fill=(255, 0, 0))

                        if ret == 1:
                            x, y, w, h = self.frame_list[j]['face']
                            # 满足闭眼条件的判断阈值
                            new_ear = frame_data_list[i]['ear']
                            # 满足打哈欠的判断阈值
                            new_mar = frame_data_list[i]['mar']
                            # 满足点头判断阈值
                            new_har = frame_data_list[i]['har']

                            # ------------------------判断瞌睡闭眼-----------------------------
                            if self.current_speed >= self.speed_threshold:
                                # 默认长宽比0.2
                                if new_ear < self.eye_ar_thresh:
                                    self.frame_list[j]['ear_number'] += 1
                                else:
                                    ear_number = self.frame_list[j]['ear_number']
                                    if ear_number > self.eye_ar_consec_frames:
                                        self.frame_list[j]['ear_total'] += 1

                                    self.frame_list[j]['ear_number'] = 0

                            ear_total = self.frame_list[j]['ear_total']
                            ear_text = '瞌睡闭眼次数：' + str(ear_total)
                            draw.text((x, y), text=ear_text, font=font, fill=(255, 0, 0))
                            if ear_total > self.blink_threshold:
                                # 显示中文
                                draw.text((x, y + 20), text='请不要打瞌睡!!!', font=font, fill=(255, 0, 0))
                                self.frame_list[j]['ear_total'] = 0

                            #------------------------判断瞌睡哈欠-----------------------------
                            if self.current_speed >= self.speed_threshold:
                                if new_mar > self.mar_thresh:  # 张嘴阈值
                                    self.frame_list[j]['mar_number'] += 1
                                else:
                                    mar_number = self.frame_list[j]['mar_number']
                                    if mar_number >= self.mouth_arconsec_frames:
                                        self.frame_list[j]['mar_total'] += 1

                                    self.frame_list[j]['mar_number'] = 0

                            mar_total = self.frame_list[j]['mar_total']
                            mar_text = '打哈欠次数：' + str(mar_total)
                            draw.text((x, y + 40), text=mar_text, font=font, fill=(255, 0, 0))
                            if mar_total > self.yawn_threshold:
                                draw.text((x, y + 50), text='打哈欠，疑似打瞌睡!!!', font=font, fill=(255, 0, 0))
                                self.frame_list[j]['mar_total'] = 0

                            #------------------------判断瞌睡哈欠-----------------------------
                            if self.current_speed >= self.speed_threshold:
                                if new_har > self.har_thresh:
                                    self.frame_list[j]['har_number'] += 1
                                else:
                                    har_number = self.frame_list[j]['har_number']
                                    if har_number > self.nod_ar_consec_frames:
                                        self.frame_list[j]['har_total'] += 1
                                    self.frame_list[j]['har_number'] = 0
                            har_total = self.frame_list[j]['har_total']
                            har_text = '点头次数: ' + str(har_total)
                            draw.text((x, y + 70), text=har_text, font=font, fill=(255, 0, 0))
                            if har_total > self.nod_threshold:
                                draw.text((x, y + 80), text='频繁点头：疑似打瞌睡!!!', font=font, fill=(255, 0, 0))
                                self.frame_list[j]['har_total'] = 0

                img2opencv = cv2.cvtColor(np.array(image_PIL), cv2.COLOR_RGB2BGR)

                cv2.imshow('head-posture', img2opencv)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

