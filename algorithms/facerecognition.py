# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/10/20
# @File:          facerecognition.py
# @Software:      PyCharm
# @Project:       face_contrast


"""
    使用face_recognition 进行人脸识别
"""

import face_recognition
import cv2
import numpy as np
from deepface import DeepFace
from config import config

# face_detection = cv2.CascadeClassifier(config.face_detection_model_path)
# while True:
#     cap = cv2.VideoCapture(r'../image/猩猩.jpg')
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
#                                                  minSize=(30, 30),
#                                                  flags=cv2.CASCADE_SCALE_IMAGE)
#     for face in faces:
#         x, y, w, h = face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     cv2.imshow('', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# while True:
#     cap = cv2.VideoCapture(r'../image/dogs.jpg')
#     # cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_locations = face_recognition.face_locations(gray)
#
#     for face in face_locations:
#         top = face[0]
#         right = face[1]
#         bottom = face[2]
#         left = face[3]
#
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#     cv2.imshow('', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break



"""
    OpenCV 基于残差网络的人脸识别
"""

model_bin = r'../model/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config_text = r'../model/deploy.prototxt'

# 加载模型和模型的自述文件  config_text: 模型配置文件，model_bin：模型的权重二进制文件
net = cv2.dnn.readNetFromCaffe(config_text, model_bin)

# set back-end
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# cap = cv2.VideoCapture(r'../image/dog.jpg')
while True:
    cap = cv2.VideoCapture(r'../image/dog.jpg')
    ret, image = cap.read()
    # image = cv2.flip(image, 1)  # 镜像翻转
    if ret is False:
        break

    # 人脸检测
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    # 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等，返回一个4通道的blob(blob可以简单理解为一个N维的数组，用于神经网络的输入)
    # iamge: 输入图像 scalefactor: 默认1.0 size：表示网络接受的数据大小 mean：表示训练时数据集的均值 crop：剪切 ddepth：数据类型
    blobImage = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    # 运行模型
    net.setInput(blobImage)
    cvout = net.forward()

    # put efficiency information
    # t, _ = net.getPerfProfile()
    # fps = 1000 / (t * 1000 / cv2.getTickFrequency())
    # lable = 'FPS: %.2f' % fps
    # print(lable)
    # cv2.putText(image, lable, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    # 绘制检测矩形
    for detection in cvout[0,0,:,:]:
        score = float(detection[2])
        objIndex = int(detection[1])
        if score > 0.5:
            print(score)
            left = detection[3] * w
            top = detection[4] * h
            right = detection[5] * w
            bottom = detection[6] * h

            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
            # cv2.putText(image, 'score: %.2f' %score, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


































