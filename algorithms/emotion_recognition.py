# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/11/08
# @File:          emotion_recognition.py
# @Software:      PyCharm
# @Project:       face_contrast


import imutils
import cv2
import numpy as np
import tensorflow as tf


from keras.preprocessing.image import img_to_array
from algorithms.coordinate2points import coordinate2points

graph = tf.get_default_graph()

def emotion_recognition(frame, face, emotions, emotion_classifier, predictor, facerec):
    """
    人脸情绪识别 （情绪识别模型为： model/emotion_model.hdf5）
    :param frame:  当前视频帧
    :param emotions: 情绪列表，包含7中表情
    :param face_detection:
    :param emotion_classifier:
    :param predictor:
    :param facerec:
    :param detectorr:
    :return:
    """
    with graph.as_default():
        if frame is not None:
            emotion_list = []
            frame = imutils.resize(frame, width=720)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fx, fy, fw, fh = face
            roi = gray[fy: fy + fh, fx: fx + fw]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion = emotions[preds.argmax()]
            # 获取脸部128个关键点特征值
            characteristic_value, _ = coordinate2points(frame=frame, face=face, predictor=predictor, facerec=facerec)

            data_dict = {
                'frame': frame,  # 当前视频帧
                'characteristic_value': characteristic_value,  # 人脸特征
                'face_coordinates': face,  # 人脸坐标
                'facial_expression': emotion  # 人脸表情
            }
            emotion_list.append(data_dict)

            return emotion_list