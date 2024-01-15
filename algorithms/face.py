# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/11/07
# @File:          face.py
# @Software:      PyCharm
# @Project:       face_contrast

import cv2
import numpy as np

from scipy.spatial import distance

from algorithms.emotion_recognition import emotion_recognition
from algorithms.coordinate2points import coordinate2points
from algorithms.utils import get_head_pose
from algorithms.utils import average_ear
from algorithms.utils import mouth_aspect_ratio


def deal_frame(frame, cvout, h, w, face_name_list, predictor, facerec):
    """
    识别人脸坐标，数据库中存在的人脸为绿色框，不存在的人脸为红色框
    :param frame: 视频帧
    :param cvout: 人脸坐标点
    :param h: 视频帧的高度
    :param w: 视频帧的高度
    :param face_name_list: 人脸数据列表
    :return: 返回数据库中存在的人脸坐标和人脸的姓名
    """

    data_list = []
    for detection in cvout[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.5:
            left = int(detection[3] * w)
            top = int(detection[4] * h)
            right = int(detection[5] * w)
            bottom = int(detection[6] * h)

            face = np.array([left, top, right - left, bottom - top])
            # 计算当前待识别人脸128个关键点的特征值
            face_feature, _ = coordinate2points(frame=frame, face=face, predictor=predictor, facerec=facerec)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            for dic in face_name_list:
                # 计算余弦值距离
                feature = dic['feature']
                dist = distance.cosine(face_feature, feature)
                if dist < 0.04:
                    name = dic['name']
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    data_dic = {
                        'frame': frame,
                        'name': name,
                        'coordinate': [left, top, right, bottom]
                    }

                    data_list.append(data_dic)

    return data_list


def get_face(frame, cvout, h, w, emotions, emotion_classifier, predictor, facerec):
    """
        获取脸部情绪和该人脸的脸部坐标点
    :param frame: 视频帧
    :param cvout: 人脸数据
    :param h: 高度
    :param w: 宽度
    :param emotions: 情绪列表
    :param emotion_classifier:
    :param predictor:
    :param facerec:
    :return: 该人脸的人脸情况和脸部坐标点列表
    """
    data_list = []
    face_number = 0
    for detection in cvout[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.5:
            left = int(detection[3] * w)
            top = int(detection[4] * h)
            right = int(detection[5] * w)
            bottom = int(detection[6] * h)
            face = np.array([left, top, right - left, bottom - top])
            face_number += 1
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            emotions_list = emotion_recognition(frame=frame, face=face, emotions=emotions,
                                                emotion_classifier=emotion_classifier, predictor=predictor,
                                                facerec=facerec)
            for facial_expression in emotions_list:
                emotion = facial_expression['facial_expression']

                data_dic = {
                    'frame': frame,
                    'face': face,  # 脸部坐标点
                    'emotion': emotion  # 该人脸的情绪
                }
                data_list.append(data_dic)

    return (data_list, face_number)


def cate_frame(frame, cvout, h, w, predictor, facerec, lStart, lEnd, rStart, rEnd, mStart, mEnd):
    """

    :param frame:  当前视频帧
    :param cvout: 返回的人脸数据
    :param h: 视频帧的height
    :param w: 视频帧的width
    :param predictor:
    :param facerec:
    :param lStart: 左眼面部标志起始索引
    :param lEnd: 左眼面部标志结束索引
    :param rStart: 右眼面部标志起始索引
    :param rEnd: 右眼面部标志结束索引
    :param mStart: 嘴部标志起始索引
    :param mEnd: 嘴部标志结束索引
    :return:
    """
    face_data_list = []
    face_number = 0
    for detection in cvout[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.5:
            # 人脸关键点检测
            left = int(detection[3] * w)
            top = int(detection[4] * h)
            right = int(detection[5] * w)
            bottom = int(detection[6] * h)
            face = np.array([left, top, right - left, bottom - top])
            face_number += 1
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            idface, shape_face = coordinate2points(frame=frame, face=face, predictor=predictor,
                                                   facerec=facerec)
            left_eye = shape_face[lStart: lEnd]
            right_eye = shape_face[rStart: rEnd]
            mouth = shape_face[mStart: mEnd]

            # 头部姿态估计
            reprojectdst, euler_angle = get_head_pose(0, shape_face)

            # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            mouth_hull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth_hull], -1, [0, 255, 0], 1)

            for (x, y) in shape_face:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # for start, end in line_pairs:
            #     cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

            ear = average_ear(left_eye=left_eye, right_eye=right_eye)
            mar = mouth_aspect_ratio(mouth=mouth)
            har = euler_angle[0, 0]

            face_dic = {
                'frame': frame,
                'face': face,
                'idface': idface,
                'ear': ear,
                'mar': mar,
                'har': har,
                'euler_angle': euler_angle,
                'ear_number': 0,
                'mar_number': 0,
                'har_number': 0,
                'ear_total': 0,
                'mar_total': 0,
                'har_total': 0
            }

            face_data_list.append(face_dic)

    return (face_data_list, face_number)