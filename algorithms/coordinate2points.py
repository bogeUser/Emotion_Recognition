# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/10/19
# @File:          coordinate2points.py
# @Software:      PyCharm
# @Project:       face_contrast


"""
    人脸轮廓128个关键点的特征值
"""

import numpy as np
import dlib
import time

from imutils import face_utils


def coordinate2points(frame, face, predictor, facerec):
    """
    人脸轮廓128个关键点特征值提取
    :param frame: 视频帧
    :param face: 人脸坐标
    :param predictor:
    :param facerec:
    :return: 脸部特征值
    """
    fx, fy, fw, fh = face
    dlib_face = dlib.rectangle(fx, fy, fw + fx, fh + fy)
    shape = predictor(frame, dlib_face)
    shape_face = face_utils.shape_to_np(shape=shape)
    start = time.time()
    cla = facerec.compute_face_descriptor(frame, shape)
    print(time.time() - start)
    face_feature = np.array(cla)
    print(face_feature)

    return (face_feature, shape_face)


















