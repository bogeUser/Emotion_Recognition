# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/10/19
# @File:          face_contrast.py
# @Software:      PyCharm
# @Project:       face_contrast

"""
    计算人脸相似度
"""

import dlib
import numpy as np

from scipy.spatial import distance


def face_distance(idface1, idface2, threshold=0.04):
    """
    计算两张人脸相似度
    :param idface1: 第一张人脸
    :param idface2: 第二张人脸
    :return:
    """
    face1 = idface1
    face2= idface2
    dist = distance.cosine(face1, face2)
    if dist > threshold:
        ret = 0
    else:
        ret = 1

    return ret


def face_similarity(frame1, frame2, face1, face2, detector, predictor, facerec, threshold=0.04):
    """
    计算人脸相似度
    :param frame1:  第一张照片
    :param frame2: 第二张照片
    :param face1: 第一张照片的人脸坐标
    :param face2: 第二张照片的人脸坐标
    :param detector:
    :param predictor:
    :param facerec:
    :param threshold:
    :return:
    """

    # 从检测到的人脸中提取坐标点
    fx1, fy1, fw1, fh1 = face1
    fx2, fy2, fw2, fh2 = face2

    dets1 = dlib.rectangle(fx1, fy1, fw1 + fx1, fh1 + fy1)
    dets2 = dlib.rectangle(fx2, fy2, fw2 + fx2, fh2 + fy2)

    shape1 = predictor(frame1, dets1)
    shape2 = predictor(frame2, dets2)

    cla1 = facerec.compute_face_descriptor(frame1, shape1)
    cla2 = facerec.compute_face_descriptor(frame2, shape2)

    id_feature1 = np.array(cla1)
    id_feature2 = np.array(cla2)

    # 计算余弦相似度
    dist = distance.cosine(id_feature1, id_feature2)

    if dist > threshold:
        ret = 0
    else:
        ret = 1

    return ret