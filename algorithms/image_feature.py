# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/10/19
# @File:          image_feature.py
# @Software:      PyCharm
# @Project:       face_contrast

import os
import dlib
import glob
import numpy as np
import cv2
import time

from config import config
from multiprocessing.dummy import Pool as ThreadPool


def file_name_list(faces_folder_path):
    """
    :param faces_folder_path:
    :return:
    """
    file_name_list = []
    for files in glob.glob(os.path.join(faces_folder_path, '*.jpg')):
        files = files.replace('\\', '/')
        file_name_list.append(files)

    return file_name_list


def face_feature_list(faces_folder_path, detector, predictor, facerec):
    """
    从图片文件夹中识别所有的图片，并计算特征值
    :param predictor:
    :param facerec:
    :return:
    """
    face_name_list = []
    file_name_lists = file_name_list(faces_folder_path=faces_folder_path)
    for file_name in file_name_lists:
        filename = file_name.split('/')[-1].split('.')[0]
        cap = cv2.VideoCapture(file_name)
        ret, frame = cap.read()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detes = detector(gray, 1)

            for k, d in enumerate(detes):
                shape = predictor(frame, d)

                # 提取特征
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                feature = np.array(face_descriptor)
                file_name = str(faces_folder_path).split('/')[-1].split('.')[0]
                face_name_dict = {
                    'name': filename,
                    'feature': feature
                }
                face_name_list.append(face_name_dict)

    return face_name_list
























