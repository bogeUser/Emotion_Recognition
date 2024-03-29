# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/09/21
# @File:          utils.py
# @Software:      PyCharm
# @Project:       Emotion_Recognition

import numpy as np
import cv2
import math

from scipy.spatial import distance as dist


# ******************************世界坐标系，相机坐标系，图像中心坐标系 像素坐标系简单关系********************************

# 世界坐标系(UVW)：填写3D参考点，该模型参考 http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]

# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def eye_aspect_ratio(eye):
    """
    计算眼睛长宽比， 当长宽比为0，或者趋接近于0的时候，表示眼睛闭合，所以可以根据这个原理判断闭眼的情况
    :param eye: 眼部ndarray值
    :return: 眼睛长宽比
    """
    # 垂直眼标志(X, Y)坐标

    # 计算两个集合之间的欧式距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 水平眼标志 (X, Y) 坐标

    # 计算两个集合之间的欧几里得距离
    C = dist.euclidean(eye[0], eye[3])

    # 计算眼睛长宽比
    ear = (A + B) / (2.0 * C)

    return ear

def average_ear(left_eye, right_eye):
    """
    计算左右眼的平均ear
    :param left_eye: 左眼坐标
    :param right_eye: 右眼坐标
    :return:
    """
    left_eye_ear = eye_aspect_ratio(left_eye)
    right_eye_ear = eye_aspect_ratio(right_eye)
    ear = (left_eye_ear + right_eye_ear) / 2.0

    return ear

def mouth_aspect_ratio(mouth):
    """
    计算嘴部长宽比
    :param mouth: 嘴部ndarray值
    :return:
    """

    # 嘴部第51，59 号位置
    A = np.linalg.norm(mouth[2] - mouth[9])
    # 嘴部第53，57 号位置
    B = np.linalg.norm(mouth[4] - mouth[7])
    # 嘴部第49，55 号位置
    C = np.linalg.norm(mouth[0] - mouth[6])

    mar = (A + B) / (2.0 * C)

    return mar

def get_head_pose(i, head):
    """
    头部姿态估计
    :param head: 头部ndarray值
    :return:
    """

    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32([
        head[17], head[21], head[22], head[26], head[36],
        head[39], head[42], head[45], head[31], head[35],
        head[48], head[54], head[57], head[8]
    ])

    # solvePnP 计算姿势-求解旋转和平移矩阵
    # rotation_vec 表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与k矩阵对应，dist_coeffs 与 D矩阵对应
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # projectPoints重投影误差：原2d点和重投影2d点的距离(输入3d点、相机内参、相机畸变、r、t，输出重投影2d点)
    project_dst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    project_dst = tuple(map(tuple, project_dst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角 calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式(将旋转矩阵转换为旋转向量)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接， vconcat 垂直拼接

    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    """
    目不正视前方计算
    """
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
    # 上下晃动
    pitch = math.degrees(math.asin(math.sin(pitch)))
    # 左右偏头
    yaw = math.degrees(math.asin(math.sin(yaw)))
    # 左右晃动
    roll = -math.degrees(math.asin(math.sin(roll)))

    # print('第%d张人脸的点头角度: %d, 左右摇头角度: %d, 左右偏头角度: %d' % (i, pitch, yaw, roll))

    return project_dst, euler_angle

