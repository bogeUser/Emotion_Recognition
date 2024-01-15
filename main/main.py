# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/09/21
# @File:          main.py
# @Software:      PyCharm
# @Project:       Emotion_Recognition


import multiprocessing
import threading

from multiprocessing import Queue

from config import config
from camera.camera import Camera


def camera_draw_name(frame_queue):
    """
    姓名
    :param frame_queue:
    :return:
    """
    camera = Camera(camera_id=0, camera_ip=r'../image/test_img/he3.jpg',
                    faces_folder_path=config.faces_folder_path)
    threading.Thread(target=camera.get_frame, args=(frame_queue, )).start()
    threading.Thread(target=camera.draw_name, args=(frame_queue, )).start()

def camera_draw_emotion(frame_queue):
    """
    情绪
    :param frame_queue:
    :return:
    """
    camera = Camera(camera_id=1, camera_ip=r'../image/test_img/emotion.jpg',
                    faces_folder_path=config.faces_folder_path)

    threading.Thread(target=camera.get_frame, args=(frame_queue, )).start()
    threading.Thread(target=camera.draw_emotion, args=(frame_queue, )).start()

def camera_draw_state(frame_queue):
    """
    头部姿态
    :param frame_queue:
    :return:
    """
    camera = Camera(camera_id=1, camera_ip=r'../image/test_img/test.jpg',
                    faces_folder_path=config.faces_folder_path)
    threading.Thread(target=camera.get_frame, args=(frame_queue, )).start()
    threading.Thread(target=camera.show_attitude, args=(frame_queue, )).start()

def run(frame_queue):
    # 人脸对比
    multiprocessing.Process(name='draw_name', target=camera_draw_name, args=(frame_queue, )).start()
    # 情绪识别
    # multiprocessing.Process(name='draw_emotion', target=camera_draw_emotion, args=(frame_queue, )).start()
    # 头部姿态
    # multiprocessing.Process(name='draw_state', target=camera_draw_state, args=(frame_queue, )).start()


if __name__ == '__main__':
    frame_queue = Queue(500)
    run(frame_queue=frame_queue)









































