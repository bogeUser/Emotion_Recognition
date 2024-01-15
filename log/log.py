# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/09/21
# @File:          log.py
# @Software:      PyCharm
# @Project:       Emotion_Recognition

"""
日志工具, 自定义日志输出格式等
"""

import logging
import os
import sys

from log.log_cutting import LogCutting


class Logger(object):
    """
    自定义日志类
    """
    def __init__(
            self,
            set_level="debug",
            name=os.path.split(os.path.splitext(sys.argv[0])[0])[-1],
            log_file_path=None,
            use_log_file=False,
            use_console=True,
            when='D',
            backup_count=0,
            encoding="utf-8",

    ):

        """
        初始化日志类
        :param set_level: 设置日志的打印级别，默认为DEBUG
        :param name: 日志中将会打印的name，默认为调用程序的name
        :param log_file_path: 日志文件夹的路径，默认为logger.py同级目录中的log文件夹下
        :param use_log_file: 是否输入日志文件，默认为False
        :param use_console: 是否在控制台打印，默认为True
       """

        self.__logger = logging.getLogger(name)

        # 格式化输出
        self.log_fmt = logging.Formatter(
            fmt='%(asctime)s [modle(%(filename)s)-method(%(funcName)s)<line:%(lineno)d>] %(threadName)s-%('
                'thread)d-> %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 日志等级
        if set_level.lower() == "critical":
            self.__logger.setLevel(logging.CRITICAL)
        elif set_level.lower() == "error":
            self.__logger.setLevel(logging.ERROR)
        elif set_level.lower() == "warning":
            self.__logger.setLevel(logging.WARNING)
        elif set_level.lower() == "info":
            self.__logger.setLevel(logging.INFO)
        elif set_level.lower() == "debug":
            self.__logger.setLevel(logging.DEBUG)
        else:
            self.__logger.setLevel(logging.NOTSET)

        # 使用日志文件输出
        if use_log_file:
            if not os.path.exists(os.path.dirname(log_file_path)):  # 日志路径不存在的话就创建
                os.makedirs(os.path.dirname(log_file_path))

            file_handler = LogCutting(
                filename=log_file_path,
                when=when,
                backup_count=backup_count,
                encoding=encoding
            )
            self.__logger.addHandler(file_handler)
            [i.setFormatter(self.log_fmt) for i in self.__logger.handlers]

        # 使用控制台输出
        if use_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.log_fmt)
            self.__logger.addHandler(console_handler)

    @property
    def logger(self):
        if not len(self.__logger.handlers) == 1:
            self.__logger.handlers = self.__logger.handlers[:1]  # 如果被实例化了多个handlers, 只取第一个
        return self.__logger

