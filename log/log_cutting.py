# @Author:        QiuYong Chen
# @contact:       chen.qiuyong@stee.stengg.com.cn
# @time:          2020/09/21
# @File:          log_cutting.py
# @Software:      PyCharm
# @Project:       Emotion_Recognition


import os
import re
import datetime
import logging

try:
    import codecs
except ImportError:
    codecs = None


class LogCutting(logging.FileHandler):
    """
    支持多进程的TimedRotatingFileHandler
    """

    def __init__(self, filename, when='D', backup_count: int = 0, encoding=None, tag: str = '', delay=False):
        """
        日志分割策略
        :param filename: 日志文件名
        :param when: 时间间隔的单位
        :param backup_count: 保留文件个数
        :param encoding: 编码
        :param tag: 文件名中的标签, 如果没有, 那么就按所有文件来做, 否者就按日志文件名中包含了此标签名的日志做处理
        :param delay: 是否开启 OutSteam缓存
        """
        self.__prefix = filename
        self.__backupCount = backup_count

        self.__tag = tag

        # 正则匹配 年-月-日
        self.__extMath = r"^\d{4}-\d{2}-\d{2}"

        if when is None:
            when = 'D'

        self.__when = when.upper()

        when_dict = {
            'S': "%Y-%m-%d-%H-%M-%S",  # 按秒切割日志文件
            'M': "%Y-%m-%d-%H-%M",   # 按分钟切割日志文件
            'H': "%Y-%m-%d-%H",  # 按小时切割日志文件
            'D': "%Y-%m-%d"   # 按天切割日志文件
        }

        # 日志文件日期后缀
        self.__suffix = when_dict.get(self.__when)
        # 拼接文件路径 格式化字符串
        self.__file_fmt = "%s %s" % (self.__prefix, self.__suffix)
        self.__file_fmt = self.__file_fmt.strip()
        # 使用当前时间，格式化文件格式化字符串
        self.__current_file_path = datetime.datetime.now().strftime(self.__file_fmt)
        self.baseFilename = os.path.abspath(self.__current_file_path)

        # 获得文件夹路径
        _dir = os.path.dirname(self.__file_fmt)
        try:
            # 如果日志文件夹不存在，则创建文件夹
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception as e:
            raise Exception("创建文件夹失败: %s" % e)

        if codecs is None:
            encoding = None

        logging.FileHandler.__init__(self, self.baseFilename, 'a+', encoding, delay)

    def __need_change(self):
        """
        更改日志写入目的写入文件
        :return True 表示已更改，False 表示未更改
        """
        # 以当前时间获得新日志文件路径
        now_file_path = datetime.datetime.now().strftime(self.__file_fmt)
        # 新日志文件日期 不等于 旧日志文件日期，则表示 已经到了日志切分的时候
        if now_file_path != self.__current_file_path:
            self.__current_file_path = now_file_path
            return True
        return False

    def __do_change_file(self):
        """
        输出信息到日志文件，并删除多于保留个数的所有日志文件
        :return
        """
        # 日志文件的绝对路径
        self.baseFilename = os.path.abspath(self.__current_file_path)

        # stream == OutStream
        # stream is not None 表示 OutStream中还有未输出完的缓存数据
        if self.stream:
            # flush close 都会刷新缓冲区，flush不会关闭stream，close则关闭stream
            self.stream.close()
            # 关闭stream后必须重新设置stream为None，否则会造成对已关闭文件进行IO操作。
            self.stream = None

        # delay 为False 表示 不OutStream不缓存数据 直接输出
        if not self.delay:
            # 这个地方如果关闭colse那么就会造成进程往已关闭的文件中写数据，从而造成IO错误
            # 从新打开一次stream
            self.stream = self._open()

        # 删除多于保留个数的所有日志文件
        if self.__backupCount > 0:
            # 删除日志
            histories = self.__get_histories()
            for history in histories:
                os.remove(history)

    def __get_histories(self):
        """
        获得过期需要删除的日志文件
        :return:
        """
        # 分离出日志文件夹绝对路径
        dir_name, _ = os.path.split(self.baseFilename)
        file_names = os.listdir(dir_name)

        result = []
        regex = re.compile(self.__extMath)

        for file_name in file_names:
            if self.__tag in file_name:  # 对包含了标签的名字才做处理
                suffix = str(file_name).split(' ')[-1].strip()
                # 匹配符合规则的日志文件，添加到result列表中
                if regex.match(suffix):
                    result.append(os.path.join(dir_name, file_name))

        result.sort()

        if len(result) > self.__backupCount:

            result = result[:len(result) - self.__backupCount]
        else:
            result = []
        return result

    def emit(self, record):
        """
        发送一个日志记录, 覆盖FileHandler中的emit方法，logging会自动调用此方法
        :param record:
        :return:
        """
        try:
            if self.__need_change():
                self.__do_change_file()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            self.handleError(record)
            raise Exception(e)
