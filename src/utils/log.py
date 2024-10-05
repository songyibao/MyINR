import os
import logging
from datetime import datetime

class GlobalLogger:
    _instance = None  # 单例模式

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, name='GlobalLogger', log_level=None, log_to_console=True, log_to_file=False, log_file_path=None):
        if self._initialized:
            return

        # 如果没有显式传入 log_level，从环境变量中读取
        if log_level is None:
            log_level = self._get_log_level_from_env()

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # 创建日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 控制台日志处理器
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # 文件日志处理器
        if log_to_file:
            # 如果没有指定日志文件路径，则使用默认的路径和文件名
            if log_file_path is None:
                # 获取当前时间的时间戳
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file_path = f'./log_{timestamp}.log'

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self._initialized = True

    def get_logger(self):
        return self.logger

    def set_level(self, level):
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def add_console_handler(self, level=logging.DEBUG):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def add_file_handler(self, log_file_path, level=logging.DEBUG):
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    @staticmethod
    def _get_log_level_from_env():
        """
        从环境变量中获取日志等级，默认为 INFO
        """
        log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
        # 映射日志等级的字符串到 logging 模块中的常量
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(log_level_str, logging.INFO)

# 使用 mlflow 的 logger
logger = logging.getLogger("mlflow")


