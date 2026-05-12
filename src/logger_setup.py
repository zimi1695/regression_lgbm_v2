"""
日志配置：所有模块通过 logging.getLogger(__name__) 使用
在主入口调用 setup_logger() 即可让日志同时输出到终端和文件
"""

import logging
import os
from datetime import datetime


def setup_logger(log_dir: str = "./logs") -> str:
    """
    初始化日志系统
    Returns: 日志文件路径
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    fmt = "[%(asctime)s] [%(levelname)s]  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 根 logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # 终端 handler（INFO 及以上）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))

    # 文件 handler（DEBUG 及以上，记录全部细节）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))

    root.addHandler(ch)
    root.addHandler(fh)

    logging.info(f"[logger] 日志文件：{log_file}")
    return log_file