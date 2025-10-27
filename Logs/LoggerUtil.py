import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_logger(log_dir: str | Path = None, 
               backup_count: int = 7
               ) -> logging.Logger:
    """
    日志记录器。

    Args:
        log_dir (str | Path, optional): 日志目录，默认是当前文件同级的 "Logs"。
        backup_count (int, optional): 保留的旧日志个数，默认 7。
    """
    # 日志目录
    if log_dir is None:
        log_dir = Path(__file__).parent / "Logs"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(exist_ok=True)  # 如果不存在就创建
    # 创建 handler，每天生成一个新日志文件
    handler = TimedRotatingFileHandler(filename=log_dir / "record.log", # 基础文件名
                                        when="midnight",                # 每天零点切分
                                        interval=1,                     # 间隔 1 天
                                        backupCount=backup_count,       # 保留最近 7 天日志
                                        encoding="utf-8",
                                        utc=False)
    # 日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 移除 basicConfig 自动添加的 handler，避免重复写入
    if logger.hasHandlers():
        logger.handlers.clear()
    # 添加新的 handler
    logger.addHandler(handler)

    return logger
