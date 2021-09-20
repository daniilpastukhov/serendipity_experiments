import logging
import logging.handlers
import os
import psutil

process = psutil.Process(os.getpid())


def init_logging(filename: str, level=logging.DEBUG, path=''):
    log_format = "[%(asctime)-15s] %(levelname)s  %(message)s"
    logging.basicConfig(format=log_format, level=level)

    logs_dir = path
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_filename = os.path.join(logs_dir, filename)
    handler = logging.handlers.RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=10
    )
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    # root logger
    logger = logging.getLogger()
    logger.addHandler(handler)
