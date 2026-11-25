"""
Logger Utils

Modified from mmcv / Pointcept.
"""

import logging
import torch.distributed as dist

logger_initialized = {}


def get_root_logger(log_file=None, log_level=logging.INFO, file_mode="w"):
    """Get the root logger.

    The logger is named "pointcept" and is initialized once: subsequent calls
    return the same instance. Only rank-0 writes to file / stdout at INFO; other
    ranks are silenced to ERROR to avoid duplicated logs under DDP.
    """
    logger = logging.getLogger("pointcept")
    if "pointcept" in logger_initialized:
        return logger

    # handle hierarchical names already initialized
    for logger_name in logger_initialized:
        if "pointcept".startswith(logger_name):
            return logger

    # avoid double logging through the root logger
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 writes the log file
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized["pointcept"] = True

    return logger
