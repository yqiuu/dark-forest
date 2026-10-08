import os
import psutil
import logging


__all__ = ["logger", "dividing_line", "memory_usage"]


def create_logger(name):
    """Create a logger with the Dark Forest console format."""
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("[dark-forest] %(levelname)s: %(message)s"))
    logger.addHandler(stream_handler)
    return logger


logger = create_logger(__name__)


def dividing_line():
    return "-" * 60


def memory_usage():
    """Return the current process's resident memory as a formatted GB value."""
    mem = psutil.Process(os.getpid()).memory_info()[0] / 1024**3
    return f"{mem:.2f} GB"
