import logging


__all__ = ['logger']


def CreateLogger(name):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s [dark-forest]'))
    logger.addHandler(stream_handler)
    return logger
logger = CreateLogger(__name__)
