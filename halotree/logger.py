import logging


__all__ = ['CreateLogger']


def CreateLogger(name):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s:%(message)s'))
    logger.addHandler(stream_handler)
    return logger

