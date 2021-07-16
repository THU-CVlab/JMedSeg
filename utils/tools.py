import os
import time
import argparse
import logging
import logging.config
import numpy as np
import matplotlib.pyplot as plt


def timewrapper(logger=None):
    def unpacker(method):
        def timed(*args, **kw):
            start_time = time.time()
            result = method(*args, **kw)
            end_time = time.time()
            if logger is not None:
                logger.info('{} finished in {:.2f} s.'.format(method.__name__,end_time - start_time))
            else:
                print('{} finished in {:.2f} s.'.format(method.__name__,end_time - start_time))
            return result
        return timed
    return unpacker

def setupLogger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    dir_path = os.path.split(log_file)[0]
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except Exception as e:
        print(e)
        exit(0)

    # formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s','%Y-%m-%d %H:%M:%S')
    handler = logging.handlers.RotatingFileHandler(log_file)        
    handler.setFormatter(formatter)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(handler2)

    return logger

