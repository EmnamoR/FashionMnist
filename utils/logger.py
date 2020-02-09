import logging
import sys

class logger(object):
  def __init__(self):
    self.LOGGING_FORMAT = '[%(pathname)s][%(funcName)s:%(lineno)d]' + \
                       '[%(levelname)s] %(message)s'
    self.LOGGING_STREAM = sys.stdout

  def get_logger(self, logger_name, log_level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    ch = logging.StreamHandler(self.LOGGING_STREAM)
    formatter = logging.Formatter(self.LOGGING_FORMAT)
    ch.setFormatter(formatter)
    ch.setFormatter(formatter)
    ch.setLevel(log_level)
    logger.addHandler(ch)
    logger.propagate = False

    return logger
