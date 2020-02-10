"""
Script to make Python logging using the logging module automatically output to stdout
"""
import logging
import sys


class Logger:
    """
      Custom logging class with stdout stream handler enabled
    """

    def __init__(self):
        self.logging_format = '[%(pathname)s][%(funcName)s:%(lineno)d]' + \
                              '[%(levelname)s] %(message)s'
        self.logging_stream = sys.stdout

    def get_logger(self, logger_name, log_level=logging.DEBUG):
        """
        configure a stream handler (using stdout instead of the default stderr)
        and add it to the root logger
        """
        lo_gger = logging.getLogger(logger_name)
        lo_gger.setLevel(log_level)
        stream_handler = logging.StreamHandler(self.logging_stream)
        formatter = logging.Formatter(self.logging_format)
        stream_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(log_level)
        lo_gger.addHandler(stream_handler)
        lo_gger.propagate = False

        return lo_gger
