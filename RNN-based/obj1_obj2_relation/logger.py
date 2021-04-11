import logging
import os

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, logpath, logfile, level='info',fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level_relations.get(level))
        format_str = logging.Formatter(fmt)
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(format_str)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(format_str)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)