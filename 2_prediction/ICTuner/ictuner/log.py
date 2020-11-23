import logging

def get_logger(level=logging.DEBUG):
    logging.basicConfig(format='[%(asctime)s] - %(levelname)s :: %(message)s', \
        level=level, \
        datefmt='%Y/%m/%d %I:%M:%S %p')
    logger = logging.getLogger('ictuner')
    logger.setLevel(level)

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')
    