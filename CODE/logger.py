import logging.config


logging.config.fileConfig('logging.config', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
