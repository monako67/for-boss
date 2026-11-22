"""Логирование с ротацией, без содержимого запросов и ключей."""
import logging.config
from config import LOG_FILE


def setup_logging():
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {'standard': {'format': '%(asctime)s %(name)s %(levelname)s %(message)s'}},
        'handlers': {
            'file': {'class': 'logging.handlers.RotatingFileHandler',
                     'filename': str(LOG_FILE), 'encoding': 'utf-8', 'maxBytes': 5 * 1024 * 1024,
                     'backupCount': 3, 'formatter': 'standard', 'level': 'INFO'},
        },
        'root': {'handlers': ['file'], 'level': 'INFO'},
    })
