import os
import sys

from environs import Env

# load .env file into os.environ

env = Env()
env.read_env()

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_SERVICE = env.str("DATA_SERVICE", "filesystem")

# Threshold calculation parameters
THRESHOLD_FREQUENCY = 0.2
THRESHOLD_TOP_SELLING_PERCENT = 0.2

# processes/threads
EXECUTOR = env.str("EXECUTOR", "processes")

OVERRIDE_ITEM_GROUP_CODE = env.str("OVERRIDE_ITEM_GROUP_CODE", "")
OVERRIDE_COUNTRY_CODE = env.str("OVERRIDE_COUNTRY_CODE", "")

DEBUG = env.bool("DEBUG", False)

# WORKERS: 3, THREADS: 5, MAX_REQUESTS: 5000, MAX_REQUESTS_JITTER: 10
WORKERS = env.int("WORKERS", 1)
THREADS = env.int("THREADS", 1)
MAX_REQUESTS = env.int("MAX_REQUESTS", 0)
MAX_REQUESTS_JITTER = env.int("MAX_REQUESTS_JITTER", 0)

PORT = env.int("PORT", 8080)

POSTGRES_HOST = env.str("POSTGRES_HOST", "localhost")
POSTGRES_PORT = env.int("POSTGRES_PORT", 5432)
POSTGRES_DB = env.str("POSTGRES_DB", "postgres")
POSTGRES_USER = env.str("POSTGRES_USER", "postgres")
POSTGRES_PASS = env.str("POSTGRES_PASS", "")
POSTGRES_TIMEOUT = env.int("POSTGRES_TIMEOUT", 600)

MIN_CONN_COUNT = env.int("MIN_CONN_COUNT", 1)
MAX_CONN_COUNT = env.int("MAX_CONN_COUNT", 10)

JSON_SORT_KEYS = False

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "simple": {"format": "[%(levelname)s] %(asctime)s - %(name)s -  %(message)s"},
        "json": {"()": "utils.CustomJsonFormatter"},
    },
    "handlers": {
        "main": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": sys.stdout,
        }
    },
    "loggers": {
        "root": {"level": "INFO", "propagate": True, "handlers": ["main"]},
    },
}
