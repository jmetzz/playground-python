import os

from environs import Env

# load .env file into os.environ
env = Env()
env.read_env()

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
ROOT_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, ".data")
CACHE_DIR = os.path.join(DATA_DIR, "cached_queries")


PROFILE = env.str("PROFILE", "local")
EXECUTOR = env.str("EXECUTOR", "processes")
CACHE_THREADS = env.int("CACHE_THREADS", 10)


DEBUG = env.bool("DEBUG", False)

POSTGRES_HOST = env.str("POSTGRES_HOST", "localhost")
POSTGRES_PORT = env.int("POSTGRES_PORT", 5432)
POSTGRES_DB = env.str("POSTGRES_DB", "postgres")
POSTGRES_USER = env.str("POSTGRES_USER", "postgres")
POSTGRES_PASS = env.str("POSTGRES_PASS", "")
POSTGRES_TIMEOUT = env.int("POSTGRES_TIMEOUT", 600)

MIN_CONN_COUNT = env.int("MIN_CONN_COUNT", 1)
MAX_CONN_COUNT = env.int("MAX_CONN_COUNT", 10)

JSON_SORT_KEYS = False
EXECUTOR_TYPE = "thread"
EXECUTOR_MAX_WORKERS = CACHE_THREADS

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"json": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"}},
    "handlers": {"main": {"class": "logging.StreamHandler", "formatter": "json"}},
}
