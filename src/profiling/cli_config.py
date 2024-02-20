"""Holds cli module specific configuration values"""

import os

from environs import Env

THIS_DIR = os.path.dirname(__file__)

env = Env()
env.read_env()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "simple": {"format": "[%(levelname)s] %(asctime)s - %(name)s -  %(message)s"},
    },
}
