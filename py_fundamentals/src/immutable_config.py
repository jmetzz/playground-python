import copy
import logging
from collections import namedtuple

import yaml


class Config:
    logger = logging.getLogger(__name__)

    @staticmethod
    def load_from_file(filename):
        try:
            with open(filename, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                    return Config._convert(config)
                except yaml.YAMLError as exc:
                    Config.logger.fatal(f"config: Cannot load config: {filename}", exc)
        except Exception as e:
            Config.logger.error("config: File not found: %s", filename)
            raise e

    @staticmethod
    def load_from_db(setting_name):
        raise NotImplementedError

    @staticmethod
    def from_dict(dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError("Given argument is not a dictionary")
        return Config._convert(copy.deepcopy(dictionary))

    @staticmethod
    def _convert(dictionary):
        """Converts a nested dictionary into an accessible object,
        allowing us to access nested property as simple as object.param.nested
        :obj: dictionary
        :returns: NestedObject
        """
        if isinstance(dictionary, dict):
            for key, value in dictionary.items():
                dictionary[key] = Config._convert(value)
            return namedtuple("CONFIG", dictionary.keys())(**dictionary)
        if isinstance(dictionary, list):
            return [Config._convert(item) for item in dictionary]
        return dictionary


if __name__ == "__main__":
    original_dict = {
        "X": {
            "GB": {
                "ERR_SV": {"MIN_CASES": 1, "MAN_CASES": 5, "THRESHOLD": 2},
                "ERR_PR": {"MIN_CASES": 1, "MAN_CASES": 5, "THRESHOLD": 2},
                "ERR_TO": {"MIN_CASES": 1, "MAN_CASES": 5, "THRESHOLD": 2},
            },
            "DE": {
                "ERR_SV": {"MIN_CASES": 1, "MAN_CASES": 5, "THRESHOLD": 2},
                "ERR_PR": {"MIN_CASES": 1, "MAN_CASES": 5, "THRESHOLD": 2},
                "ERR_TO": {"MIN_CASES": 1, "MAN_CASES": 5, "THRESHOLD": 2},
            },
        }
    }
    knee_config = Config.from_dict(original_dict)

    print(f"type(knee_config): {type(knee_config)}")
    print(knee_config)
    print(knee_config.X.GB)
