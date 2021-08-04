import copy
import yaml
import logging
from collections import namedtuple


class Config:
    logger = logging.getLogger(__name__)

    @staticmethod
    def load_from_file(filename):
        try:
            with open(filename, 'r') as stream:
                try:
                    config = yaml.load(stream)
                    return Config._convert(config)
                except yaml.YAMLError as exc:
                    Config.logger.fatal('config: Cannot load config: {}'.format(filename), exc)
        except Exception as e:
            Config.logger.error('config: File not found: {}'.format(filename))
            raise e

    @staticmethod
    def load_from_db(setting_name):
        raise NotImplementedError

    @staticmethod
    def from_dict(dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError('Given argument is not a dictionary')
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
        elif isinstance(dictionary, list):
            return [Config._convert(item) for item in dictionary]
        else:
            return dictionary


if __name__ == "__main__":
    original_dict = {
        "X": {
            "GB": {
                "ERR_SV": {
                    "MIN_CASES": 1,
                    "MAN_CASES": 5,
                    "THRESHOLD": 2
                },
                "ERR_PR": {
                    "MIN_CASES": 1,
                    "MAN_CASES": 5,
                    "THRESHOLD": 2
                },
                "ERR_TO": {
                    "MIN_CASES": 1,
                    "MAN_CASES": 5,
                    "THRESHOLD": 2
                },
            },
            "DE": {
                "ERR_SV": {
                    "MIN_CASES": 1,
                    "MAN_CASES": 5,
                    "THRESHOLD": 2
                },
                "ERR_PR": {
                    "MIN_CASES": 1,
                    "MAN_CASES": 5,
                    "THRESHOLD": 2
                },
                "ERR_TO": {
                    "MIN_CASES": 1,
                    "MAN_CASES": 5,
                    "THRESHOLD": 2
                },
            }

        }
    }
    knee_config = Config.from_dict(original_dict)

    # print(type(original_dict))
    # print(original_dict)
    # print(type(knee_config))
    # print(knee_config)

    # print(knee_config.X.GB)
    print(getattr(getattr(knee_config, "X"), "GB"))


