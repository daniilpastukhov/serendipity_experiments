import configparser


def parse_config(path) -> configparser.ConfigParser:
    """
    Read and return the config from specified path.
    :param path: Path to the config as string.
    :return: ConfigParser instance.
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config
