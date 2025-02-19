from __future__ import annotations

from pathlib import Path

import appdirs
import yaml
from packaging import version

APPDIRS = appdirs.AppDirs(appname="hawcsimulator", appauthor="usask-arg")


def user_config_file_location() -> Path:
    """
    File location of the user config file
    """
    user_config_dir = APPDIRS.user_config_dir

    return Path(user_config_dir).joinpath("config.yml")


def load_user_config():
    """
    Opens the user config file as a dictionary
    """
    user_config_file = user_config_file_location()

    try:
        with Path.open(user_config_file) as f:
            if version.parse(yaml.__version__) > version.parse("5"):
                config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                config = yaml.load(f)

        if config is not None:
            return config
        return {}
    except FileNotFoundError:
        return {}


def save_user_config(user_config: dict):
    """
    Saves the user config from a dictionry
    """
    user_config_file = user_config_file_location()

    Path(user_config_file).parent.mkdir(exist_ok=True, parents=True)

    with Path.open(user_config_file, "w") as f:
        yaml.dump(user_config, f, default_flow_style=False)
