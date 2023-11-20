"""
Logging module
"""
import logging
#from .configs.config import get_app_settings
#from .configs.settings.base import AppEnvTypes


def log_level() -> None:
    """
    Get environment and set log level
    if env is dev then log level = DEBUG
    @return: log level
    """
    #app_env = get_app_settings().app_env
    #if app_env == AppEnvTypes.DEV:
    #    return logging.DEBUG

    return logging.DEBUG


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=log_level(),
)
log = logging.getLogger()