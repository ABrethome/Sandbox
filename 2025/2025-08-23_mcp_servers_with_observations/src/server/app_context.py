import logging 
from pydantic import BaseModel

from src.server.loader import load_yaml_config
from src.server.constants import PATH_TO_CONFIG

logger = logging.getLogger(__name__)


class AppContext(BaseModel):
    pass


def load_app_context() -> AppContext:
    config = load_yaml_config(PATH_TO_CONFIG)
    if not config:
        raise ValueError("Config is empty. Please ensure a config file is provided.")

    logger.info("Loading the App Context...")

    return AppContext()
