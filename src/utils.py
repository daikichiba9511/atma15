import uuid
from datetime import datetime
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Final

__all__ = ["get_stream_logger", "add_file_handler"]

logger = getLogger(__name__)

DEFAULT_FORMAT: Final[
    str
] = "[%(levelname)s] %(asctime)s - %(pathname)s : %(lineno)d : %(funcName)s : %(message)s"


def get_stream_logger(level: int = INFO, format: str = DEFAULT_FORMAT) -> Logger:
    logger = getLogger()
    logger.setLevel(level)

    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(Formatter(format))
    logger.addHandler(handler)
    return logger


def add_file_handler(
    logger: Logger, filename: str, level: int = INFO, format: str = DEFAULT_FORMAT
) -> None:
    handler = FileHandler(filename=filename)
    handler.setLevel(level)
    handler.setFormatter(Formatter(format))
    logger.addHandler(handler)


def get_called_time_str() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_unique_id(length: int = 8) -> str:
    unique_id = uuid.uuid4().hex
    if length > len(unique_id):
        raise ValueError(f"length must be less than or equal to {len(unique_id)}")
    return unique_id[:length]
