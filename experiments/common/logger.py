import inspect
import logging
import os


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        frame = inspect.stack()[1]
        name = os.path.basename(frame.filename)

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(filename)s:%(lineno)d [%(name)s] %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger