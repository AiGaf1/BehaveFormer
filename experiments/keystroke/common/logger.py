import logging
import inspect
import os


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        frame = inspect.stack()[1]
        name = os.path.basename(frame.filename)

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
