import logging
from .pyt_utils import ensure_dir


def print_and_log_info(logger: logging.Logger, string: str):
    """Log info.

    Args:
        logger (logging.Logger): Logger.
        string (str): String to log.
    """
    logger.info(string)


def get_logger(file_path: str, name="train") -> logging.Logger:
    """Get logger.

    Args:
        file_path (str): Path to log file.
        name (str, optional): Logger name. Defaults to "train".

    Returns:
        logging.Logger: Logger.
    """
    log_dir = "/".join(file_path.split("/")[:-1])
    ensure_dir(log_dir)

    logger = logging.getLogger(name)
    hdlr = logging.FileHandler(file_path, mode="a")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
