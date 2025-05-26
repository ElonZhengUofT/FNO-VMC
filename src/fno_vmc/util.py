import yaml
import logging


def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_logger(logfile: str = None, level=logging.INFO):
    """Configure root logger."""
    logger = logging.getLogger()
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)