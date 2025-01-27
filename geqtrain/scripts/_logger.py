import logging
import sys


def set_up_script_logger(verbose: str = "INFO"):
    level = getattr(logging, verbose.upper())
    logging.basicConfig(level=level,)
