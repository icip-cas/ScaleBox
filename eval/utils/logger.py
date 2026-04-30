# Centralized logging configuration for ScaleBox evaluation framework.

import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "scalebox.eval",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    # Setup and return a configured logger instance.
    # Args:
    # name: Logger name
    # level: Logging level (default: INFO)
    # format_string: Custom format string (optional)
    # Returns:
    # Configured logger instance
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger

def get_logger(name: str = "scalebox.eval") -> logging.Logger:
    # Get or create a logger instance.
    return logging.getLogger(name)
