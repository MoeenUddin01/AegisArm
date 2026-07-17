"""Logging configuration for JointGuardian.

Provides a thin wrapper around the standard library ``logging`` module.
All application code should import and use ``get_logger(__name__)`` rather
than calling ``print()`` directly — except in ``scripts/`` CLI
entrypoints where console output is the intended UX.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger for the application.

    Call once at the start of a script.  Subsequent ``get_logger()``
    calls inherit this configuration.

    Args:
        level: Minimum severity to emit (default ``INFO``).
        log_file: Optional file path — when provided, logs are written
            to both stderr and this file.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
