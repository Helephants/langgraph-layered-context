"""
Logging utilities for the framework.
"""
import sys
from loguru import logger as loguru_logger
from typing import Optional


def setup_logger(
    name: str = "context-rag",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Configure loguru logger.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
    """
    # Remove default handler
    loguru_logger.remove()

    # Add console handler
    loguru_logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )

    # Add file handler if specified
    if log_file:
        loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="500 MB",
        )


def get_logger(name: str):
    """Get a logger instance."""
    return loguru_logger.bind(name=name)
