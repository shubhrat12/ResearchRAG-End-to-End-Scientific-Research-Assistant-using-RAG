"""Logging utilities for LangChainGPT."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from ..config.settings import get_settings

# Default log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """Configure logging for the entire application.
    
    Args:
        level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default from settings)
        log_format: Log message format
    """
    settings = get_settings()
    
    # Use parameters or settings if not provided
    level = level or settings.logging.level
    log_format = log_format or settings.logging.format
    
    # Determine log file
    if log_file is None:
        log_file = settings.logging.file
    
    # If log file is still None, create a default log file in logs directory
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "langchain_gpt.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Log initial message
    root_logger.info(f"Logging configured: level={level}, log_file={log_file}")


def setup_logger(
    name: str, 
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Configure and return a logger.
    
    Args:
        name: The name of the logger
        level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to console only)
        log_format: Log message format
        
    Returns:
        logging.Logger: Configured logger
    """
    settings = get_settings()
    
    # Use parameters or settings if not provided
    level = level or settings.logging.level
    log_file = log_file or settings.logging.file
    log_format = log_format or settings.logging.format
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default application logger
app_logger = setup_logger("langchain_gpt")


def get_logger(name: str = "langchain_gpt") -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: The name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    if name == "langchain_gpt":
        return app_logger
    return setup_logger(name) 