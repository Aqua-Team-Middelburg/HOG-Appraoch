"""
Logging utilities for the pipeline.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional
import colorlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    color_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Whether to log to console
        color_output: Whether to use colored console output
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    if console_output:
        if color_output:
            # Colored console handler
            console_handler = colorlog.StreamHandler(sys.stdout)
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)
        else:
            # Standard console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(format_string)
            console_handler.setFormatter(console_formatter)
        
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)
    
    return logger


class PipelineLogger:
    """
    Pipeline-specific logger with additional utilities.
    """
    
    def __init__(self, name: str, config: dict):
        """
        Initialize pipeline logger.
        
        Args:
            name: Logger name
            config: Logging configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger based on configuration."""
        log_config = self.config.get('system', {}).get('logging', {})
        
        level = log_config.get('level', 'INFO')
        format_string = log_config.get('format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_logging = log_config.get('console_logging', True)
        file_logging = log_config.get('file_logging', True)
        
        # Set up log file path
        log_file = None
        if file_logging:
            output_dir = self.config.get('paths', {}).get('output_dir', 'output')
            logs_dir = self.config.get('paths', {}).get('logs_dir', 'logs')
            log_file = os.path.join(output_dir, logs_dir, f'{self.name}.log')
        
        return setup_logging(
            level=level,
            log_file=log_file,
            console_output=console_logging,
            format_string=format_string
        )
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(f"[{self.name}] {message}")
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(f"[{self.name}] {message}")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(f"[{self.name}] {message}")
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(f"[{self.name}] {message}")
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(f"[{self.name}] {message}")


def get_pipeline_logger(name: str, config: dict) -> PipelineLogger:
    """
    Get a pipeline logger instance.
    
    Args:
        name: Logger name
        config: Configuration dictionary
        
    Returns:
        Pipeline logger instance
    """
    return PipelineLogger(name, config)