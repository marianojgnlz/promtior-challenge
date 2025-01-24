import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os

class Logger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Logger._initialized:
            Logger._initialized = True
            self.setup_logging()

    def setup_logging(self):
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)

        # Component loggers setup
        components = [
            'main', 'processor', 'document_loader', 'query_analyzer',
            'retriever', 'source_processor', 'html_cleaner'
        ]

        for component in components:
            logger = logging.getLogger(component)
            logger.setLevel(logging.INFO)

            # File handler for each component
            file_handler = RotatingFileHandler(
                log_dir / f"{component}.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.INFO)
            file_format = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance for a specific component."""
        return logging.getLogger(name)

# Initialize logger on module import
logger_instance = Logger() 