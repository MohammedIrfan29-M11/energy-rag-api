import logging
import sys
from datetime import datetime

def setup_logging():
    """Sets up logging configuration for the application.
    Logs will be written to both the console and a file named 'app.log'.
    The log format includes the timestamp, log level, and message.
    Call this once at startup in main.py.
    Everything flows through this single configuration.
    """

    logger = logging.getLogger('app')
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    console_formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(name)s| %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y-%m-%d")}.log')
    file_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger