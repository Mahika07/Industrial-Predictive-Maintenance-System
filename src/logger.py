import logging
import os
from datetime import datetime

# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Logger configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(module)s: %(message)s",
    level=logging.INFO,
)

def get_logger(name: str):
    """
    Returns a logger instance with a custom name.
    Usage:
        logger = get_logger(__name__)
        logger.info("Message...")
    """
    return logging.getLogger(name)
