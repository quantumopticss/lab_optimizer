import logging
import os

MAIN_SCRIPT_DIR = os.path.abspath(os.getcwd())
os.makedirs("logging", exist_ok=True)
LOG_DIR = os.path.join(MAIN_SCRIPT_DIR, "logging")

LOG_FILE = os.path.join(LOG_DIR, "mypackage.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.info("Logging initialized in %s", LOG_FILE)
