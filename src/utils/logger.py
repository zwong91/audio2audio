import logging
import time

from functools import wraps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
class TimingLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_timing(self, func_name: str, elapsed_time: float):
        self.logger.info(f"{func_name} took {elapsed_time:.2f} seconds")
    
    def error(self, message: str):
        self.logger.error(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


def timer_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.log_timing(func.__name__, elapsed_time)
        return result
    return wrapper