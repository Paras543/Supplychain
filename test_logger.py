import sys
from src.logger import logging
from src.exception import CustomException

if __name__ == "__main__":
    logging.info("Testing logger")
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Divide by zero error")
        raise CustomException(e, sys)
    
    