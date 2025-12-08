import sys
from logger import get_logger

logger = get_logger(__name__)

def error_message_detail(error, error_detail: sys):
    """
    Generates detailed error message with script name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return f"Error occurred in script [{file_name}] at line [{line_number}] → {str(error)}"


class CustomException(Exception):
    """
    Custom Exception class for better debugging and error tracking.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

        # Log the error automatically
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
