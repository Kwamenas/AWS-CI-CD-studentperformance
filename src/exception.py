import traceback
from src.logger import logging

def format_exception(error:Exception)-> str:
    """
    Return a nicely formatted string with details about an exception.

    Parameters
    ----------
    error : Exception
        The exception object.

    Returns
    -------
    str
        A formatted message with exception type, message, and traceback.
    """
    error_type=type(error).__name__
    error_message=str(error)
    error_traceback="".join(traceback.format_tb(error.__traceback__))
    

    formated=(
        f"Exception type: {error_type}\n"
        f"Message: {error_message}\n"
        f"Traceback: {error_traceback}\n"

    )

    logging.error(formated)

    return formated


class CustomException(Exception):
    """
      A simple custom exception class that wraps standard exceptions
      with a detailed message.
    """

    def __init__(self, error:Exception):
        super().__init__(str(error))
        self.details=format_exception(error)
    
    def __str__(self):
        return self.details

    
