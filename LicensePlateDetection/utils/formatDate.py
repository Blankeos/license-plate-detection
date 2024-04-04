
import datetime

def formatDate(date: datetime.datetime) -> str:
    """
    Format a datetime object into a human-readable string.

    Args:
        date (datetime.datetime): The date to format.

    Returns:
        str: The formatted date.
    """
    return date.strftime("%B %d, %Y %H:%M:%S")