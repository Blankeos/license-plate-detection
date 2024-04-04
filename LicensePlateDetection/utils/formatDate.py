
import datetime

"""
Format a datetime object into a human-readable string.

Args:
    date (datetime.datetime): The date to format.

Returns:
    str: The formatted date.
"""
def formatDate(date: datetime.datetime) -> str:
    return date.strftime("%B %d, %Y %H:%M:%S")