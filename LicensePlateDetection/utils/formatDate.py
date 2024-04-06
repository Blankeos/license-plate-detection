
import datetime

def formatDate(date: datetime.datetime | str) -> str:
    """
    Format a datetime object into a human-readable string.

    Args:
        date (datetime.datetime): The date to format.

    Returns:
        str: The formatted date.
    """
    if isinstance(date, str):
        date = datetime.datetime.fromisoformat(date)
    return date.strftime("%B %d, %Y %I:%M %p")