import os
import sys
from loguru import logger
from datetime import datetime


def setup_logging(logger, log_level: str) -> None:
    logger.remove()  # Remove default logger
    logger.add(sink=sys.stdout, level=log_level)  # Set new logging level


def get_current_timestamp():
    """Get the current timestamp in format yyyy-mm-dd hh:mm:ss.

    Returns:
        str: The current timestamp as a string in "YYYY-MM-DD HH:MM:SS" format.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def convert_to_binary(filename: str) -> bytes:
    """Convert a file to binary data.

    Args:
        filename (str): Path to the file to be converted.

    Returns:
        bytes: Binary data of the file.
    """
    with open(filename, "rb") as file:
        data = file.read()
    return data


def unix_timestamp_to_timestamp(unix_timestamp: int) -> str:
    """Convert a Unix timestamp to a standard timestamp.

    Args:
        unix_timestamp (int): The Unix timestamp to convert.

    Returns:
        str: The formatted timestamp as a string in "YYYY-MM-DD HH:MM:SS" format.
    """
    return datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d %H:%M:%S")


def extract_metadata(filename: str) -> dict:
    """Extract metadata from a file.

    Returns a dictionary with the following keys:
    - filename: Name of the file
    - size: Size of the file in bytes
    - created_at: Creation time of the file in standard timestamp format
    - type: File extension (without the dot)
    """
    metadata = {}
    metadata["filename"] = filename
    metadata["size"] = os.path.getsize(filename)  # Size in bytes
    metadata["created_at"] = unix_timestamp_to_timestamp(
        os.path.getctime(filename)
    )  # Creation time
    metadata["type"] = os.path.splitext(filename)[1][
        1:
    ]  # Get file extension without dot
    return metadata
