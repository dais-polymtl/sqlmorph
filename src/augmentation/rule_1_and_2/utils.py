import os

def ensure_directory(path):
    """
    Ensure that a directory exists; if not, create it.

    Args:
        path (str): The directory path to ensure.
    """
    os.makedirs(path, exist_ok=True)
