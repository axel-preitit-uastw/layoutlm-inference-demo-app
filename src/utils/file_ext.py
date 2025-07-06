import os


def get_file_ext(file_path: str) -> str:
    """
    Get the file extension from a file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The file extension, or an empty string if no extension is found.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext[1:]  # Remove the leading dot
    return ext.lower() if ext else ''
