from pathlib import Path


def write_output(data: str, path: str) -> int:
    """Writes string data to a file, creating parent directories if needed.

    Args:
        data: The string content to write to the file.
        path: The destination file path.

    Returns:
        The number of characters written to the file.

    Raises:
        RuntimeError: If the write operation fails for any reason.

    Side Effects:
        - Creates parent directories if they do not exist.
        - Overwrites the file if it already exists.
    """
    file = Path(path)
    try:
        file.parent.mkdir(exist_ok=True, parents=True)
        return file.write_text(data, encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Writing output operation failed: {e}")
