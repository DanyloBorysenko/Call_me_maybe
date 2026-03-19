from pathlib import Path


class OutputWriterError(Exception):
    pass


def write_output(data: str, path: str) -> int:
    try:
        file = Path(path)
        file.parent.mkdir(exist_ok=True, parents=True)
        return file.write_text(data)
    except Exception as e:
        raise OutputWriterError(f"Writing output operation failed: {e}")
