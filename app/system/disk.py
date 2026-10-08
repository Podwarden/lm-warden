import shutil
from pathlib import Path


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free
