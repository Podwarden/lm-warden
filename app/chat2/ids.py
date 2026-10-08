import secrets
import time


def new_id() -> str:
    """Time-ordered id: 13 hex chars of epoch-ms + 16 random hex chars."""
    return f"{int(time.time() * 1000):013x}{secrets.token_hex(8)}"
