from datetime import UTC, datetime


def now_iso() -> str:
    """UTC timestamp, fixed width `YYYY-MM-DDTHH:MM:SS.mmmZ`, so text order == time order."""
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
