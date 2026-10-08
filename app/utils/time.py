from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def minute_bucket(dt: datetime | None = None) -> int:
    """Return integer minute bucket (epoch seconds // 60)."""
    if dt is None:
        dt = utc_now()
    return int(dt.timestamp()) // 60
