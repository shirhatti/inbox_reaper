"""Timezone-aware time helpers.

All timestamps are created, stored and compared as aware UTC datetimes.
Versions before this module wrote naive local-time ISO strings (for example the
token ``expires_at`` kept in the keyring); :func:`parse_timestamp` still reads
those, interpreting them as local time, so existing credentials keep working.
"""

from datetime import UTC, datetime, timedelta


def utcnow() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


def to_utc(value: datetime, *, naive_is_local: bool) -> datetime:
    """Convert ``value`` to aware UTC.

    Naive values are taken as local time when ``naive_is_local`` is true (how
    older versions wrote timestamps), otherwise as UTC (e.g. RFC 5322 "-0000").
    """
    if value.tzinfo is None:
        value = value.astimezone() if naive_is_local else value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_timestamp(value: str | float) -> datetime:
    """Parse a stored timestamp into an aware UTC datetime.

    Accepts ISO 8601 strings (with or without an offset) and Unix timestamps.

    Raises:
        ValueError: If a string is not valid ISO 8601
        TypeError: If ``value`` is neither a string nor a number
    """
    if isinstance(value, bool):
        raise TypeError(f"Not a timestamp: {value!r}")
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, UTC)
    if not isinstance(value, str):
        raise TypeError(f"Not a timestamp: {value!r}")
    return to_utc(datetime.fromisoformat(value), naive_is_local=True)


def expiry_after(seconds: float) -> str:
    """ISO 8601 UTC timestamp ``seconds`` from now, for storing token expiry."""
    return (utcnow() + timedelta(seconds=seconds)).isoformat()
