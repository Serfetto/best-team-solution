"""Utility helpers for the parsing service."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%d.%m.%Y %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d",
    "%d.%m.%Y",
)

STANDARD_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        dt = None
        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def format_datetime(value: datetime | str | None) -> str | None:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = parse_date(value) if value else None
    if dt is None:
        return None
    return dt.replace(microsecond=0).strftime(STANDARD_DATETIME_FORMAT)


def ensure_csv(path: Path, fieldnames: Sequence[str]) -> bool:
    """Ensure CSV file exists with header.

    Returns True if the file already existed, False if it was created.
    """
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
    return False


def append_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> int:
    ensure_csv(path, fieldnames)
    count = 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count
