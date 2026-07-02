from __future__ import annotations

import math


def clamp_percent(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(100.0, max(0.0, number))


def parse_number(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else default
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return default
    try:
        number = float(text.replace("%", ""))
    except (ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def parse_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return int(value)
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return default
    try:
        number = float(text.replace(",", ""))
    except (ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return int(number)


def bytes_to_mib(value: int | float | None) -> float:
    if not value:
        return 0.0
    return float(value) / 1024.0 / 1024.0


def format_bytes_mib(value: int | float | None) -> str:
    mib = bytes_to_mib(value)
    if mib >= 1024 * 10:
        return f"{mib / 1024.0:.1f}GiB"
    if mib >= 1024:
        return f"{mib / 1024.0:.2f}GiB"
    return f"{mib:.0f}MiB"


def percent_text(value: float | int | None, digits: int = 0) -> str:
    value = clamp_percent(value)
    if digits <= 0:
        return f"{value:.0f}%"
    return f"{value:.{digits}f}%"
