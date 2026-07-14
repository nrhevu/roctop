from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator


_PROFILE_ENABLED = os.environ.get("ROCTOP_PROFILE", "").lower() not in ("", "0", "false", "no")


@contextmanager
def profile_span(name: str) -> Iterator[None]:
    if not _PROFILE_ENABLED:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            print(f"roctop profile {name}: {elapsed_ms:.2f}ms", file=sys.stderr)
        except (OSError, ValueError):
            pass
