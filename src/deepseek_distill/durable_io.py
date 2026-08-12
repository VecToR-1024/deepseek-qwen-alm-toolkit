"""Small, bounded retries for transient Windows file-sharing locks."""

from __future__ import annotations

import errno
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, TypeVar


_TRANSIENT_ERRNOS = {
    errno.EACCES,
    errno.EAGAIN,
    errno.EBUSY,
}
_TRANSIENT_WINDOWS_ERRORS = {5, 32, 33}
_T = TypeVar("_T")


def open_text_append(path: Path) -> IO[str]:
    """Open one append handle, retrying only a transient failure to open it.

    Retrying the open is safe: no bytes have been written yet. The subsequent
    write is deliberately not retried because replaying a partially completed
    append could duplicate a JSONL record.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _retry_file_lock(lambda: path.open("a", encoding="utf-8", newline="\n"))


def replace_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    """Atomically replace a file after bounded transient-lock retries."""
    _retry_file_lock(lambda: os.replace(source, destination))


def _retry_file_lock(
    operation: Callable[[], _T],
    *,
    attempts: int = 10,
    initial_delay_seconds: float = 0.05,
    maximum_delay_seconds: float = 0.5,
) -> _T:
    delay = initial_delay_seconds
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OSError as error:
            if attempt == attempts or not _is_transient_file_lock(error):
                raise
            time.sleep(delay)
            delay = min(delay * 2, maximum_delay_seconds)
    raise AssertionError("retry loop must return or raise")


def _is_transient_file_lock(error: OSError) -> bool:
    return (
        isinstance(error, PermissionError)
        or error.errno in _TRANSIENT_ERRNOS
        or getattr(error, "winerror", None) in _TRANSIENT_WINDOWS_ERRORS
    )
