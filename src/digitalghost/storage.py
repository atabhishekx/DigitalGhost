"""Crash-conscious local file persistence. No SQL database is used."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()


class StorageError(RuntimeError):
    """Raised when local state cannot be safely read or written."""


def _atomic_replace_bytes(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        # Best-effort directory sync makes the rename durable on POSIX filesystems.
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_bytes(path: str | Path, content: bytes, *, backup: bool = False) -> None:
    """Atomically write exact bytes, optionally preserving the previous file as .bak."""
    target = Path(path)
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    with _LOCK:
        try:
            if backup and target.exists():
                _atomic_replace_bytes(target.with_suffix(target.suffix + ".bak"), target.read_bytes())
            _atomic_replace_bytes(target, content)
        except OSError as exc:
            raise StorageError(f"Could not safely write {target}: {exc}") from exc


def atomic_write_text(path: str | Path, content: str, *, backup: bool = False) -> None:
    if not isinstance(content, str):
        raise TypeError("content must be str")
    atomic_write_bytes(path, content.encode("utf-8"), backup=backup)


def write_json(path: str | Path, data: Any, *, backup: bool = False) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", backup=backup)


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    try:
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        backup = target.with_suffix(target.suffix + ".bak")
        if backup.exists():
            try:
                with backup.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                pass
        raise StorageError(f"Could not read valid JSON from {target}: {exc}") from exc


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True) + "\n"
    with _LOCK:
        try:
            with target.open("a", encoding="utf-8", newline="") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise StorageError(f"Could not append to {target}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    records = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except json.JSONDecodeError as exc:
                raise StorageError(f"Invalid JSONL at {target}:{line_no}: {exc}") from exc
    return records
