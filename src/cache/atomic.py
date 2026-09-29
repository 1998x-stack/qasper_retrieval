"""Atomic same-filesystem persistence helpers."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Union


PathLike = Union[str, Path]


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory after a rename."""
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(str(directory), flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_with(path: PathLike, writer: Callable[[Path], None]) -> None:
    """Write through a temporary sibling and atomically replace the target.

    The caller writes a complete artifact to the provided temporary path.
    If the writer fails, the previous target remains untouched.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        _fsync_directory(target.parent)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Atomically replace a file with bytes."""
    def _write(temporary: Path) -> None:
        temporary.write_bytes(data)

    atomic_write_with(path, _write)


def atomic_write_text(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    """Atomically replace a text file."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: PathLike, data: Any, indent: int = 2) -> None:
    """Atomically write JSON; suitable for manifest-last logical commits."""
    payload = json.dumps(
        data, ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False
    )
    atomic_write_text(path, payload + "\n")
