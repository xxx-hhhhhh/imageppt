"""Rollback published page artifacts if a verified candidate cannot be saved."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def preserve_published_state(root: Path, filenames: list[str]):
    directory = root.resolve()
    targets = [root / name for name in filenames]
    if any(not path.resolve().is_relative_to(directory) for path in targets):
        raise ValueError("Publication target escaped project directory")
    previous = {path: path.read_bytes() if path.is_file() else None for path in targets}
    try:
        yield
    except Exception:
        for path, contents in previous.items():
            if contents is None:
                if path.is_file():
                    path.unlink()  # Only a new artifact created by this failed publication.
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                staging = path.with_name(path.name + ".rollback_" + uuid4().hex)
                staging.write_bytes(contents)
                staging.replace(path)
        raise
