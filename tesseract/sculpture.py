"""Famous trees as 4D sculptures.

Ambient solids live in `tesseract.codeblock`. This module clones a gallery
repository when someone wants the real history instead of the stylized block.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from tesseract.codeblock import SCULPTURES, ambient_sculpture, hold_sculpture, list_sculptures
from tesseract.life import repo_root

GALLERY = {
    "linux": {"url": "https://github.com/torvalds/linux.git", "include": ("kernel/sched",)},
    "doom": {"url": "https://github.com/id-Software/DOOM.git", "include": ()},
    "quake": {"url": "https://github.com/id-Software/Quake.git", "include": ()},
    "git": {"url": "https://github.com/git/git.git", "include": ()},
    "cpython": {"url": "https://github.com/python/cpython.git", "include": ("Python",)},
    "redis": {"url": "https://github.com/redis/redis.git", "include": ()},
    "sqlite": {"url": "https://github.com/sqlite/sqlite.git", "include": ()},
}


def gallery_dir() -> Path:
    return repo_root() / ".gallery"


def fetch(name: str, *, depth: int = 200) -> Path:
    """Clone a gallery tree into `.gallery/<name>`, or return it if already present."""
    key = name.strip().lower()
    if key not in GALLERY:
        raise ValueError(f"unknown gallery sculpture {name!r}; choose from {', '.join(GALLERY)}")
    dest = gallery_dir() / key
    if (dest / ".git").is_dir():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    spec = GALLERY[key]
    subprocess.run(
        ["git", "clone", "--depth", str(depth), spec["url"], str(dest)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return dest


def include_for(name: str) -> tuple[str, ...]:
    spec = GALLERY.get(name.strip().lower(), {})
    return tuple(spec.get("include") or ())


__all__ = [
    "GALLERY",
    "SCULPTURES",
    "ambient_sculpture",
    "fetch",
    "gallery_dir",
    "hold_sculpture",
    "include_for",
    "list_sculptures",
]
