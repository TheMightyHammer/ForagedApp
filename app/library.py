"""Library utilities for finding and filtering cookbooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ALLOWED_SUFFIXES = {".pdf", ".epub"}


@dataclass(frozen=True)
class Cookbook:
    name: str
    path: Path
    rel_path: str
    suffix: str
    size: int
    mtime: int

    @property
    def abs_path(self) -> Path:
        return self.path

    @property
    def file_name(self) -> str:
        return self.path.name


def list_cookbooks(root: Path | str) -> List[Cookbook]:
    """Recursively list cookbooks under root with basic metadata."""
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []

    items: List[Cookbook] = []
    for p in root_path.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            continue
        stat = p.stat()
        try:
            rel_path = str(p.relative_to(root_path))
        except ValueError:
            rel_path = str(p)
        items.append(
            Cookbook(
                name=p.stem,
                path=p,
                rel_path=rel_path,
                suffix=suffix,
                size=stat.st_size,
                mtime=int(stat.st_mtime),
            )
        )
    items.sort(key=lambda c: c.name.lower())
    return items


def filter_cookbooks(cookbooks: Iterable[Cookbook], q: Optional[str]) -> List[Cookbook]:
    """Filter cookbooks by a simple case-insensitive substring match on title."""
    if not q:
        return list(cookbooks)
    qn = q.strip().lower()
    return [c for c in cookbooks if qn in c.name.lower()]
