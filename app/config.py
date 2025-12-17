"""Application configuration utilities."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


def get_env(name: str, default: str) -> str:
    """Read an env var with backwards-compatible fallbacks."""
    candidates = [name]
    if name.startswith("BAYLEAF_"):
        suffix = name[len("BAYLEAF_") :]
        candidates.extend(
            [
                f"Bayleaf_{suffix}",
                f"FORAGED_{suffix}",
                f"Foraged_{suffix}",
            ]
        )

    for key in candidates:
        raw = os.getenv(key)
        if raw is None:
            continue
        value = raw.strip()
        return value or default

    return default


def get_db_path() -> str:
    return get_env("BAYLEAF_DB_PATH", "/data/bayleaf.db")


def get_library_dir() -> str:
    return get_env("BAYLEAF_LIBRARY_DIR", "/cookbooks")


@dataclass(frozen=True)
class Settings:
    library_dir: Path
    env: str = "dev"
    db_path: Path | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    library_dir = Path(get_library_dir()).expanduser()
    env = get_env("BAYLEAF_ENV", "dev") or "dev"
    db_path = Path(get_db_path()).expanduser()
    return Settings(library_dir=library_dir, env=env, db_path=db_path)
