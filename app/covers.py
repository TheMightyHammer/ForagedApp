from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


# --- Helpers ---

def _cover_cache_name(covers_dir: Path, cache_key: str, ext: str) -> Path:
    """Generate a stable cache filename for a book based on a cache key."""
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    ext = ext if ext.startswith(".") else f".{ext}"
    return covers_dir / f"{digest}{ext}"


def _read_zip_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def _norm_path(base: str, href: str) -> str:
    # Normalise an OPF href relative to the OPF directory
    if not base:
        return href
    return str(Path(base) / href).replace("\\", "/")


def _guess_ext(content_type: Optional[str], href: Optional[str]) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct == "image/jpeg" or ct == "image/jpg":
            return ".jpg"
        if ct == "image/png":
            return ".png"
        if ct == "image/webp":
            return ".webp"
        if ct == "image/gif":
            return ".gif"
    if href:
        ext = Path(href).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return ".jpg" if ext == ".jpeg" else ext
        guessed = mimetypes.guess_extension(mimetypes.guess_type(href)[0] or "")
        if guessed:
            return guessed
    return ".jpg"


# --- EPUB cover extraction (stdlib only, no Pillow/ebooklib) ---

def extract_epub_cover(epub_path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    """Extract cover bytes and preferred extension from an EPUB.

    Returns: (bytes_or_None, ext_or_None)

    Strategy:
    1) Read META-INF/container.xml to find the OPF package document
    2) Parse OPF manifest/spine metadata
       - EPUB3: manifest item with properties='cover-image'
       - EPUB2: <meta name='cover' content='id'> then manifest item by that id
    3) Heuristic fallback: pick the largest image file in the archive that looks like a cover
    """
    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            names = set(zf.namelist())

            container_path = "META-INF/container.xml"
            if container_path not in names:
                return (None, None)

            container_xml = _read_zip_text(zf, container_path)
            try:
                croot = ET.fromstring(container_xml)
            except ET.ParseError:
                return (None, None)

            # container.xml namespaces vary, so search without hardcoding
            opf_full_path = None
            for elem in croot.iter():
                if elem.tag.endswith("rootfile"):
                    opf_full_path = elem.attrib.get("full-path") or elem.attrib.get("fullpath")
                    if opf_full_path:
                        break

            if not opf_full_path or opf_full_path not in names:
                # Some malformed EPUBs. Fall back to heuristic.
                return _heuristic_cover_from_zip(zf)

            opf_dir = str(Path(opf_full_path).parent).replace("\\", "/")
            if opf_dir == ".":
                opf_dir = ""

            opf_xml = _read_zip_text(zf, opf_full_path)
            try:
                proot = ET.fromstring(opf_xml)
            except ET.ParseError:
                return _heuristic_cover_from_zip(zf)

            # Build manifest: id -> (href, media-type, properties)
            manifest: dict[str, tuple[str, Optional[str], str]] = {}
            for elem in proot.iter():
                if elem.tag.endswith("item"):
                    _id = elem.attrib.get("id")
                    href = elem.attrib.get("href")
                    if not _id or not href:
                        continue
                    media_type = elem.attrib.get("media-type")
                    props = (elem.attrib.get("properties") or "").strip()
                    manifest[_id] = (href, media_type, props)

            # 1) EPUB3 cover-image property
            for _id, (href, media_type, props) in manifest.items():
                if "cover-image" in props.split():
                    rel = _norm_path(opf_dir, href)
                    if rel in names:
                        data = zf.read(rel)
                        return (data, _guess_ext(media_type, href))

            # 2) EPUB2 <meta name="cover" content="id"/>
            cover_id = None
            for elem in proot.iter():
                if elem.tag.endswith("meta"):
                    if (elem.attrib.get("name") or "").lower() == "cover":
                        cover_id = elem.attrib.get("content")
                        break

            if cover_id and cover_id in manifest:
                href, media_type, _props = manifest[cover_id]
                rel = _norm_path(opf_dir, href)
                if rel in names:
                    data = zf.read(rel)
                    return (data, _guess_ext(media_type, href))

            # 3) Heuristic fallback inside the EPUB
            return _heuristic_cover_from_zip(zf)

    except zipfile.BadZipFile:
        return (None, None)
    except Exception as exc:
        logger.warning("Failed extracting EPUB cover for %s: %s", epub_path, exc)
        return (None, None)


def _heuristic_cover_from_zip(zf: zipfile.ZipFile) -> Tuple[Optional[bytes], Optional[str]]:
    """Heuristic: choose the largest image, preferring filenames containing 'cover'."""
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".gif")

    candidates = []
    for info in zf.infolist():
        name = info.filename
        lower = name.lower()
        if lower.endswith(image_exts) and not lower.startswith("__macosx/"):
            # Prefer cover-like paths
            score = 0
            if "cover" in lower:
                score += 100
            if re.search(r"/cover\b", lower):
                score += 50
            if "thumbnail" in lower or "thumb" in lower:
                score -= 25
            # Bigger file usually better
            size = info.file_size
            candidates.append((score, size, name))

    if not candidates:
        return (None, None)

    candidates.sort(reverse=True)
    _score, _size, best_name = candidates[0]

    try:
        data = zf.read(best_name)
        ext = Path(best_name).suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        return (data, ext)
    except Exception:
        return (None, None)


# --- Public API used by the app ---

def get_or_create_epub_cover(
    epub_path: Path,
    covers_dir: Path | str = "/data/covers",
    cache_key: Optional[str] = None,
) -> Optional[Path]:
    """Return a cached cover image path for an EPUB.

    Notes:
    - Uses only stdlib to extract the embedded cover.
    - Saves the extracted image as-is (PNG stays PNG, JPEG stays JPEG, etc.).
    - `covers_dir` should be a mounted volume for persistence.
    """
    covers_dir = Path(covers_dir)
    try:
        covers_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    key = cache_key or str(epub_path)

    raw_bytes, ext = extract_epub_cover(epub_path)
    if not raw_bytes:
        return None

    ext = ext or ".jpg"
    cover_path = _cover_cache_name(covers_dir, key, ext=ext)

    if cover_path.exists():
        return cover_path

    try:
        cover_path.write_bytes(raw_bytes)
        return cover_path
    except Exception:
        return None


# Backwards compatibility: older code may call get_or_create_epub_cover(epub_path, rel_path)

def get_or_create_epub_cover_legacy(epub_path: Path, rel_path: str) -> Optional[Path]:
    return get_or_create_epub_cover(epub_path, covers_dir="/data/covers", cache_key=rel_path)