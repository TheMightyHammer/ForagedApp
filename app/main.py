"""Bayleaf: FastAPI entrypoint.

Milestone: cookbook library view that lists local EPUB and PDF files.

Next milestone: SQLite-backed indexing for fast library plus recipe metadata.
"""

import os
import sqlite3
import hashlib
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # Pillow is optional in dev
    Image = ImageDraw = ImageFont = None  # type: ignore

import logging

from urllib.parse import quote
from urllib.parse import unquote

logger = logging.getLogger("bayleaf")

from app.config import get_env, get_settings
from app.library import ALLOWED_SUFFIXES, list_cookbooks
from app.recipes import extract_epub_recipes




APP_NAME = "Bayleaf"

# Minimum characters before we apply a text filter (helps live-search UX)
DEFAULT_MIN_SEARCH_CHARS = 3

class ProgressIn(BaseModel):
    cfi: str

def _min_search_chars() -> int:
    try:
        return max(0, int(get_env("BAYLEAF_MIN_SEARCH_CHARS", str(DEFAULT_MIN_SEARCH_CHARS))))
    except Exception:
        return DEFAULT_MIN_SEARCH_CHARS


def _normalise_search_query(q: str | None) -> str:
    """Return a normalised query string.

    If the query is shorter than the configured minimum, treat it as empty.
    """
    qn = (q or "").strip()
    if len(qn) < _min_search_chars():
        return ""
    return qn


# --- EPUB metadata helpers (no external deps) ---

def _epub_metadata(epub_path: Path) -> dict:
    """Extract common metadata from an EPUB.

    Best-effort only. Returns a dict with keys:
    title, author, isbn, publisher, published_year, language, description
    """

    def _parse_year(date_str: str | None) -> int | None:
        if not date_str:
            return None
        s = (date_str or "").strip()
        # Common forms: YYYY, YYYY-MM-DD, 2019-01-01T00:00:00Z
        digits = "".join(ch for ch in s[:4] if ch.isdigit())
        if len(digits) == 4:
            try:
                y = int(digits)
                if 1000 <= y <= 3000:
                    return y
            except Exception:
                return None
        return None

    out = {
        "title": None,
        "author": None,
        "isbn": None,
        "publisher": None,
        "published_year": None,
        "language": None,
        "description": None,
    }

    try:
        with zipfile.ZipFile(epub_path, "r") as zf:
            # 1) Locate the OPF via META-INF/container.xml
            try:
                container_xml = zf.read("META-INF/container.xml")
            except KeyError:
                return out

            try:
                c_root = ET.fromstring(container_xml)
            except Exception:
                return out

            # Namespaces vary. Use wildcard namespace.
            opf_path = None
            for el in c_root.findall(".//{*}rootfile"):
                full_path = el.attrib.get("full-path") or el.attrib.get("full_path")
                if full_path:
                    opf_path = full_path
                    break
            if not opf_path:
                return out

            try:
                opf_bytes = zf.read(opf_path)
            except KeyError:
                return out

            try:
                o_root = ET.fromstring(opf_bytes)
            except Exception:
                return out

            def _first_text(paths: list[str]) -> str | None:
                for xpath in paths:
                    node = o_root.find(xpath)
                    if node is None:
                        continue
                    txt = (node.text or "").strip()
                    if txt:
                        return txt
                return None

            # Prefer Dublin Core metadata. Wildcard namespaces handle dc:*
            title = _first_text([
                ".//{*}metadata/{*}title",
                ".//{*}title",
            ])

            # There can be multiple creators. Join distinct values.
            creators: list[str] = []
            for c in (o_root.findall(".//{*}metadata/{*}creator") + o_root.findall(".//{*}creator")):
                t = (c.text or "").strip()
                if t and t not in creators:
                    creators.append(t)
            author = ", ".join(creators) if creators else None

            publisher = _first_text([
                ".//{*}metadata/{*}publisher",
                ".//{*}publisher",
            ])

            language = _first_text([
                ".//{*}metadata/{*}language",
                ".//{*}language",
            ])

            description = _first_text([
                ".//{*}metadata/{*}description",
                ".//{*}description",
            ])

            date_str = _first_text([
                ".//{*}metadata/{*}date",
                ".//{*}date",
            ])
            published_year = _parse_year(date_str)

            # ISBN is not always present. Often stored as dc:identifier with an ISBN-ish value.
            isbn = None
            identifiers: list[str] = []
            for i in (o_root.findall(".//{*}metadata/{*}identifier") + o_root.findall(".//{*}identifier")):
                t = (i.text or "").strip()
                if t:
                    identifiers.append(t)
            for ident in identifiers:
                norm = ident.replace("ISBN", "").replace("isbn", "").replace(":", "").strip()
                digits = "".join(ch for ch in norm if ch.isdigit() or ch in {"X", "x"})
                if len(digits) in {10, 13}:
                    isbn = digits.upper()
                    break

            out["title"] = title
            out["author"] = author
            out["isbn"] = isbn
            out["publisher"] = publisher
            out["published_year"] = published_year
            out["language"] = language
            out["description"] = description

            return out
    except Exception:
        return out


def _derive_title_author_from_filename(file_name: str) -> tuple[str, str]:
    """Fallback title/author from filename: 'Author - Title.ext'."""
    stem = Path(file_name).stem
    parts = [p.strip() for p in stem.split(" - ", 1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return (parts[1], parts[0])
    return (stem, "")

# --- Cover thumbnail helpers (Phase 1: placeholders, cached on disk) ---

def _covers_dir() -> Path:
    """Directory where generated cover thumbnails are cached."""
    # Allow override, default to a persistent path mounted via the /data volume.
    p = Path(get_env("BAYLEAF_COVERS_DIR", "/data/covers"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cover_cache_key(rel_path: str, mtime: int) -> str:
    """Stable cache key for a book. Includes mtime so edits regenerate the cover."""
    return hashlib.sha1(f"{rel_path}|{mtime}".encode("utf-8")).hexdigest()


def _title_from_filename(file_name: str) -> str:
    """Best-effort title extraction from file name.

    Many files are in the format: 'Author - Title.ext'.
    """
    stem = Path(file_name).stem
    parts = [p.strip() for p in stem.split(" - ", 1)]
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return stem


def _make_placeholder_cover(out_path: Path, title: str) -> None:
    # If Pillow isn't available, fall back to a tiny SVG placeholder.
    if Image is None or ImageDraw is None:
        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='600' height='900'>
<rect width='100%' height='100%' fill='#F4F1EC'/>
<rect x='50' y='60' width='500' height='780' fill='none' stroke='#5F7A6A' stroke-width='6'/>
<text x='80' y='110' font-family='sans-serif' font-size='24' fill='#5F7A6A'>Bayleaf</text>
<text x='80' y='180' font-family='sans-serif' font-size='22' fill='#1F2A24'>{(title or 'Untitled').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')}</text>
</svg>"""
        out_path.write_text(svg, encoding="utf-8")
        return

    w, h = 600, 900
    img = Image.new("RGB", (w, h), "#F4F1EC")
    draw = ImageDraw.Draw(img)

    # Use a default font to keep the Docker image simple.
    try:
        font = ImageFont.load_default() if ImageFont is not None else None
    except Exception:
        font = None

    # Border
    draw.rectangle([50, 60, w - 50, h - 60], outline="#5F7A6A", width=6)

    # Header
    draw.text((80, 90), "Bayleaf", fill="#5F7A6A", font=font)

    # Wrap title lines crudely by character count.
    text = (title or "Untitled").strip()
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        if len(test) > 26 and line:
            lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)
    lines = lines[:10]

    y = 170
    for ln in lines:
        draw.text((80, y), ln, fill="#1F2A24", font=font)
        y += 45

    img.save(out_path, "PNG", optimize=True)


def _attach_cover_urls(app: FastAPI, books: list[dict]) -> None:
    """Attach a cover URL to each book dict for templates."""
    for b in books:
        rel_path = (b.get("rel_path") or "").strip()
        if not rel_path:
            b["cover_url"] = ""
            continue
        try:
            # url_path_for gives a relative URL, which is ideal for reverse proxies.
            b["cover_url"] = str(app.url_path_for("cover", rel_path=rel_path))
        except Exception:
            b["cover_url"] = f"/cover/{rel_path}"


# Helper to normalise cookbook entries (dicts or objects) to dicts for templates/indexer
def _cookbook_to_dict(b) -> dict:
    """Normalise cookbook entries (dicts or objects) to template-friendly dicts.

    Important: some implementations of `list_cookbooks()` may return objects that don't
    expose `rel_path` but do expose an absolute `path` (or `file_path`).

    We carry both `rel_path` (preferred) and `abs_path` (fallback) so callers can
    derive a stable rel_path for DB keys and URLs.
    """

    if isinstance(b, dict):
        rel_path = b.get("rel_path")
        abs_path = b.get("abs_path") or b.get("path") or b.get("file_path")
        name = b.get("name")
        file_name = b.get("file_name")
        suffix = b.get("suffix")
        size = b.get("size")
        mtime = b.get("mtime")
    else:
        rel_path = getattr(b, "rel_path", None)
        abs_path = getattr(b, "abs_path", None) or getattr(b, "path", None) or getattr(b, "file_path", None)
        name = getattr(b, "name", None)
        file_name = getattr(b, "file_name", None)
        suffix = getattr(b, "suffix", None)
        size = getattr(b, "size", None)
        mtime = getattr(b, "mtime", None)

    rel_path_str = str(rel_path) if rel_path else ""
    abs_path_str = str(abs_path) if abs_path else ""
    candidate_path = rel_path_str or abs_path_str
    candidate_name = Path(candidate_path).name if candidate_path else ""

    file_name_str = str(file_name).strip() if file_name else ""
    if not file_name_str:
        file_name_str = candidate_name

    # Prefer provided display name, otherwise derive from the file name.
    if name is not None and str(name).strip():
        name_str = str(name).strip()
    else:
        if file_name_str:
            name_str = Path(file_name_str).stem or file_name_str
        elif candidate_path:
            name_str = Path(candidate_path).stem
        else:
            name_str = "Untitled"

    # Prefer suffix if provided. Otherwise infer from the best-known filename.
    if suffix is None or str(suffix).strip() == "":
        suffix_source = file_name_str or candidate_path or ""
        suffix_str = Path(suffix_source).suffix if suffix_source else ""
    else:
        suffix_str = str(suffix)

    return {
        "rel_path": rel_path_str,
        "abs_path": abs_path_str,
        "name": name_str,
        "file_name": file_name_str or name_str,
        "suffix": suffix_str,
        "size": int(size or 0),
        "mtime": int(mtime or 0),
    }



def _db_path() -> str:
    return get_env("BAYLEAF_DB_PATH", "/data/bayleaf.db")


def _connect_db() -> sqlite3.Connection:
    db_path = Path(_db_path())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    # WAL improves read/write concurrency. Safe for our single-user MVP.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")

    # Core tables
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS books (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          rel_path TEXT NOT NULL UNIQUE,
          file_name TEXT NOT NULL,
          file_type TEXT NOT NULL,
          file_size INTEGER NOT NULL,
          modified_mtime INTEGER NOT NULL,

          title TEXT,
          author TEXT,
          publisher TEXT,
          published_year INTEGER,
          language TEXT,
          isbn TEXT,

          cover_rel_path TEXT,
          description TEXT,

          created_at INTEGER NOT NULL DEFAULT (unixepoch()),
          updated_at INTEGER NOT NULL DEFAULT (unixepoch())
        );

        CREATE INDEX IF NOT EXISTS idx_books_file_type ON books(file_type);
        CREATE INDEX IF NOT EXISTS idx_books_author ON books(author);
        CREATE INDEX IF NOT EXISTS idx_books_title ON books(title);

        CREATE TABLE IF NOT EXISTS recipes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          book_id INTEGER NOT NULL,

          title TEXT NOT NULL,
          ingredients_text TEXT,
          method_text TEXT,

          servings TEXT,
          prep_time_minutes INTEGER,
          cook_time_minutes INTEGER,
          total_time_minutes INTEGER,

          -- Uniqueness must be based on where the recipe came from inside the book,
          -- not the title. A single book can contain multiple "Cupcakes" recipes.
          source_type TEXT NOT NULL,
          source_key TEXT NOT NULL,

          location_type TEXT,
          location_value TEXT,
          image_href TEXT,

          created_at INTEGER NOT NULL DEFAULT (unixepoch()),
          updated_at INTEGER NOT NULL DEFAULT (unixepoch()),

          FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE,
          UNIQUE(book_id, source_type, source_key)
        );

        CREATE INDEX IF NOT EXISTS idx_recipes_book_id ON recipes(book_id);
        CREATE INDEX IF NOT EXISTS idx_recipes_title ON recipes(title);

        -- Tags (shared between books and recipes)
        CREATE TABLE IF NOT EXISTS tags (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS book_tags (
          book_id INTEGER NOT NULL,
          tag_id INTEGER NOT NULL,
          PRIMARY KEY (book_id, tag_id),
          FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE,
          FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_book_tags_tag_id ON book_tags(tag_id);

        CREATE TABLE IF NOT EXISTS recipe_tags (
          recipe_id INTEGER NOT NULL,
          tag_id INTEGER NOT NULL,
          PRIMARY KEY (recipe_id, tag_id),
          FOREIGN KEY(recipe_id) REFERENCES recipes(id) ON DELETE CASCADE,
          FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_recipe_tags_tag_id ON recipe_tags(tag_id);

        -- Optional but useful: track indexing runs for debugging.
        CREATE TABLE IF NOT EXISTS indexing_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          started_at INTEGER NOT NULL DEFAULT (unixepoch()),
          finished_at INTEGER,
          indexed_books INTEGER NOT NULL DEFAULT 0,
          indexed_recipes INTEGER NOT NULL DEFAULT 0,
          errors TEXT
        );

        -- Reading progress (EPUB location saved as CFI). One row per book for MVP.
        CREATE TABLE IF NOT EXISTS reading_progress (
          rel_path TEXT PRIMARY KEY,
          cfi TEXT NOT NULL,
          updated_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        """
    )

    _ensure_recipes_columns(conn)

    # Full text search for recipes. Uses FTS5. This is built into modern SQLite.
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts USING fts5(
          title,
          ingredients_text,
          method_text,
          content='recipes',
          content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS recipes_ai AFTER INSERT ON recipes BEGIN
          INSERT INTO recipes_fts(rowid, title, ingredients_text, method_text)
          VALUES (new.id, new.title, new.ingredients_text, new.method_text);
        END;

        CREATE TRIGGER IF NOT EXISTS recipes_ad AFTER DELETE ON recipes BEGIN
          INSERT INTO recipes_fts(recipes_fts, rowid, title, ingredients_text, method_text)
          VALUES ('delete', old.id, old.title, old.ingredients_text, old.method_text);
        END;

        CREATE TRIGGER IF NOT EXISTS recipes_au AFTER UPDATE ON recipes BEGIN
          INSERT INTO recipes_fts(recipes_fts, rowid, title, ingredients_text, method_text)
          VALUES ('delete', old.id, old.title, old.ingredients_text, old.method_text);
          INSERT INTO recipes_fts(rowid, title, ingredients_text, method_text)
          VALUES (new.id, new.title, new.ingredients_text, new.method_text);
        END;
        """
    )

    conn.commit()


def _ensure_recipes_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(recipes)").fetchall()
    }
    if "image_href" not in existing:
        conn.execute("ALTER TABLE recipes ADD COLUMN image_href TEXT")



def _upsert_book(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    file_name: str,
    file_type: str,
    file_size: int,
    modified_mtime: int,
    title: str | None = None,
    author: str | None = None,
    isbn: str | None = None,
    publisher: str | None = None,
    published_year: int | None = None,
    language: str | None = None,
    description: str | None = None,
) -> None:
    # Best-effort metadata. We only overwrite when we have a non-empty value.
    title = (title or "").strip() or None
    author = (author or "").strip() or None
    isbn = (isbn or "").strip() or None

    publisher = (publisher or "").strip() or None
    language = (language or "").strip() or None
    description = (description or "").strip() or None
    # Keep descriptions reasonably sized for MVP (prevents accidental megabytes in DB)
    if description and len(description) > 20000:
        description = description[:20000]

    if published_year is not None:
        try:
            published_year = int(published_year)
        except Exception:
            published_year = None
        if published_year is not None and not (1000 <= published_year <= 3000):
            published_year = None

    conn.execute(
        """
        INSERT INTO books (
          rel_path, file_name, file_type, file_size, modified_mtime,
          title, author, isbn,
          publisher, published_year, language, description,
          updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
        ON CONFLICT(rel_path) DO UPDATE SET
          file_name=excluded.file_name,
          file_type=excluded.file_type,
          file_size=excluded.file_size,
          modified_mtime=excluded.modified_mtime,
          title=COALESCE(excluded.title, books.title),
          author=COALESCE(excluded.author, books.author),
          isbn=COALESCE(excluded.isbn, books.isbn),
          publisher=COALESCE(excluded.publisher, books.publisher),
          published_year=COALESCE(excluded.published_year, books.published_year),
          language=COALESCE(excluded.language, books.language),
          description=COALESCE(excluded.description, books.description),
          updated_at=unixepoch();
        """,
        (
            rel_path,
            file_name,
            file_type,
            file_size,
            modified_mtime,
            title,
            author,
            isbn,
            publisher,
            published_year,
            language,
            description,
        ),
    )


def _book_id_for_rel_path(conn: sqlite3.Connection, rel_path: str) -> int | None:
    row = conn.execute("SELECT id FROM books WHERE rel_path = ?", (rel_path,)).fetchone()
    if row is None:
        return None
    return int(row["id"])


def _count_recipes_for_book(conn: sqlite3.Connection, book_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM recipes WHERE book_id = ?",
        (book_id,),
    ).fetchone()
    return int(row["c"] or 0) if row is not None else 0


def _count_recipes_for_book_source(
    conn: sqlite3.Connection, book_id: int, source_type: str
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM recipes WHERE book_id = ? AND source_type = ?",
        (book_id, source_type),
    ).fetchone()
    return int(row["c"] or 0) if row is not None else 0


def _truncate_text(value: str | None, limit: int) -> str | None:
    if not value:
        return None
    if len(value) <= limit:
        return value
    return value[:limit]


def _upsert_recipe(conn: sqlite3.Connection, book_id: int, recipe: dict) -> None:
    title = (recipe.get("title") or "").strip()
    if not title:
        return

    ingredients_text = _truncate_text(recipe.get("ingredients_text"), 20000)
    method_text = _truncate_text(recipe.get("method_text"), 50000)
    source_type = recipe.get("source_type") or "epub"
    source_key = recipe.get("source_key") or ""
    location_type = recipe.get("location_type")
    location_value = recipe.get("location_value")
    image_href = recipe.get("image_href")

    if not source_key or not method_text:
        return

    conn.execute(
        """
        INSERT INTO recipes (
          book_id,
          title,
          ingredients_text,
          method_text,
          source_type,
          source_key,
          location_type,
          location_value,
          image_href
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(book_id, source_type, source_key) DO UPDATE SET
          title=excluded.title,
          ingredients_text=excluded.ingredients_text,
          method_text=excluded.method_text,
          location_type=excluded.location_type,
          location_value=excluded.location_value,
          image_href=excluded.image_href,
          updated_at=unixepoch()
        """,
        (
            book_id,
            title,
            ingredients_text,
            method_text,
            source_type,
            source_key,
            location_type,
            location_value,
            image_href,
        ),
    )


def _index_recipes(
    conn: sqlite3.Connection,
    library_dir: Path,
    book_dicts: list[dict],
    recipe_book_limit: int,
) -> int:
    root = Path(library_dir)
    candidates = []
    for b in book_dicts:
        rel_path = str(b.get("rel_path") or "")
        suffix_value = b.get("suffix") or Path(rel_path).suffix
        suffix = str(suffix_value or "").lower()
        if suffix != ".epub":
            continue
        candidates.append(b)

    extracted = 0
    processed = 0
    skipped = 0
    limit = None if recipe_book_limit <= 0 else recipe_book_limit
    if candidates:
        logger.info(
            "Recipe extraction starting: epubs=%s limit=%s",
            len(candidates),
            limit if limit is not None else "all",
        )
    for b in candidates[:limit]:
        rel_path = str(b.get("rel_path") or "")
        if not rel_path:
            continue
        book_id = _book_id_for_rel_path(conn, rel_path)
        if book_id is None:
            continue

        abs_path = b.get("abs_path")
        if abs_path:
            epub_path = Path(abs_path)
        else:
            epub_path = (root / rel_path).resolve()

        is_crumbs = "crumbs" in epub_path.name.lower() and "doilies" in epub_path.name.lower()
        if is_crumbs:
            if _count_recipes_for_book_source(conn, book_id, "epub:crumbs_doilies") > 0:
                skipped += 1
                continue
            conn.execute("DELETE FROM recipes WHERE book_id = ?", (book_id,))
        else:
            if _count_recipes_for_book(conn, book_id) > 0:
                skipped += 1
                continue

        processed += 1
        try:
            recipes = extract_epub_recipes(epub_path, max_recipes=200)
        except Exception as exc:
            logger.warning("Recipe extraction failed for %s: %s", rel_path, exc)
            continue

        if not recipes:
            continue

        with conn:
            for recipe in recipes:
                _upsert_recipe(conn, book_id, recipe)

        extracted += len(recipes)

    if candidates:
        logger.info(
            "Recipe extraction complete: processed=%s skipped=%s recipes=%s",
            processed,
            skipped,
            extracted,
        )

    return extracted


def _reextract_recipes_for_book(
    conn: sqlite3.Connection,
    library_dir: Path,
    rel_path: str,
    *,
    max_recipes: int = 200,
) -> dict:
    rel_path = (rel_path or "").strip()
    if not rel_path:
        return {"error": "Missing book path", "extracted": 0}

    book_id = _book_id_for_rel_path(conn, rel_path)
    if book_id is None:
        return {"error": "Book not indexed", "extracted": 0}

    file_path = _safe_resolve(library_dir, rel_path)
    if file_path.suffix.lower() != ".epub":
        return {"error": "Only EPUB recipes can be re-extracted", "extracted": 0}

    logger.info("Recipe re-extract starting: %s", rel_path)
    recipes = extract_epub_recipes(file_path, max_recipes=max_recipes)

    with conn:
        conn.execute("DELETE FROM recipes WHERE book_id = ?", (book_id,))
        for recipe in recipes:
            _upsert_recipe(conn, book_id, recipe)

    logger.info("Recipe re-extract complete: %s recipes=%s", rel_path, len(recipes))
    return {"rel_path": rel_path, "extracted": len(recipes)}


def _index_books(conn: sqlite3.Connection, library_dir: Path | str) -> tuple[int, int]:
    """Scan the library folder and sync the books table.

    This is intentionally "best effort". If a book fails to index, the app should still run.
    """

    if isinstance(library_dir, str):
        library_dir = Path(library_dir)
    if not library_dir.exists():
        return 0, 0

    books = list_cookbooks(library_dir)
    book_dicts = [_cookbook_to_dict(b) for b in books]
    # Ensure every book has a stable rel_path. This is critical because `books.rel_path` is UNIQUE.
    # If rel_path is empty, every upsert collapses into a single row.
    root = Path(library_dir)
    for bd in book_dicts:
        if bd.get("rel_path"):
            continue

        abs_path = bd.get("abs_path") or ""
        if abs_path:
            try:
                bd["rel_path"] = str(Path(abs_path).resolve().relative_to(root.resolve()))
            except Exception:
                # If we can't safely relativise, fall back to file name (still unique-ish)
                bd["rel_path"] = Path(abs_path).name
        else:
            # Last resort. At least avoid empty string.
            bd["rel_path"] = bd.get("name") or "unknown"

    # Defensive: if a previous request left the connection mid-transaction,
    # don't try to start a nested transaction.
    if getattr(conn, "in_transaction", False):
        try:
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass

    # Use the connection context manager to create a single transaction.
    # This avoids `cannot start a transaction within a transaction`.
    try:
        with conn:
            # Track which rel_paths we see during this index run so we can prune stale DB rows
            # (e.g. when a file is renamed or deleted).
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS _seen_paths (rel_path TEXT PRIMARY KEY)")
            conn.execute("DELETE FROM _seen_paths")
            for b in book_dicts:
                rel_path = str(b["rel_path"])
                # Record the path as seen for this run
                conn.execute(
                    "INSERT OR IGNORE INTO _seen_paths (rel_path) VALUES (?)",
                    (rel_path,),
                )
                file_name = str(b.get("file_name") or Path(rel_path).name)
                suffix_value = b.get("suffix") or Path(file_name).suffix or Path(rel_path).suffix
                suffix = str(suffix_value or "").lower()
                file_type = suffix.lstrip(".") or Path(rel_path).suffix.lower().lstrip(".")
                file_size = int(b.get("size") or 0)
                modified_mtime = int(b.get("mtime") or 0)

                # Metadata enrichment (EPUB only for now)
                title: str | None = None
                author: str | None = None
                isbn: str | None = None
                publisher: str | None = None
                published_year: int | None = None
                language: str | None = None
                description: str | None = None

                abs_path = b.get("abs_path")
                # Resolve absolute path for metadata extraction.
                if abs_path:
                    try:
                        abs_p = Path(abs_path)
                    except Exception:
                        abs_p = None
                else:
                    abs_p = None

                # If we don't have abs_path, reconstruct it from library root + rel_path.
                if abs_p is None:
                    try:
                        abs_p = (root / rel_path).resolve()
                    except Exception:
                        abs_p = None

                if abs_p is not None and abs_p.suffix.lower() == ".epub":
                    md = _epub_metadata(abs_p)
                    title = md.get("title")
                    author = md.get("author")
                    isbn = md.get("isbn")
                    publisher = md.get("publisher")
                    published_year = md.get("published_year")
                    language = md.get("language")
                    description = md.get("description")

                # Fallback to filename parsing when metadata missing.
                if not title or not author:
                    t2, a2 = _derive_title_author_from_filename(file_name)
                    if not title:
                        title = t2
                    if not author and a2:
                        author = a2

                _upsert_book(
                    conn,
                    rel_path=rel_path,
                    file_name=file_name,
                    file_type=file_type,
                    file_size=file_size,
                    modified_mtime=modified_mtime,
                    title=title,
                    author=author,
                    isbn=isbn,
                    publisher=publisher,
                    published_year=published_year,
                    language=language,
                    description=description,
                )
            # Remove DB rows for files that no longer exist in the library folder.
            # This prevents duplicates after renames and clears out deleted books.
            conn.execute(
                """
                DELETE FROM books
                WHERE rel_path NOT IN (SELECT rel_path FROM _seen_paths)
                """
            )
    except Exception:
        # Ensure we never leave the connection in a transaction.
        try:
            conn.rollback()
        except Exception:
            pass
        raise

    recipes_indexed = 0
    try:
        settings = get_settings()
        recipes_indexed = _index_recipes(
            conn, root, book_dicts, settings.recipe_index_limit
        )
    except Exception as exc:
        logger.warning("Recipe indexing skipped: %s", exc)

    return len(book_dicts), recipes_indexed


def _query_books(conn: sqlite3.Connection, q: str | None, sort: str | None = "title") -> tuple[list[dict], int]:
    qn = _normalise_search_query(q)
    sort_key = (sort or "title").strip().lower()

    # Sorting options.
    if sort_key in {"mtime", "modified", "recent"}:
        order_by = "modified_mtime DESC, file_name COLLATE NOCASE ASC"
    elif sort_key in {"author"}:
        order_by = "coalesce(author, '') COLLATE NOCASE ASC, coalesce(title, file_name) COLLATE NOCASE ASC"
    elif sort_key in {"year", "published", "publication", "published_year"}:
        # Put unknown years last.
        order_by = "coalesce(published_year, 9999) ASC, coalesce(title, file_name) COLLATE NOCASE ASC"
    elif sort_key in {"file", "filename", "name"}:
        order_by = "file_name COLLATE NOCASE ASC"
    else:
        # title
        order_by = "coalesce(title, file_name) COLLATE NOCASE ASC"

    if qn:
        like = f"%{qn.lower()}%"
        rows = conn.execute(
            f"""
            SELECT rel_path, file_name, file_type, file_size, modified_mtime, title, author, published_year
            FROM books
            WHERE lower(coalesce(title, '')) LIKE ? OR lower(coalesce(author, '')) LIKE ? OR lower(file_name) LIKE ?
            ORDER BY {order_by}
            """,
            (like, like, like),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT rel_path, file_name, file_type, file_size, modified_mtime, title, author, published_year
            FROM books
            ORDER BY {order_by}
            """
        ).fetchall()

    # Shape results to match what templates already expect.
    books: list[dict] = []
    for r in rows:
        rel_path = str(r["rel_path"])
        file_name = str(r["file_name"])
        file_type = str(r["file_type"] or "")
        suffix = f".{file_type}" if file_type else Path(file_name).suffix or Path(rel_path).suffix
        title = (r["title"] or "").strip() if "title" in r.keys() else ""
        author = (r["author"] or "").strip() if "author" in r.keys() else ""
        published_year = r["published_year"] if "published_year" in r.keys() else None
        py = int(published_year) if published_year is not None else None
        display_name = title or (Path(file_name).stem or file_name)
        books.append(
            {
                "rel_path": rel_path,
                "name": display_name,
                "file_name": file_name,
                "suffix": suffix,
                "size": int(r["file_size"]),
                "mtime": int(r["modified_mtime"]),
                "author": author,
                "published_year": py,
            }
        )

    total = conn.execute("SELECT COUNT(*) AS c FROM books").fetchone()["c"]
    return books, int(total)


def _query_recipes(conn: sqlite3.Connection, q: str | None) -> tuple[list[dict], int]:
    qn = _normalise_search_query(q)

    base_select = """
        SELECT
          r.id,
          r.title,
          r.ingredients_text,
          r.method_text,
          r.location_value,
          r.image_href,
          b.rel_path,
          coalesce(b.title, b.file_name) AS book_title
        FROM recipes r
        JOIN books b ON b.id = r.book_id
    """

    if qn:
        rows = conn.execute(
            base_select
            + """
            WHERE r.id IN (
              SELECT rowid FROM recipes_fts WHERE recipes_fts MATCH ?
            )
            ORDER BY r.title COLLATE NOCASE ASC
            """,
            (qn,),
        ).fetchall()
    else:
        rows = conn.execute(
            base_select + " ORDER BY r.title COLLATE NOCASE ASC"
        ).fetchall()

    recipes: list[dict] = []
    for row in rows:
        recipes.append(
            {
                "id": row["id"],
                "title": row["title"],
                "ingredients_text": row["ingredients_text"],
                "method_text": row["method_text"],
                "location_value": row["location_value"],
                "image_href": row["image_href"],
                "rel_path": row["rel_path"],
                "book_title": row["book_title"],
            }
        )

    total = len(recipes)
    return recipes, total


# --- API query helper for live-search ---
def _query_books_api(conn: sqlite3.Connection, q: str | None, sort: str | None) -> list[dict]:
    """Query books for the live-search API.

    Returns a list of dicts including title/author fields when available.
    """

    q = _normalise_search_query(q).lower()
    sort = (sort or "title").strip().lower()

    # Sorting options. Default to title-ish.
    if sort in {"mtime", "modified", "recent"}:
        order_by = "modified_mtime DESC, file_name COLLATE NOCASE ASC"
    elif sort in {"author"}:
        order_by = "coalesce(author, '') COLLATE NOCASE ASC, coalesce(title, file_name) COLLATE NOCASE ASC"
    elif sort in {"year", "published", "publication", "published_year"}:
        # Put unknown years last.
        order_by = "coalesce(published_year, 9999) ASC, coalesce(title, file_name) COLLATE NOCASE ASC"
    elif sort in {"file", "filename", "name"}:
        order_by = "file_name COLLATE NOCASE ASC"
    else:
        # title
        order_by = "coalesce(title, file_name) COLLATE NOCASE ASC"

    if q:
        like = f"%{q}%"
        rows = conn.execute(
            f"""
            SELECT rel_path, file_name, file_type, file_size, modified_mtime, title, author, published_year
            FROM books
            WHERE
              lower(coalesce(title, '')) LIKE ?
              OR lower(coalesce(author, '')) LIKE ?
              OR lower(file_name) LIKE ?
            ORDER BY {order_by}
            """,
            (like, like, like),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT rel_path, file_name, file_type, file_size, modified_mtime, title, author, published_year
            FROM books
            ORDER BY {order_by}
            """
        ).fetchall()

    out: list[dict] = []
    for r in rows:
        rel_path = str(r["rel_path"])
        file_name = str(r["file_name"])
        file_type = str(r["file_type"] or "")
        suffix = f".{file_type}" if file_type else Path(file_name).suffix or Path(rel_path).suffix

        # Prefer DB title if present, otherwise derive from filename.
        title = (r["title"] or "").strip() if "title" in r.keys() else ""
        author = (r["author"] or "").strip() if "author" in r.keys() else ""
        published_year = r["published_year"] if "published_year" in r.keys() else None
        py = int(published_year) if published_year is not None else None

        if not title:
            # Many files are "Author - Title.ext".
            stem = Path(file_name).stem
            parts = [p.strip() for p in stem.split(" - ", 1)]
            if len(parts) == 2:
                if not author:
                    author = parts[0]
                title = parts[1]
            else:
                title = stem

        out.append(
            {
                "filename": rel_path,  # kept as 'filename' for the frontend script
                "rel_path": rel_path,
                "file_name": file_name,
                "file_type": file_type,
                "suffix": suffix,
                "size": int(r["file_size"]),
                "mtime": int(r["modified_mtime"]),
                "title": title,
                "author": author,
                "published_year": py,
            }
        )

    return out


# --- Reading progress helpers (EPUB CFI persistence) ---

def _get_progress_cfi(conn: sqlite3.Connection, rel_path: str) -> str | None:
    try:
        row = conn.execute(
            "SELECT cfi FROM reading_progress WHERE rel_path = ?",
            (rel_path,),
        ).fetchone()
        if row is None:
            return None
        cfi = row[0]
        return str(cfi) if cfi else None
    except Exception:
        return None


def _set_progress_cfi(conn: sqlite3.Connection, rel_path: str, cfi: str) -> None:
    rel_path = (rel_path or "").strip()
    cfi = (cfi or "").strip()
    if not rel_path or not cfi:
        return
    conn.execute(
        """
        INSERT INTO reading_progress (rel_path, cfi, updated_at)
        VALUES (?, ?, unixepoch())
        ON CONFLICT(rel_path) DO UPDATE SET
          cfi=excluded.cfi,
          updated_at=unixepoch();
        """,
        (rel_path, cfi),
    )
    conn.commit()


def _safe_resolve(root: Path, rel_path: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / rel_path).resolve()

    try:
        candidate.relative_to(root_resolved)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Not found")

    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=404, detail="Not found")

    return candidate


def _ensure_indexed(app: FastAPI) -> None:
    """Ensure DB exists and has at least one indexing pass.

    This is defensive. If startup indexing was skipped (e.g. reload timing, empty /data mount,
    first run after changing DB path), we self-heal on first request.
    """

    conn: sqlite3.Connection | None = getattr(app.state, "db", None)
    if conn is None:
        conn = _connect_db()
        _init_db(conn)
        app.state.db = conn

    # If the books table is empty, do a best-effort index.
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM books").fetchone()
        current = int(row["c"]) if row is not None else 0
    except Exception:
        # If schema is missing for any reason, recreate it.
        _init_db(conn)
        current = 0

    if current > 0:
        return

    settings = get_settings()
    try:
        index_conn = _connect_db()
        try:
            _init_db(index_conn)
            indexed_books, indexed_recipes = _index_books(index_conn, settings.library_dir)
        finally:
            try:
                index_conn.close()
            except Exception:
                pass
        logger.info(
            "Indexed %s books (%s recipes) into DB at %s",
            indexed_books,
            indexed_recipes,
            _db_path(),
        )
    except Exception as exc:
        # Do not crash the app. Record the error so we can surface it.
        logger.exception("Indexing failed: %s", exc)
        app.state.last_index_error = str(exc)


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version="0.3.0",
        description="Self-hosted cookbook and recipe search app.",
    )

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/api/cookbooks", response_class=JSONResponse)
    def api_cookbooks(q: str | None = None, sort: str | None = "title") -> dict:
        """Live-search endpoint used by the frontend.

        Returns a JSON payload shaped for the JS search script in library.html.
        """

        _ensure_indexed(app)

        conn: sqlite3.Connection = app.state.db
        # Only filter once the query is long enough. Otherwise return the full shelf.
        books = _query_books_api(conn, q=q, sort=sort)
        _attach_cover_urls(app, books)

        return {
            "count": len(books),
            "books": books,
        }


    @app.get("/api/progress/{rel_path:path}", response_class=JSONResponse, name="api_get_progress")
    def api_get_progress(rel_path: str) -> dict:
        """Return last saved reading location for a book.

        For EPUBs we store the location as an EPUB CFI string.
        """
        _ensure_indexed(app)
        conn: sqlite3.Connection = app.state.db
        cfi = _get_progress_cfi(conn, rel_path)
        return {"rel_path": rel_path, "cfi": cfi}


    @app.put("/api/progress/{rel_path:path}", response_class=JSONResponse, name="api_put_progress")
    def api_put_progress(rel_path: str, payload: ProgressIn) -> dict:
        """Save reading location for a book."""
        _ensure_indexed(app)
        conn: sqlite3.Connection = app.state.db
        _set_progress_cfi(conn, rel_path, payload.cfi)
        return {"ok": True}

    @app.on_event("startup")
    def _startup() -> None:
        conn = _connect_db()
        _init_db(conn)
        app.state.db = conn

        settings = get_settings()
        logger.info("Startup. library_dir=%s db_path=%s", settings.library_dir, _db_path())

        def _run_startup_index() -> None:
            app.state.indexing_in_progress = True
            try:
                startup_conn = _connect_db()
                try:
                    _init_db(startup_conn)
                    indexed_books, indexed_recipes = _index_books(
                        startup_conn, settings.library_dir
                    )
                finally:
                    try:
                        startup_conn.close()
                    except Exception:
                        pass
                logger.info(
                    "Startup indexing complete. indexed_books=%s indexed_recipes=%s",
                    indexed_books,
                    indexed_recipes,
                )
            except Exception as exc:
                logger.exception("Startup indexing failed: %s", exc)
                app.state.last_index_error = str(exc)
            finally:
                app.state.indexing_in_progress = False

        import threading

        threading.Thread(target=_run_startup_index, daemon=True).start()

    @app.get("/health", response_class=JSONResponse)
    def health() -> dict:
        conn: sqlite3.Connection | None = getattr(app.state, "db", None)
        count = None
        if conn is not None:
            try:
                count = int(conn.execute("SELECT COUNT(*) FROM books").fetchone()[0])
            except Exception:
                count = None

        return {
            "status": "ok",
            "app": APP_NAME,
            "env": get_env("BAYLEAF_ENV", "dev"),
            "db_path": _db_path(),
            "books_indexed": count,
            "min_search_chars": _min_search_chars(),
            "indexing_in_progress": bool(getattr(app.state, "indexing_in_progress", False)),
            "reindex_in_progress": bool(getattr(app.state, "reindex_in_progress", False)),
        }

    @app.post("/admin/reindex", response_class=JSONResponse)
    def admin_reindex(
        full: bool = Query(False),
        async_: bool = Query(False, alias="async"),
    ) -> dict:
        """Re-run indexing.

        MVP note: This must be protected (auth or reverse proxy rules) before exposing publicly.
        """
        if getattr(app.state, "reindex_in_progress", False):
            return {"error": "Reindex already in progress"}

        def _run_reindex() -> dict:
            settings = get_settings()

            library_dir = settings.library_dir
            if not library_dir.exists():
                return {
                    "indexed_books": 0,
                    "indexed_recipes": 0,
                    "library_dir": str(library_dir),
                    "error": "Library directory not found",
                    "epub_count": 0,
                }

            # Use a fresh connection for reindexing so we never collide with the shared
            # connection's transaction state (health checks, concurrent requests, etc.).
            conn = _connect_db()
            try:
                _init_db(conn)
                if full:
                    with conn:
                        conn.execute("DELETE FROM recipes")
                indexed_books, indexed_recipes = _index_books(conn, settings.library_dir)

                # Refresh the long-lived connection so subsequent requests (health/home)
                # see the newly committed data immediately.
                old: sqlite3.Connection | None = getattr(app.state, "db", None)
                try:
                    if old is not None:
                        old.close()
                except Exception:
                    pass

                fresh = _connect_db()
                _init_db(fresh)
                app.state.db = fresh

                epub_count = sum(
                    1
                    for b in list_cookbooks(library_dir)
                    if str(getattr(b, "suffix", "")).lower() == ".epub"
                )
                return {
                    "indexed_books": indexed_books,
                    "indexed_recipes": indexed_recipes,
                    "library_dir": str(settings.library_dir),
                    "epub_count": epub_count,
                }
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        if async_:
            import threading

            def _worker() -> None:
                app.state.reindex_in_progress = True
                try:
                    app.state.last_reindex_result = _run_reindex()
                except Exception as exc:
                    logger.exception("Admin reindex failed: %s", exc)
                    app.state.last_index_error = str(exc)
                    app.state.last_reindex_result = {"error": str(exc)}
                finally:
                    app.state.reindex_in_progress = False

            threading.Thread(target=_worker, daemon=True).start()
            return {"status": "started", "full": full}

        try:
            return _run_reindex()
        except Exception as exc:
            logger.exception("Admin reindex failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/admin/recipe-report", response_class=JSONResponse)
    def admin_recipe_report(
        path: str = Query(..., alias="path"),
        limit: int = Query(12, ge=1, le=200),
    ) -> dict:
        """Return a sample of extracted recipes for a single book."""
        settings = get_settings()
        file_path = _safe_resolve(settings.library_dir, path)
        if file_path.suffix.lower() != ".epub":
            raise HTTPException(status_code=400, detail="Only EPUB files are supported")

        recipes = extract_epub_recipes(file_path, max_recipes=limit)
        sample = [
            {
                "title": r.get("title"),
                "source_key": r.get("source_key"),
                "image_href": r.get("image_href"),
                "has_ingredients": bool(r.get("ingredients_text")),
                "has_method": bool(r.get("method_text")),
            }
            for r in recipes
        ]
        return {
            "rel_path": path,
            "limit": limit,
            "extracted": len(recipes),
            "sample": sample,
        }

    @app.post("/admin/reextract-recipes", response_class=JSONResponse)
    def admin_reextract_recipes(path: str = Query(..., alias="path")) -> dict:
        """Force re-extraction of recipes for a single book."""
        settings = get_settings()
        conn = _connect_db()
        try:
            _init_db(conn)
            result = _reextract_recipes_for_book(conn, settings.library_dir, path)
            if result.get("error"):
                return result
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Recipe re-extract failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            try:
                conn.close()
            except Exception:
                pass

    @app.get("/", response_class=HTMLResponse, name="home")
    def home(request: Request, q: str | None = None, sort: str | None = "title"):
        settings = get_settings()

        _ensure_indexed(app)

        notice = ""
        if not settings.library_dir.exists():
            notice = (
                f"Library folder not found at {settings.library_dir}. "
                "Check your compose volume and BAYLEAF_LIBRARY_HOST_DIR."
            )

        conn: sqlite3.Connection = app.state.db
        # Apply server-side filtering only once the query is long enough.
        books, total = _query_books(conn, q, sort=sort)
        _attach_cover_urls(app, books)

        if total == 0:
            # Fallback to filesystem scan so the UI never shows empty when files are mounted.
            try:
                fs_books_raw = list_cookbooks(settings.library_dir)
                fs_books = [_cookbook_to_dict(b) for b in fs_books_raw]
                # Derive rel_path for rendering links.
                root = Path(settings.library_dir)
                for bd in fs_books:
                    if bd.get("rel_path"):
                        continue
                    abs_path = bd.get("abs_path") or ""
                    if abs_path:
                        try:
                            bd["rel_path"] = str(Path(abs_path).resolve().relative_to(root.resolve()))
                        except Exception:
                            bd["rel_path"] = Path(abs_path).name
                    else:
                        bd["rel_path"] = bd.get("name") or "unknown"
                books = fs_books
                total = len(fs_books)
                _attach_cover_urls(app, books)
                if getattr(app.state, "last_index_error", ""):
                    notice = (
                        (notice + " " if notice else "")
                        + f"Indexing error: {app.state.last_index_error}"
                    )
                else:
                    notice = (
                        (notice + " " if notice else "")
                        + "DB appears empty. Showing filesystem results. Use /admin/reindex to rebuild the index."
                    )
            except Exception as exc:
                notice = (
                    (notice + " " if notice else "")
                    + f"Could not scan filesystem: {exc}"
                )

        if books:
            _attach_cover_urls(app, books)

        return templates.TemplateResponse(
            "library.html",
            {
                "request": request,
                "app_name": APP_NAME,
                "logo_url": "/static/Bayleaf_Logo.png",
                # Primary names
                "books": books,
                "total": total,
                # Backwards-compatible aliases for templates that still use older names
                "cookbooks": books,
                "count": total,
                "q": (q or "").strip(),
                "sort": (sort or "title").strip(),
                "library_dir": str(settings.library_dir),
                "notice": notice,
            },
        )

    @app.get("/read", response_class=HTMLResponse, name="read_book_query")
    def read_book_query(request: Request, path: str = Query(..., alias="path")):
        # Delegate to the path-based handler for shared logic.
        return read_book(request, path)

    @app.get("/read/{rel_path:path}", response_class=HTMLResponse, name="read_book")
    def read_book(request: Request, rel_path: str):
        settings = get_settings()
        file_path = _safe_resolve(settings.library_dir, rel_path)
        title = file_path.stem
        file_type = file_path.suffix.lower().lstrip(".")

        # Prefer relative URLs for reverse-proxy friendliness.
        file_url = f"/book?path={quote(rel_path)}"
        progress_url = str(app.url_path_for("api_get_progress", rel_path=rel_path))
        progress_put_url = str(app.url_path_for("api_put_progress", rel_path=rel_path))

        return templates.TemplateResponse(
            "read.html",
            {
                "request": request,
                "title": title,
                "rel_path": rel_path,
                "file_type": file_type,
                "file_url": file_url,
                "progress_url": progress_url,
                "progress_put_url": progress_put_url,
            },
        )

    @app.get("/recipes", response_class=HTMLResponse, name="recipes")
    def recipes_page(request: Request, q: str | None = None):
        conn = app.state.db
        recipes, total = _query_recipes(conn, q)
        cover_books = [{"rel_path": r["rel_path"]} for r in recipes]
        _attach_cover_urls(app, cover_books)
        cover_map = {b["rel_path"]: b.get("cover_url") for b in cover_books}
        for r in recipes:
            r["cover_url"] = cover_map.get(r["rel_path"])
            image_href = r.get("image_href") or ""
            if image_href:
                r["image_url"] = f"/recipe-image?path={quote(r['rel_path'])}&img={quote(image_href)}"
            else:
                r["image_url"] = r.get("cover_url")
        return templates.TemplateResponse(
            "recipes.html",
            {
                "request": request,
                "recipes": recipes,
                "total": total,
                "q": (q or "").strip(),
            },
        )

    @app.get("/book", name="book")
    def book(path: str = Query(..., alias="path")):
        # Stream an EPUB/PDF using a query parameter. This avoids path-routing edge cases
        # with spaces, commas, and other characters.
        return file(path)

    @app.get("/recipe-image", name="recipe_image")
    def recipe_image(
        path: str = Query(..., alias="path"),
        img: str = Query(..., alias="img"),
    ):
        settings = get_settings()
        file_path = _safe_resolve(settings.library_dir, path)
        if file_path.suffix.lower() != ".epub":
            raise HTTPException(status_code=404, detail="Not found")

        from ebooklib import epub as epub_lib  # local import to keep startup fast

        img_href = unquote(img).lstrip("/")
        try:
            book = epub_lib.read_epub(str(file_path))
            item = book.get_item_with_href(img_href) or book.get_item_with_name(img_href)
            if item is None:
                for prefix in ("EPUB/", "OEBPS/"):
                    if img_href.startswith(prefix):
                        trimmed = img_href[len(prefix) :]
                        item = book.get_item_with_href(trimmed) or book.get_item_with_name(trimmed)
                        if item is not None:
                            break
            if item is None:
                raise HTTPException(status_code=404, detail="Not found")
            media_type = item.media_type or "application/octet-stream"
            return Response(content=item.get_content(), media_type=media_type)
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Recipe image fetch failed for %s: %s", img_href, exc)
            raise HTTPException(status_code=404, detail="Not found") from exc

    @app.get("/file/{rel_path:path}", name="file")
    def file(rel_path: str):
        settings = get_settings()
        file_path = _safe_resolve(settings.library_dir, rel_path)

        media_type = "application/octet-stream"
        if file_path.suffix.lower() == ".pdf":
            media_type = "application/pdf"
        if file_path.suffix.lower() == ".epub":
            media_type = "application/epub+zip"

        return FileResponse(path=str(file_path), media_type=media_type, filename=file_path.name)


    @app.head("/cover/{rel_path:path}")
    def cover_head(rel_path: str):
        # FastAPI doesn't always auto-register HEAD for this route in our setup.
        # Delegate to the GET handler. Starlette will strip the body for HEAD.
        return cover(rel_path)

    @app.get("/cover/{rel_path:path}", name="cover")
    def cover(rel_path: str):
        """Return a cover thumbnail for a book.

        Priority:
        1) Real EPUB cover (cached under /data/covers)
        2) Placeholder cover (cached)
        """
        settings = get_settings()
        file_path = _safe_resolve(settings.library_dir, rel_path)

        mtime = int(file_path.stat().st_mtime)
        key = _cover_cache_key(rel_path, mtime)
        covers_dir = _covers_dir()

        suffix = file_path.suffix.lower()

        # 1) Real cover for EPUBs via app/covers.py
        if suffix == ".epub":
            try:
                from app.covers import get_or_create_epub_cover

                out_path = get_or_create_epub_cover(
                    epub_path=file_path,
                    covers_dir=covers_dir,
                    cache_key=key,
                )

                if out_path is not None:
                    out_path = Path(out_path)
                    if out_path.exists() and out_path.is_file():
                        ext = out_path.suffix.lower()
                        if ext in {".jpg", ".jpeg"}:
                            return FileResponse(str(out_path), media_type="image/jpeg")
                        if ext == ".webp":
                            return FileResponse(str(out_path), media_type="image/webp")
                        if ext == ".svg":
                            return FileResponse(str(out_path), media_type="image/svg+xml")
                        return FileResponse(str(out_path), media_type="image/png")
            except Exception as exc:
                logger.info("EPUB cover extraction failed for %s: %s", rel_path, exc)

        # 2) Fallback: cached placeholder
        out_path = covers_dir / f"{key}.png"
        if not out_path.exists():
            title = _title_from_filename(file_path.name)
            _make_placeholder_cover(out_path, title)

        # If we had to fall back to SVG, serve it correctly
        if out_path.suffix.lower() == ".svg":
            return FileResponse(str(out_path), media_type="image/svg+xml")

        return FileResponse(str(out_path), media_type="image/png")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=get_env("BAYLEAF_HOST", "0.0.0.0"),
        port=int(get_env("BAYLEAF_PORT", "8000")),
        reload=get_env("BAYLEAF_RELOAD", "true").lower() in {"1", "true", "yes"},
        log_level=get_env("BAYLEAF_LOG_LEVEL", "info"),
    )
