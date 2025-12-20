from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import click
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import json
import sqlite3


# -------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------


@dataclass
class Recipe:
    title: str
    normalised_title: str
    book_title: str
    book_path: str
    href: Optional[str] = None      # link target in the epub, if any
    section: Optional[str] = None   # e.g. Desserts, Starters etc


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def normalise_title(raw: str) -> str:
    """Clean up a recipe title, ready for sorting and display."""
    text = raw.strip()

    # Remove common trailing hints like "(see ...)"
    text = re.sub(r"\(see[^)]+\)$", "", text, flags=re.IGNORECASE).strip()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Avoid empty after cleaning
    if not text:
        return raw.strip()

    # Title case, but keep "ALL CAPS" as is
    def smart_cap(word: str) -> str:
        if word.isupper():
            return word
        return word.capitalize()

    return " ".join(smart_cap(w) for w in text.split(" "))


def looks_like_recipe_candidate(text: str) -> bool:
    """Simple heuristic for whether some text is probably a recipe title."""
    text = text.strip()
    if not text:
        return False

    # Length sanity
    if len(text) < 3 or len(text) > 80:
        return False

    lower = text.lower()

    # Filter out pure page numbers like "264"
    if re.fullmatch(r"\d+", text):
        return False

    # Filter out simple numeric ranges like "12-14" or "12–14"
    if re.fullmatch(r"\d+\s*[-–]\s*\d+", text):
        return False

    # Also filter strings that are mostly digits (e.g. "119-20")
    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count > 0 and digit_count / len(text) > 0.6:
        return False

    bad_words = [
        "index",
        "page",
        "see also",
        "see ",
        "chapter",
        "contents",
        "about the author",
    ]
    if any(b in lower for b in bad_words):
        return False

    # Avoid lines that look mostly like page references or ranges at the end
    if re.search(r"\d{2,}\s*-\s*\d{2,}$", text):
        return False

    # Avoid too much punctuation
    if sum(1 for c in text if c in ".:,;/\\") > 4:
        return False

    return True


# -------------------------------------------------------------------
# EPUB parsing
# -------------------------------------------------------------------


def load_epub(book_path: Path) -> epub.EpubBook:
    return epub.read_epub(str(book_path))


def get_book_title(book: epub.EpubBook, book_path: Path) -> str:
    meta_title = book.get_metadata("DC", "title")
    if meta_title and meta_title[0] and meta_title[0][0]:
        return meta_title[0][0].strip()
    return book_path.stem


def iter_document_items(book: epub.EpubBook):
    """Yield (item, soup) for each XHTML/HTML 'document' in the EPUB."""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_bytes = item.get_content()
        try:
            html_text = html_bytes.decode("utf-8", errors="ignore")
        except Exception:
            continue
        soup = BeautifulSoup(html_text, "html.parser")
        yield item, soup


def looks_like_index_doc(item, soup: BeautifulSoup) -> bool:
    """
    Heuristic to detect an 'index' document in an EPUB.

    Signals.
      - Filename or id contains 'index'
      - First heading contains 'index' or 'recipe index'
      - Contains lots of links in short paragraphs or list items
    """
    name = getattr(item, "file_name", "") or ""
    id_ = getattr(item, "id", "") or ""
    name_lower = name.lower()
    id_lower = id_.lower()

    if "index" in name_lower or "recipe-index" in name_lower:
        return True

    # Check headings
    headings = soup.find_all(["h1", "h2", "h3"])
    for h in headings:
        text = (h.get_text() or "").strip().lower()
        if "index" in text or "recipe index" in text:
            return True

    # Link density heuristic. A lot of cookbook indexes are "lists of links".
    links = soup.find_all("a")
    if not links:
        return False

    # If many links and they mostly appear in <li> or <p>, treat as index-like
    list_like_parents = 0
    for a in links:
        parent = a.parent
        if parent and parent.name in ("li", "p", "div"):
            list_like_parents += 1

    ratio = list_like_parents / max(len(links), 1)
    return len(links) >= 20 and ratio > 0.5


def extract_index_entries_from_doc(
    book: epub.EpubBook,
    book_path: Path,
    item,
    soup: BeautifulSoup,
    section_hint: Optional[str] = None,
) -> List[Recipe]:
    """
    From a suspected index document, pull out candidate recipe titles.
    We mainly look at <a> elements, since indexes usually link into chapters.
    """
    recipes: List[Recipe] = []

    book_title = get_book_title(book, book_path)

    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip()
        if not looks_like_recipe_candidate(text):
            continue

        href = a.get("href") or None

        # Skip obvious page markers like "264" that usually have anchors like "#pg264"
        if href and "#pg" in href.lower():
            continue

        # Only keep links that clearly point into a chapter and a specific anchor
        # e.g. "chapter003_01.xhtml#cru0001918"
        if not (href and "chapter" in href.lower() and "#" in href):
            continue

        # Exact-title blacklist for common non-recipe entries
        lower_text = text.lower()
        non_recipe_titles = {
            "acknowledgements",
            "acknowledgments",
            "copyright",
            "frontmatter",
            "welcome",
            "team photos",
            "essential ingredients",
            "essential kit",
            "kitchen rules",
            "above",
            "left",
            "method",
            "prep",
            "tip",
        }
        if lower_text in non_recipe_titles:
            continue

        recipes.append(
            Recipe(
                title=text,
                normalised_title=normalise_title(text),
                book_title=book_title,
                book_path=str(book_path),
                href=href,
                section=section_hint,
            )
        )

    return recipes


def extract_recipes_from_epub_index(book_path: Path) -> List[Recipe]:
    """
    High level function. Given a single EPUB, return a list of Recipe objects
    based on index-like documents.
    """
    book = load_epub(book_path)

    # Optional. use ToC to infer sections like "Starters", "Desserts" etc
    # For v1, we will simply try to detect index docs and ignore section grouping.

    recipes: List[Recipe] = []

    for item, soup in iter_document_items(book):
        if not looks_like_index_doc(item, soup):
            continue

        # In future. derive section_hint from the context or heading text.
        section_hint = None
        doc_recipes = extract_index_entries_from_doc(
            book, book_path, item, soup, section_hint=section_hint
        )
        recipes.extend(doc_recipes)

    # Deduplicate by (normalised_title, href) if multiple index docs exist
    unique = {}
    for r in recipes:
        key = (r.normalised_title.lower(), r.href)
        if key not in unique:
            unique[key] = r

    final_recipes = list(unique.values())
    final_recipes.sort(key=lambda r: r.normalised_title.lower())
    return final_recipes


# -------------------------------------------------------------------
# Recipe plaintext extraction
# -------------------------------------------------------------------

def extract_recipe_plaintext(book_path: Path, href: str) -> str:
    """
    Given an EPUB path and a recipe href (e.g. 'chapter001_01.xhtml#cru0000056'),
    return a best-effort plain-text version of the recipe by grabbing the heading
    and its following content until the next major heading.
    """
    book = load_epub(book_path)

    # Split href into document and anchor
    if "#" in href:
        doc_name, anchor = href.split("#", 1)
    else:
        doc_name, anchor = href, None

    # Find the target document item
    target_item = None
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        file_name = getattr(item, "file_name", "") or ""
        if file_name == doc_name or doc_name in file_name:
            target_item = item
            break

    if target_item is None:
        return f"[Error] Could not find document for href '{href}'"

    html_bytes = target_item.get_content()
    html_text = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")

    # Locate the starting element for the anchor
    start = None
    if anchor:
        start = soup.find(id=anchor)
        if start is None:
            # Some EPUBs use name=anchor on <a> tags
            start = soup.find("a", attrs={"name": anchor})

    if start is None:
        # Fallback: just return the whole document text
        return soup.get_text("\n", strip=True)

    # If the anchor is inside a heading, use the heading as the start node
    if start.name == "a" and start.parent and start.parent.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        start = start.parent

    # Collect heading + following siblings until the next major heading
    parts: list[str] = []
    heading_text = start.get_text(" ", strip=True)
    if heading_text:
        parts.append(heading_text)

    for sib in start.next_siblings:
        # Stop when we hit the next major heading
        if getattr(sib, "name", None) in ("h1", "h2", "h3", "h4", "h5", "h6"):
            break
        # Only collect meaningful blocks
        if getattr(sib, "name", None) in ("p", "ul", "ol", "div", "section", "table"):
            block_text = sib.get_text(" ", strip=True)
            if block_text:
                parts.append(block_text)

    # As a safety, if we somehow collected nothing, fall back to whole document
    if not parts:
        return soup.get_text("\n", strip=True)

    return "\n\n".join(parts)


def extract_recipe_image_href(book_path: Path, href: str) -> Optional[str]:
    """Find the nearest image to a recipe anchor and return its EPUB-relative href."""
    book = load_epub(book_path)

    if "#" in href:
        doc_name, anchor = href.split("#", 1)
    else:
        doc_name, anchor = href, None

    target_item = None
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        file_name = getattr(item, "file_name", "") or ""
        if file_name == doc_name or doc_name in file_name:
            target_item = item
            break

    if target_item is None:
        return None

    html_bytes = target_item.get_content()
    html_text = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html_text, "html.parser")

    start = None
    if anchor:
        start = soup.find(id=anchor)
        if start is None:
            start = soup.find("a", attrs={"name": anchor})

    if start is None:
        return None

    if start.name == "a" and start.parent and start.parent.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        start = start.parent

    def _resolve_src(src: str) -> Optional[str]:
        src = (src or "").strip()
        if not src or src.startswith("data:") or "://" in src:
            return None
        base_dir = "/".join(doc_name.split("/")[:-1])
        joined = f"{base_dir}/{src}" if base_dir else src
        parts = []
        for part in joined.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        return "/".join(parts)

    # First pass: scan forward siblings for an image.
    for sib in start.next_siblings:
        if getattr(sib, "name", None) in ("img",):
            src = sib.get("src")
            resolved = _resolve_src(src)
            if resolved:
                return resolved
        if getattr(sib, "name", None) in ("p", "div", "section", "figure"):
            img = sib.find("img")
            if img:
                resolved = _resolve_src(img.get("src"))
                if resolved:
                    return resolved

    # Second pass: search within a nearby container.
    container = start.parent
    while container is not None and getattr(container, "name", None) not in ("section", "div", "article", "body"):
        container = container.parent
    if container is not None:
        img = container.find("img")
        if img:
            resolved = _resolve_src(img.get("src"))
            if resolved:
                return resolved

    return None


def split_recipe_text(raw: str) -> tuple[list[str], str]:
    """Split a plain-text recipe into ingredients (as bullet items) and method.

    Assumes the first non-empty line is the title (already known from metadata).
    Tries, in order:
      1. Marker-based split using 'INGREDIENTS' and 'METHOD'/'DIRECTIONS'.
      2. A naive quantity-based heuristic if no clear markers exist.
    """
    # Remove blank lines and trim
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return [], ""

    # Drop the first line, which should be the recipe title heading
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    if not body:
        return [], ""

    text = body

    # ------------------------------------------------------------------
    # 1. Marker-based split for formats like "INGREDIENTS ... DIRECTIONS ..."
    # ------------------------------------------------------------------
    m_ing = re.search(r"\bINGREDIENTS\b", text, flags=re.IGNORECASE)
    m_method = re.search(r"\b(METHOD|DIRECTIONS?)\b", text, flags=re.IGNORECASE)

    ingredients_block = ""
    method_block = ""

    if m_ing and m_method and m_method.start() > m_ing.end():
        # INGREDIENTS ... DIRECTIONS ...
        ingredients_block = text[m_ing.end() : m_method.start()].strip()
        method_block = text[m_method.end() :].strip()
    elif m_ing:
        # INGREDIENTS ... (no explicit method marker)
        ingredients_block = text[m_ing.end() :].strip()
        method_block = ""
    elif m_method:
        # METHOD/DIRECTIONS ... (treat text before as "ingredients-ish")
        ingredients_block = text[: m_method.start()].strip()
        method_block = text[m_method.end() :].strip()

    if ingredients_block or method_block:
        ingredients = _parse_ingredients_from_block(ingredients_block) if ingredients_block else []
        # If we have no explicit method block, treat the remaining text as method
        if not method_block:
            # Use the full text minus the ingredients block as a best-effort method text
            if ingredients_block:
                method_block = text.replace(ingredients_block, "", 1).strip()
            else:
                method_block = text

        return ingredients, method_block

    # ------------------------------------------------------------------
    # 2. Fallback: naive quantity-based split for books without markers
    # ------------------------------------------------------------------
    return _naive_split_ingredients_and_method(body)


# Helper: Parse an ingredients block into lines, handling headings and splitting on quantity patterns
def _parse_ingredients_from_block(block: str) -> list[str]:
    """Parse an ingredients block into a list of ingredient lines.

    Handles headings like 'For the Pork:' and uses simple quantity-based
    heuristics to split long lines into separate items where possible.
    """
    block = (block or "").strip()
    if not block:
        return []

    # Put each "For the X:" on its own line to preserve structure
    block = re.sub(r"(For the [^:]+:)", r"\n\1", block, flags=re.IGNORECASE)

    lines: list[str] = []
    for ln in block.splitlines():
        ln = ln.strip()
        if ln:
            lines.append(ln)

    ingredients: list[str] = []

    # Pattern to find chunks that look like "125g butter" or "¼ tsp salt"
    qty_pattern = r"(\d+[^\d¼½¾⅓⅔]+|[¼½¾⅓⅔]\s*[^0-9¼½¾⅓⅔]+)"

    for ln in lines:
        # Keep section headings as their own bullet
        if re.match(r"^for the ", ln, flags=re.IGNORECASE):
            ingredients.append(ln.strip())
            # Parse the rest of the line after the colon as actual ingredients
            if ":" in ln:
                rest = ln.split(":", 1)[1].strip()
                if rest:
                    matches = list(re.finditer(qty_pattern, rest))
                    if matches:
                        for m in matches:
                            item = m.group(0).strip()
                            if item:
                                ingredients.append(item)
                    else:
                        ingredients.append(rest)
            continue

        # For normal lines, try to split into quantity-based chunks
        matches = list(re.finditer(qty_pattern, ln))
        if matches:
            for m in matches:
                item = m.group(0).strip()
                if item:
                    ingredients.append(item)
        else:
            ingredients.append(ln)

    # If we somehow ended up with nothing, fall back to a single block
    if not ingredients and block:
        ingredients.append(block.strip())

    return ingredients


# Helper: Fallback splitter based on quantities and units
def _naive_split_ingredients_and_method(text: str) -> tuple[list[str], str]:
    """Fallback splitter when there are no clear INGREDIENTS/METHOD markers.

    Uses simple quantity + unit heuristics to decide which lines are ingredients.
    """
    text = (text or "").strip()
    if not text:
        return [], ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], ""

    unit_keywords = [
        "g",
        "gram",
        "grams",
        "kg",
        "ml",
        "l ",
        "litre",
        "litres",
        "liter",
        "liters",
        "tsp",
        "teaspoon",
        "teaspoons",
        "tbsp",
        "tablespoon",
        "tablespoons",
        "cup",
        "cups",
        "oz",
        "ounce",
        "ounces",
        "lb",
        "pound",
        "pounds",
        "pinch",
        "clove",
        "cloves",
        "slice",
        "slices",
        "stick",
        "sticks",
    ]

    ingredients: list[str] = []
    method_lines: list[str] = []

    for ln in lines:
        low = ln.lower()
        has_digit = any(ch.isdigit() for ch in ln) or re.search(r"[¼½¾⅓⅔]", ln)
        has_unit = any(u in low for u in unit_keywords)

        # Treat "Serves X"/"Makes X" as yield/ingredient-like
        if re.match(r"^(serves|makes)\b", low):
            ingredients.append(ln)
            continue

        if has_digit and has_unit:
            ingredients.append(ln)
        else:
            method_lines.append(ln)

    # If nothing classified as ingredients, treat everything as method
    if not ingredients:
        return [], "\n\n".join(method_lines).strip()

    method_text = "\n\n".join(method_lines).strip()
    return ingredients, method_text


# -------------------------------------------------------------------
# Method step splitter
# -------------------------------------------------------------------

def split_method_into_steps(text: str) -> list[str]:
    """Split a method paragraph into numbered, reasonably sized steps.

    Uses sentence boundaries (., !, ?) followed by a capital letter as hints,
    then merges short sentences into the previous step so you don't end up
    with too many tiny steps.
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentence-like chunks
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return []

    steps: list[str] = []
    buffer = ""

    for s in sentences:
        if not buffer:
            buffer = s
            continue

        # If the current buffer is short, merge the next sentence into it
        if len(buffer) < 80:
            buffer = buffer + " " + s
        else:
            steps.append(buffer)
            buffer = s

    if buffer:
        steps.append(buffer)

    return steps


# -------------------------------------------------------------------
# Structured recipe export helpers
# -------------------------------------------------------------------

def build_structured_recipes(book_path: Path, recipes: List[Recipe]) -> list[dict]:
    """
    Given a book path and a list of Recipe objects, return a list of structured
    recipe dicts including ingredients and method steps.
    """
    structured: list[dict] = []
    for r in recipes:
        if not r.href:
            # Without an href we cannot reliably locate the recipe in the EPUB
            continue
        text = extract_recipe_plaintext(book_path, r.href)
        ingredients, method = split_recipe_text(text)
        steps = split_method_into_steps(method) if method else []
        structured.append(
            {
                "book_title": r.book_title,
                "book_path": r.book_path,
                "title": r.title,
                "normalised_title": r.normalised_title,
                "href": r.href,
                "ingredients": ingredients,
                "method": method,
                "steps": steps,
            }
        )
    return structured


def write_recipes_to_db(db_path: Path, recipes: list[dict]) -> None:
    """
    Create or append to a SQLite database with a 'recipes' table containing
    one row per structured recipe.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_title TEXT,
            book_path TEXT,
            recipe_title TEXT,
            normalised_title TEXT,
            href TEXT,
            ingredients_json TEXT,
            method_text TEXT,
            steps_json TEXT
        )
        """
    )
    for rec in recipes:
        cur.execute(
            """
            INSERT INTO recipes (
                book_title,
                book_path,
                recipe_title,
                normalised_title,
                href,
                ingredients_json,
                method_text,
                steps_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.get("book_title"),
                rec.get("book_path"),
                rec.get("title"),
                rec.get("normalised_title"),
                rec.get("href"),
                json.dumps(rec.get("ingredients") or [], ensure_ascii=False),
                rec.get("method") or "",
                json.dumps(rec.get("steps") or [], ensure_ascii=False),
            ),
        )
    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


@click.command()
@click.argument("epub_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output JSON file for extracted recipes.",
)
@click.option(
    "--recipe-title",
    "recipe_title",
    type=str,
    default=None,
    help="If provided, extract and print this single recipe by title (case-insensitive).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="If provided, write all structured recipes for this book into a SQLite database at this path.",
)
def cli(epub_path: str, out_path: Optional[str], recipe_title: Optional[str], db_path: Optional[str]) -> None:
    """
    Extract recipes for a single EPUB and print a summary,
    or print a specific recipe if --recipe-title is provided.
    """
    book_path = Path(epub_path)
    recipes = extract_recipes_from_epub_index(book_path)

    if recipe_title:
        wanted = recipe_title.strip().lower()
        target = None
        for r in recipes:
            if r.title.lower() == wanted or r.normalised_title.lower() == wanted:
                target = r
                break

        if target is None:
            click.echo(f"[Error] No recipe found with title matching '{recipe_title}'")
            click.echo("Here are some available titles:")
            for r in recipes[:30]:
                click.echo(f" - {r.title}")
            return

        if not target.href:
            click.echo(f"[Error] Recipe '{target.title}' has no href, cannot locate in EPUB.")
            return

        text = extract_recipe_plaintext(book_path, target.href)
        ingredients, method = split_recipe_text(text)

        click.echo(f"=== {target.title} ({target.href}) ===\n")

        if ingredients:
            click.echo("Ingredients:\n")
            for item in ingredients:
                click.echo(f"- {item}")

        if method:
            click.echo("\nMethod:\n")
            steps = split_method_into_steps(method)
            if not steps:
                # Fallback to printing the raw method text if splitting fails
                click.echo(method.strip())
            else:
                for idx, step in enumerate(steps, start=1):
                    click.echo(f"{idx}. {step}")

        return

    click.echo(f"Found {len(recipes)} candidate recipes in {book_path.name}")
    for r in recipes[:30]:
        href_display = f" ({r.href})" if r.href else ""
        click.echo(f" - {r.normalised_title}{href_display}")

    structured: list[dict] | None = None
    if out_path or db_path:
        structured = build_structured_recipes(book_path, recipes)

    if out_path and structured is not None:
        out_file = Path(out_path)
        out_file.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"\nSaved {len(structured)} structured recipes to {out_file}")

    if db_path and structured is not None:
        db_file = Path(db_path)
        write_recipes_to_db(db_file, structured)
        click.echo(f"\nSaved {len(structured)} structured recipes to database {db_file}")


if __name__ == "__main__":
    cli()
