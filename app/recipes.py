"""Recipe extraction helpers for EPUB files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
import re
import posixpath
from pathlib import PurePosixPath
from typing import Iterable, List, Optional

from ebooklib import epub

from app.config import get_env
from app.epub_recipe_engine import (
    extract_recipe_image_href,
    extract_recipe_plaintext,
    extract_recipes_from_epub_index,
    split_recipe_text,
)

IGNORE_TITLE_RE = re.compile(
    r"(acknowledg|equipment|conversion|glossary|index|about the author|contents|"
    r"introduction|chapter|bibliography|notes|preface)",
    re.IGNORECASE,
)

INGREDIENTS_RE = re.compile(r"\bingredient", re.IGNORECASE)
METHOD_RE = re.compile(r"\b(method|direction|instruction|preparation)\b", re.IGNORECASE)


@dataclass
class TextBlock:
    tag: str
    text: str
    level: Optional[int] = None
    element_id: Optional[str] = None

    @property
    def is_heading(self) -> bool:
        return self.level is not None


class BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.blocks: List[TextBlock] = []
        self._active_tag: Optional[str] = None
        self._active_id: Optional[str] = None
        self._buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag == "img":
            attr_map = {k: v for k, v in attrs}
            src = (attr_map.get("src") or "").strip()
            if src:
                self.blocks.append(TextBlock(tag="img", text=src))
            return

        if self._active_tag is not None:
            return
        if tag in {"p", "li", "h1", "h2", "h3", "h4"}:
            self._active_tag = tag
            attr_map = {k: v for k, v in attrs}
            self._active_id = attr_map.get("id")
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._active_tag is None:
            return
        self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._active_tag:
            return

        text = " ".join(" ".join(self._buf).split()).strip()
        if text:
            level = None
            if tag.startswith("h") and tag[1:].isdigit():
                level = int(tag[1:])
            self.blocks.append(
                TextBlock(tag=tag, text=text, level=level, element_id=self._active_id)
            )

        self._active_tag = None
        self._active_id = None
        self._buf = []


def _iter_spine_docs(book: epub.EpubBook) -> Iterable[tuple[str, bytes]]:
    for item_id, _linear in book.spine:
        try:
            item = book.get_item_with_id(item_id)
        except Exception:
            continue
        if item is None:
            continue
        content = item.get_content()
        if not content:
            continue
        yield item.get_name(), content


def _parse_blocks(html: bytes) -> List[TextBlock]:
    parser = BlockParser()
    parser.feed(html.decode("utf-8", errors="ignore"))
    return parser.blocks


def _title_ignored(title: str) -> bool:
    if IGNORE_TITLE_RE.search(title):
        return True
    custom = _custom_ignore_re()
    if custom and custom.search(title):
        return True
    return False


@lru_cache(maxsize=1)
def _custom_ignore_re() -> re.Pattern[str] | None:
    raw = get_env("BAYLEAF_RECIPE_IGNORE_TITLES", "").strip()
    if not raw:
        return None
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return None
    escaped = [re.escape(token) for token in tokens]
    return re.compile("|".join(escaped), re.IGNORECASE)


def _extract_section(blocks: List[TextBlock]) -> List[dict]:
    sections: List[dict] = []
    current: dict | None = None

    for block in blocks:
        if block.is_heading and block.level is not None and block.level <= 4:
            if current:
                sections.append(current)
            current = {
                "title": block.text,
                "title_id": block.element_id,
                "blocks": [],
            }
            continue

        if current is not None:
            current["blocks"].append(block)

    if current:
        sections.append(current)

    return sections


def _collect_recipe_from_section(section: dict, href: str) -> Optional[dict]:
    title = (section.get("title") or "").strip()
    if not title or _title_ignored(title):
        return None

    ingredients: List[str] = []
    method: List[str] = []
    current_bucket: Optional[str] = None
    image_href: Optional[str] = None

    for block in section.get("blocks", []):
        if block.tag == "img" and not image_href:
            src = block.text.strip()
            if not src or src.startswith("data:") or "://" in src:
                continue
            if src.startswith("/"):
                image_href = src.lstrip("/")
            else:
                base_dir = posixpath.dirname(href)
                image_href = posixpath.normpath(posixpath.join(base_dir, src)).lstrip("/")
            continue
        if block.is_heading:
            if INGREDIENTS_RE.search(block.text):
                current_bucket = "ingredients"
            elif METHOD_RE.search(block.text):
                current_bucket = "method"
            else:
                current_bucket = None
            continue

        if current_bucket == "ingredients":
            ingredients.append(block.text)
        elif current_bucket == "method":
            method.append(block.text)

    if not method:
        return None

    if section.get("title_id"):
        source_key = f"{href}#{section['title_id']}"
    else:
        source_key = href

    return {
        "title": title,
        "ingredients_text": "\n".join(ingredients).strip() or None,
        "method_text": "\n".join(method).strip(),
        "source_type": "epub",
        "source_key": source_key,
        "image_href": image_href,
        "location_type": "href",
        "location_value": source_key,
    }


def extract_epub_recipes(epub_path: Path, *, max_recipes: int | None = None) -> List[dict]:
    if _is_crumbs_doilies(epub_path):
        return extract_crumbs_doilies_recipes(epub_path, max_recipes=max_recipes)

    index_recipes = _extract_epub_recipes_from_index(epub_path, max_recipes=max_recipes)
    if index_recipes:
        return index_recipes

    book = epub.read_epub(str(epub_path))
    recipes: List[dict] = []

    for href, html in _iter_spine_docs(book):
        blocks = _parse_blocks(html)
        sections = _extract_section(blocks)
        for section in sections:
            recipe = _collect_recipe_from_section(section, href)
            if recipe:
                recipes.append(recipe)
                if max_recipes and len(recipes) >= max_recipes:
                    return recipes

    return recipes


def _is_crumbs_doilies(epub_path: Path) -> bool:
    name = epub_path.name.lower()
    return "crumbs" in name and "doilies" in name


def _extract_epub_recipes_from_index(
    epub_path: Path, *, max_recipes: int | None = None
) -> List[dict]:
    try:
        candidates = extract_recipes_from_epub_index(epub_path)
    except Exception:
        return []

    recipes: List[dict] = []
    seen_keys: set[str] = set()
    limit = None if max_recipes is None or max_recipes <= 0 else max_recipes

    for candidate in candidates:
        title = (candidate.title or "").strip()
        href = (candidate.href or "").strip()
        if not title or not href:
            continue
        if _title_ignored(title):
            continue
        if href in seen_keys:
            continue

        text = extract_recipe_plaintext(epub_path, href)
        image_href = extract_recipe_image_href(epub_path, href)
        ingredients, method = split_recipe_text(text)
        method_text = (method or "").strip()
        if not method_text:
            continue

        recipes.append(
            {
                "title": title,
                "ingredients_text": "\n".join(ingredients).strip() or None,
                "method_text": method_text,
                "source_type": "epub:index",
                "source_key": href,
                "location_type": "href",
                "location_value": href,
                "image_href": image_href,
            }
        )
        seen_keys.add(href)

        if limit is not None and len(recipes) >= limit:
            break

    return recipes


def _has_class(classes: Iterable[str], *targets: str) -> bool:
    if not classes:
        return False
    normalized = {c.lower().replace("-", "_") for c in classes if c}
    for target in targets:
        if target.lower().replace("-", "_") in normalized:
            return True
    return False


class CrumbsParser(HTMLParser):
    def __init__(self, href: str) -> None:
        super().__init__()
        self.href = href
        self.recipes: List[dict] = []
        self._current: dict | None = None
        self._capture: str | None = None
        self._buf: List[str] = []
        self._ingredient_section: Optional[str] = None
        self._in_tip = False
        self._pending_recipe_id: Optional[str] = None

    def _start_recipe(self, title: str, recipe_id: Optional[str]) -> None:
        if self._current:
            self._finish_recipe()
        self._current = {
            "title": title,
            "id": recipe_id,
            "ingredients": [],
            "method": [],
            "image_href": None,
        }
        self._ingredient_section = None
        self._in_tip = False

    def _finish_recipe(self) -> None:
        if not self._current:
            return
        title = (self._current.get("title") or "").strip()
        recipe_id = self._current.get("id")
        method = self._current.get("method") or []
        if not title or not method:
            self._current = None
            return
        source_key = f"{self.href}#{recipe_id}" if recipe_id else self.href
        ingredients_text = "\n".join(self._current.get("ingredients") or []).strip() or None
        method_text = "\n".join(method).strip()
        self.recipes.append(
            {
                "title": title,
                "ingredients_text": ingredients_text,
                "method_text": method_text,
                "source_type": "epub:crumbs_doilies",
                "source_key": source_key,
                "location_type": "href",
                "location_value": source_key,
                "image_href": self._current.get("image_href"),
            }
        )
        self._current = None

    def _push_text(self, text: str) -> None:
        if not self._current:
            return
        if self._capture == "ingredient_header":
            heading = text.strip()
            if heading:
                self._ingredient_section = heading
        elif self._capture == "ingredient_item":
            line = text.strip()
            if line:
                ingredients = self._current["ingredients"]
                if self._ingredient_section and (
                    not ingredients or ingredients[-1] != self._ingredient_section
                ):
                    ingredients.append(self._ingredient_section)
                ingredients.append(line)
        elif self._capture == "method_heading":
            heading = text.strip()
            if heading and not self._in_tip:
                self._current["method"].append(heading)
        elif self._capture == "method":
            line = text.strip()
            if line and not self._in_tip:
                self._current["method"].append(line)
        elif self._capture == "recipe_title":
            title = text.strip()
            if title:
                self._start_recipe(title, self._pending_recipe_id)

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        attr_map = {k: v for k, v in attrs}
        classes = (attr_map.get("class") or "").split()

        if tag == "h2" and _has_class(classes, "rec_head", "recipe_head", "rec_title", "rec-title"):
            self._capture = "recipe_title"
            self._buf = []
            self._pending_recipe_id = attr_map.get("id")
            return

        if tag == "h5" and _has_class(classes, "ingredient_header", "ingredient-head", "ingredient_title"):
            self._capture = "ingredient_header"
            self._buf = []
            return

        if tag == "li" and _has_class(classes, "ingred", "ingredient", "ingredient_item"):
            self._capture = "ingredient_item"
            self._buf = []
            return

        if tag == "h4" and _has_class(classes, "rec_subhead", "rec-subhead", "method_head"):
            self._in_tip = False
            self._capture = "method_heading"
            self._buf = []
            return

        if tag == "h4" and _has_class(classes, "tip_head", "tip-head", "tip_subhead"):
            self._capture = "method_heading"
            self._buf = []
            self._in_tip = True
            return

        if tag == "p" and (
            _has_class(
                classes,
                "method",
                "method2",
                "rec_intro",
                "rec-intro",
                "no_indent",
                "method_text",
            )
        ):
            self._capture = "method"
            self._buf = []
            return

        if tag == "img" and self._current and not self._current.get("image_href"):
            src = (attr_map.get("src") or "").strip()
            if src and not src.startswith("data:") and "://" not in src:
                base_dir = posixpath.dirname(self.href)
                image_href = posixpath.normpath(posixpath.join(base_dir, src))
                image_href = image_href.lstrip("/")
                self._current["image_href"] = image_href
            return

    def handle_data(self, data: str) -> None:
        if not self._capture:
            return
        self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._capture:
            return
        text = " ".join(" ".join(self._buf).split()).strip()
        if text:
            self._push_text(text)
        self._capture = None
        self._buf = []


def extract_crumbs_doilies_recipes(
    epub_path: Path, *, max_recipes: int | None = None
) -> List[dict]:
    book = epub.read_epub(str(epub_path))
    recipes: List[dict] = []

    for href, html in _iter_spine_docs(book):
        parser = CrumbsParser(href)
        parser.feed(html.decode("utf-8", errors="ignore"))
        parser._finish_recipe()
        for recipe in parser.recipes:
            if _title_ignored(recipe.get("title", "")):
                continue
            recipes.append(recipe)
            if max_recipes and len(recipes) >= max_recipes:
                return recipes

    return recipes
