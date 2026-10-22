"""Localization of the public home page (owner's request 2026-10-21; PodWarden
epic #2134 decisions 4-8). Only Home and the chrome rendered with it are
translated; every other page stays English and is linked from every locale.

English is the single source of truth. A translatable piece of text is a
*segment*, marked in the HTML source where the English is written:

  <h2 data-t="home.install.h2">Where will you run it?</h2>   the element's inner HTML
  <img alt="..." data-t-alt="home.why-fit.img1.alt">           one attribute's value

plus a few strings that live in Python (the <title>, meta description,
JSON-LD featureList, og:image:alt): render.PY_SEGMENTS. The markers are
stripped from every page served, so the English output is byte-identical to
the unmarked source.

Catalogs, all in this directory:

  en.json         the source catalog, written by `scripts/landing_i18n.py
                  write-source` from the marked HTML: {key: {text, note, hash}}
  <locale>.json   one per translated locale: {key: {text, source_hash}}, where
                  source_hash is en.json's hash of the English the
                  translation was made from. A locale with no file is not
                  served (its URL 404s) and not offered anywhere.
  glossary.json   do-not-translate terms and preferred translations.

How to add or refresh a locale: README.md beside this file.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from app.landing.pages import LOCALES, Locale

__all__ = ["LOCALES", "Locale"]

DIR = Path(__file__).parent

# --- Hashing ---------------------------------------------------------------------


def source_hash(text: str) -> str:
    """The hash a translation records of the English it was made from."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --- Catalogs --------------------------------------------------------------------


def _load(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: not a JSON object")
    return data


def _load_catalogs() -> dict[str, dict[str, str]]:
    """locale tag -> {key: translated text}, for every locale with a file.
    Read once at import, like the rest of the site."""
    out: dict[str, dict[str, str]] = {}
    for loc in LOCALES:
        if loc.tag == "en":
            continue
        path = DIR / f"{loc.tag}.json"
        if path.is_file():
            out[loc.tag] = {k: v["text"] for k, v in _load(path).items()}
    return out


_CATALOGS = _load_catalogs()


def available() -> tuple[Locale, ...]:
    """English plus every locale whose catalog ships, in LOCALES order."""
    return tuple(loc for loc in LOCALES if loc.tag == "en" or loc.tag in _CATALOGS)


def is_available(tag: str) -> bool:
    return tag == "en" or tag in _CATALOGS


def text(locale: str, key: str, english: str) -> str:
    """One segment in `locale`; the English when there is no translation."""
    if locale == "en":
        return english
    return _CATALOGS.get(locale, {}).get(key, english)


def locale(tag: str) -> Locale:
    return next(loc for loc in LOCALES if loc.tag == tag)


def home_path(tag: str) -> str:
    return "/" if tag == "en" else f"/{tag}/"


# --- Choosing a language for a visitor of / -----------------------------------------

#: The cookie the switcher sets: "en" or a locale tag.
COOKIE = "lw_lang"
COOKIE_MAX_AGE = 365 * 24 * 3600

#: Crawlers are never redirected: each language has its own URL for them,
#: announced by hreflang and the sitemap.
_BOT_RE = re.compile(
    r"bot\b|bot/|crawl|spider|slurp|google|mediapartners|facebookexternalhit|embedly|preview|"
    r"lighthouse|pagespeed|archiver|bingpreview|yandex|baidu|sogou|petalsearch",
    re.IGNORECASE,
)


def is_bot(user_agent: str | None) -> bool:
    return bool(user_agent) and _BOT_RE.search(user_agent or "") is not None


def _match(language_range: str) -> str | None:
    """The locale a language range asks for: "en", a locale tag, or None
    when we publish nothing for it. Chinese: Simplified (zh, zh-Hans, zh-CN,
    zh-SG) only; Traditional (zh-TW, zh-HK, zh-MO, zh-Hant) is not served
    Simplified. Portuguese: every variant gets pt-BR."""
    parts = language_range.strip().lower().split("-")
    primary, rest = parts[0], set(parts[1:])
    if primary == "en":
        return "en"
    if primary == "zh":
        if "hans" in rest or not rest & {"tw", "hk", "mo", "hant"}:
            return "zh-CN"
        return None
    if primary == "pt":
        return "pt-BR"
    for loc in LOCALES:
        if loc.tag.lower() == primary:
            return loc.tag
    return None


def parse_accept_language(header: str | None) -> list[str]:
    """Language ranges by preference: q descending, header order on ties,
    q=0 ("not acceptable") and malformed q dropped."""
    ranked: list[tuple[float, int, str]] = []
    for i, item in enumerate((header or "").split(",")):
        lang, _, params = item.strip().partition(";")
        lang = lang.strip()
        if not lang or lang == "*":
            continue
        q = 1.0
        for param in params.split(";"):
            name, _, value = param.strip().partition("=")
            if name.strip().lower() == "q":
                try:
                    q = float(value)
                except ValueError:
                    q = 0.0
        if q > 0:
            ranked.append((-q, i, lang))
    return [lang for _, _, lang in sorted(ranked)]


def negotiate(accept_language: str | None, cookie: str | None) -> str:
    """The language for a visitor of the English Home: the switcher's
    cookie when it names a published language, else the first range of
    Accept-Language (the browser's, which follows the OS locale unless the
    visitor changed it) that we publish. English when nothing matches."""
    if cookie and is_available(cookie):
        return cookie
    for lang in parse_accept_language(accept_language):
        tag = _match(lang)
        if tag is not None and is_available(tag):
            return tag
    return "en"


# --- Scanning marked HTML --------------------------------------------------------

#: Elements that never have content or an end tag.
_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "source", "track", "wbr",
}  # fmt: skip
#: Elements a segment may contain: text-level markup the translator keeps.
INLINE = {
    "a", "abbr", "b", "br", "code", "em", "i", "kbd", "small", "span", "strong",
    "sub", "sup", "wbr",
}  # fmt: skip
#: Attributes whose values are read by people (alt) or assistive tech.
TEXT_ATTRS = ("alt", "aria-label", "title")
#: Inside an SVG drawing only these carry text to translate: what a screen
#: reader reads. The drawings' own labels stay English, like the measured
#: figures' (README.md, "What is not translated").
_SVG_TEXT = {"title", "desc"}
_TOKEN_RE = re.compile(r"\{\{[^}]*\}\}")
_ALNUM_RE = re.compile(r"\w", re.UNICODE)
_MARK_RE = re.compile(r' data-t(?:-[a-z][a-z-]*)?="[^"]*"')


@dataclass
class Node:
    tag: str
    attrs: dict[str, str]
    start: int  # offset of "<"
    open_end: int  # offset just after the start tag's ">"
    close_start: int = -1  # offset of the end tag's "<" (-1: void / unclosed)
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    direct_text: list[str] = field(default_factory=list)


class _Scanner(HTMLParser):
    def __init__(self, src: str) -> None:
        super().__init__(convert_charrefs=False)
        self.src = src
        self._lines = [0]
        for m in re.finditer("\n", src):
            self._lines.append(m.end())
        self.root = Node("#root", {}, 0, 0)
        self.stack: list[Node] = [self.root]
        self.tags: dict[int, str] = {}

    def _off(self) -> int:
        line, col = self.getpos()
        return self._lines[line - 1] + col

    def _open(self, tag: str, attrs: list[tuple[str, str | None]], void: bool) -> None:
        raw = self.get_starttag_text() or ""
        at = self._off()
        node = Node(tag, {k: (v or "") for k, v in attrs}, at, at + len(raw))
        self.tags[id(node)] = raw
        node.parent = self.stack[-1]
        node.parent.children.append(node)
        if not void and tag not in _VOID:
            self.stack.append(node)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._open(tag, attrs, False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._open(tag, attrs, True)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack[i].close_start = self._off()
                del self.stack[i:]
                return

    def _text(self, data: str) -> None:
        self.stack[-1].direct_text.append(data)

    def handle_data(self, data: str) -> None:
        self._text(data)

    def handle_entityref(self, name: str) -> None:
        self._text(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        self._text(html.unescape(f"&#{name};"))


def scan(src: str) -> tuple[Node, dict[int, str]]:
    sc = _Scanner(src)
    sc.feed(src)
    sc.close()
    return sc.root, sc.tags


def _walk(node: Node) -> Iterator[Node]:
    for child in node.children:
        yield child
        yield from _walk(child)


def _has_words(s: str) -> bool:
    return bool(_ALNUM_RE.search(_TOKEN_RE.sub("", s)))


@dataclass(frozen=True)
class Segment:
    key: str
    attr: str  # "" for an element's inner HTML, else the attribute name
    start: int
    end: int
    tag: str
    scope: str  # nearest ancestor-or-self id, for the translator's note


def scope_of(node: Node | None) -> str:
    while node is not None:
        if node.attrs.get("id"):
            return node.attrs["id"]
        node = node.parent
    return ""


def _attr_span(raw: str, attr: str) -> tuple[int, int]:
    m = re.search(rf'\s{re.escape(attr)}="([^"]*)"', raw)
    if not m:
        raise ValueError(f"marker data-t-{attr} on a tag without {attr}=: {raw[:80]}")
    return m.start(1), m.end(1)


def segments(src: str) -> list[Segment]:
    """Every marked segment of a document, in document order."""
    root, tags = scan(src)
    out: list[Segment] = []
    for node in _walk(root):
        raw = tags[id(node)]
        for name, key in node.attrs.items():
            if name == "data-t":
                if node.close_start < 0:
                    raise ValueError(f"data-t={key!r} on an element with no end tag")
                out.append(
                    Segment(key, "", node.open_end, node.close_start, node.tag, scope_of(node))
                )
            elif name.startswith("data-t-"):
                attr = name[len("data-t-") :]
                a, b = _attr_span(raw, attr)
                out.append(
                    Segment(key, attr, node.start + a, node.start + b, node.tag, scope_of(node))
                )
    keys = [s.key for s in out]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"segment keys used twice: {dupes}")
    spans = sorted((s.start, s.end, s.key) for s in out)
    for (_a0, a1, ak), (b0, _b1, bk) in zip(spans, spans[1:], strict=False):
        if b0 < a1:
            raise ValueError(f"segment {bk!r} sits inside segment {ak!r}")
    return out


def strip_markers(src: str) -> str:
    return _MARK_RE.sub("", src)


def translate(src: str, locale: str) -> str:
    """The marked document with every segment in `locale` (English where a
    key has no translation), markers removed."""
    if locale != "en":
        cat = _CATALOGS.get(locale, {})
        for seg in sorted(segments(src), key=lambda s: s.start, reverse=True):
            if seg.key in cat:
                src = src[: seg.start] + cat[seg.key] + src[seg.end :]
    return strip_markers(src)


# --- Coverage: text a reader sees that no marker covers ---------------------------


@dataclass(frozen=True)
class Unmarked:
    node: Node
    attr: str  # "" = the element's text
    text: str


def _needs_marker(s: str, dnt: list[str]) -> bool:
    """Words a reader would read in another language: not only do-not-
    translate terms (a tile that says just "Codex CLI"), tokens or
    punctuation. Digits count: their separators are localized."""
    s = _TOKEN_RE.sub("", html.unescape(s))
    for term in sorted(dnt, key=len, reverse=True):
        s = s.replace(term, " ")
    return _has_words(s)


def needs_translation(s: str) -> bool:
    return _needs_marker(s, dnt_terms())


def unmarked(src: str) -> list[Unmarked]:
    """Visible text with no segment: an element whose own text has words but
    that sits in no data-t element, and alt / aria-label / title values with
    words but no data-t-<attr>. Code, styles, scripts, the install command and
    SVG drawings (other than their <title>/<desc>) are not text to translate,
    nor is text inside an element with its own lang= (a language's own name)."""
    root, _ = scan(src)
    found: list[Unmarked] = []
    dnt = dnt_terms()

    def visit(node: Node, covered: bool, in_svg: bool) -> None:
        if node.tag in {"script", "style", "pre", "head", "code"}:
            return
        if node.attrs.get("lang") and node.tag != "html":
            return  # already in its own language: the switcher's native names
        svg = in_svg or node.tag == "svg"
        here = covered or "data-t" in node.attrs
        for attr in TEXT_ATTRS:
            val = node.attrs.get(attr)
            if (
                val
                and _needs_marker(val, dnt)
                and f"data-t-{attr}" not in node.attrs
                and not covered
            ):
                found.append(Unmarked(node, attr, val))
        own = "".join(node.direct_text)
        readable = not svg or node.tag in _SVG_TEXT
        if readable and not here and _needs_marker(own, dnt):
            found.append(Unmarked(node, "", own.strip()))
        for child in node.children:
            visit(child, here, svg)

    for child in root.children:
        visit(child, False, False)
    return found


def segment_root(node: Node) -> Node:
    """The element a new marker for this text belongs on: the text's element,
    or the outermost inline-only ancestor holding it (so "<li>A <a>b</a> c</li>"
    is one segment, not three)."""
    best = node
    cur = node
    while cur.parent is not None and cur.parent.tag != "#root":
        p = cur.parent
        if p.tag in {"svg"} or not all(_inline_only(c) for c in p.children):
            break
        if _has_words("".join(p.direct_text)):
            best = p
            cur = p
            continue
        break
    return best


def _inline_only(node: Node) -> bool:
    return node.tag in INLINE and all(_inline_only(c) for c in node.children)


# --- Placeholders a translation must keep ------------------------------------------

_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_ARG_RE = re.compile(r"(?<!\{)\{[a-z]+\}(?!\})")
_CODE_RE = re.compile(r"<code>(.*?)</code>", re.S)


def placeholders(s: str) -> list[str]:
    """The tags, {{tokens}} and {arguments} of a segment, sorted: a translation may move
    them, never drop, add or alter one."""
    return sorted(_TAG_RE.findall(s) + _TOKEN_RE.findall(s) + _ARG_RE.findall(s))


def code_spans(s: str) -> list[str]:
    return sorted(_CODE_RE.findall(s))


_GROUP_SEP_RE = re.compile(r"(?<=\d)[    ,.'’](?=\d)")


def numbers(s: str) -> list[str]:
    """The digit runs of a segment's visible text, with decimal and thousands
    separators removed, so "2.05" (en) and "2,05" (de), or "11,178" and
    "11 178" (fr), compare equal."""
    plain = html.unescape(_TOKEN_RE.sub("", _TAG_RE.sub(" ", s)))
    return sorted(re.findall(r"\d+", _GROUP_SEP_RE.sub("", plain)))


# --- The glossary ------------------------------------------------------------------


def glossary() -> dict[str, object]:
    data: dict[str, object] = json.loads((DIR / "glossary.json").read_text(encoding="utf-8"))
    return data


def dnt_terms() -> list[str]:
    terms: list[str] = list(glossary()["do_not_translate"])  # type: ignore[call-overload]
    return terms


def term_in(term: str, s: str) -> bool:
    """`term` as a whole word (or run of words) in `s`."""
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w])", s) is not None


# --- Checking a locale catalog ------------------------------------------------------

#: Finding kinds, in the order the check reports them.
PROBLEM_KINDS = (
    "missing",  # a source key with no translation
    "stale",  # source_hash is not the current English's hash
    "unknown",  # a translated key the source no longer has
    "placeholders",  # a tag or {{token}} / {n} dropped, added or altered
    "code",  # a <code> span not kept verbatim
    "numbers",  # digits differ (separators may follow the locale)
    "glossary",  # a do-not-translate term in the English missing from the translation
    "untranslated",  # identical to the English, not allowlisted
    "attribute",  # an attribute translation that could break out of its quotes
)


def catalog_problems(
    locale: str, catalog: dict[str, dict[str, str]], source: dict[str, dict[str, str]]
) -> dict[str, list[str]]:
    """Every finding for one locale's catalog against the source catalog,
    by kind (PROBLEM_KINDS). Empty lists everywhere: the catalog is good."""
    out: dict[str, list[str]] = {kind: [] for kind in PROBLEM_KINDS}
    g = glossary()
    dnt: list[str] = list(g["do_not_translate"])  # type: ignore[call-overload]
    same_ok = set(g.get("identical_ok", {}).get(locale, []))  # type: ignore[attr-defined]
    digits_ok = set(g.get("numbers_ok", {}).get(locale, []))  # type: ignore[attr-defined]
    term_ok: dict[str, list[str]] = g.get("glossary_ok", {})  # type: ignore[assignment]
    for key, seg in source.items():
        if key not in catalog:
            out["missing"].append(key)
            continue
        tr = catalog[key]
        if tr.get("source_hash") != seg["hash"]:
            out["stale"].append(key)
        en, text = seg["text"], tr["text"]
        if placeholders(en) != placeholders(text):
            out["placeholders"].append(key)
        if code_spans(en) != code_spans(text):
            out["code"].append(key)
        if numbers(en) != numbers(text) and key not in digits_ok:
            out["numbers"].append(key)
        lost = [
            t for t in dnt if term_in(t, en) and t not in text and t not in term_ok.get(key, [])
        ]
        if lost:
            out["glossary"].append(f"{key}: {', '.join(lost)}")
        if text == en and key not in same_ok and needs_translation(en):
            out["untranslated"].append(key)
        if _is_attribute(key) and ('"' in text or "<" in text):
            out["attribute"].append(key)
    out["unknown"] = sorted(set(catalog) - set(source))
    return out


def _is_attribute(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in {
        *TEXT_ATTRS,
        "resume",
        "dot",
        "done",
        "fail",
    } or key.startswith("meta.")
