#!/usr/bin/env python3
"""The public website's translation pipeline (app/landing/i18n/README.md).

The text is split into *parts* (app.landing.i18n.PARTS plus "home"): Home
and the chrome of every page ("home"), what every other page shares
("common"), one per page ("faq", "features", ...) and the generated
coding-agent pages ("clients"). Each command takes an optional part; without
one it covers every part.

  write-source        write every part's source catalog (app/landing/i18n/
                      en.json for "home", en/<part>.json for the others) from
                      the marked English. Run after every English edit; tests
                      fail while one is out of date.
  check [LOCALE...] [--part PART]
                      per locale and part: missing, stale (English changed
                      since the translation), unknown keys, broken
                      placeholders, code, numbers, glossary terms, untranslated
                      text. Exit 1 on any finding. A part with no file yet is
                      reported, not a failure (the page stays English there).
  todo LOCALE [PART]  the segments LOCALE still needs, as JSON on stdout:
                      {key: {text, note, hash}} -- the hand-off to a
                      translation pass, which answers with {key: {text,
                      source_hash: hash}} entries in that part's file
                      (i18n.catalog_path: <LOCALE>.json for "home",
                      <LOCALE>/<part>.json otherwise). Without PART, every
                      part's, merged (keys say their part).
  pages               which pages each locale publishes, and which parts hold
                      the others back.
  unmarked FILE       visible English in FILE with no data-t marker.
  mark FILE PREFIX    add data-t markers to FILE's unmarked text, keyed
                      PREFIX.<section id>.<tag><n>. A drafting aid: review
                      the keys it picks before committing.

Run from the repository root with the project's Python:
  ../../.venv/bin/python scripts/landing_i18n.py check ru
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.landing import i18n  # noqa: E402
from app.landing.pages import LOCALES  # noqa: E402

ALL_PARTS = ("home", *i18n.PARTS)


def _dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _source(part: str) -> dict[str, dict[str, str]]:
    from app.landing.render import source_catalog

    return source_catalog(part)


def write_source() -> int:
    for part in ALL_PARTS:
        path = i18n.source_path(part)
        new = _dump(_source(part))
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if new == old:
            print(f"{path.relative_to(ROOT)}: up to date")
            continue
        path.parent.mkdir(exist_ok=True)
        path.write_text(new, encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: written ({len(json.loads(new))} segments)")
    return 0


def problems(locale: str, part: str = "home") -> dict[str, list[str]]:
    """Every finding for one locale's catalog of one part, by kind."""
    path = i18n.catalog_path(part, locale)
    cat = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return i18n.catalog_problems(locale, cat, _source(part))


def check(locales: list[str], parts: list[str]) -> int:
    bad = 0
    for part in parts:
        path = i18n.source_path(part)
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != _source(part):
            print(
                f"{path.relative_to(ROOT)}: out of date -- run: scripts/landing_i18n.py write-source"
            )
            bad = 1
    tags = locales or [loc.tag for loc in LOCALES if loc.tag != "en"]
    for tag in tags:
        absent = [p for p in parts if not i18n.catalog_path(p, tag).exists()]
        if absent:
            n = sum(len(_source(p)) for p in absent)
            print(
                f"{tag}: no catalog for {', '.join(absent)} ({n} segments; those pages not served)"
            )
        for part in parts:
            n = len(_source(part))
            if part in absent:
                continue
            found = {k: v for k, v in problems(tag, part).items() if v}
            if not found:
                print(f"{tag} {part}: {n} segments, complete")
                continue
            bad = 1
            print(f"{tag} {part}:")
            for kind, keys in found.items():
                more = " ..." if len(keys) > 12 else ""
                print(f"  {kind} ({len(keys)}): {', '.join(keys[:12])}{more}")
    return bad


def todo(locale: str, parts: list[str]) -> int:
    out: dict[str, dict[str, str]] = {}
    for part in parts:
        p = problems(locale, part)
        want = set(p["missing"]) | set(p["stale"])
        out.update({k: v for k, v in _source(part).items() if k in want})
    print(_dump(out), end="")
    return 0


def pages() -> int:
    from app.landing.clients import CLIENT_PAGES
    from app.landing.pages import NOT_FOUND, PAGES

    for loc in LOCALES:
        if loc.tag == "en":
            continue
        shipped, held = [], Counter[str]()
        for page in (*PAGES, *CLIENT_PAGES.values(), NOT_FOUND):
            if i18n.page_ships(page, loc.tag):
                shipped.append(page.path)
            else:
                held.update(p for p in i18n.page_parts(page) if not i18n.part_ships(p, loc.tag))
        print(f"{loc.tag}: {len(shipped)} pages published", end="")
        print(f"; held back by {', '.join(sorted(held))}" if held else "")
    return 0


def _unmarked(path: Path) -> list[i18n.Unmarked]:
    return i18n.unmarked(path.read_text(encoding="utf-8"))


def show_unmarked(path: Path) -> int:
    found = _unmarked(path)
    for u in found:
        where = f"@{u.attr}" if u.attr else ""
        print(f"<{u.node.tag}>{where} #{i18n.scope_of(u.node)}: {u.text[:70]!r}")
    return 1 if found else 0


def mark(path: Path, prefix: str) -> int:
    src = path.read_text(encoding="utf-8")
    existing = {s.key for s in i18n.segments(src)}
    counts: Counter[str] = Counter()
    inserts: dict[int, str] = {}  # offset of a start tag's ">" or "/>" -> markers
    text_roots: dict[int, i18n.Node] = {}
    attr_nodes: dict[int, tuple[i18n.Node, list[str]]] = {}
    for u in i18n.unmarked(src):
        if u.attr:
            attr_nodes.setdefault(u.node.start, (u.node, []))[1].append(u.attr)
        else:
            node = i18n.segment_root(u.node)
            text_roots.setdefault(node.start, node)
    for start in sorted(set(text_roots) | set(attr_nodes)):
        node = text_roots.get(start) or attr_nodes[start][0]
        scope = re.sub(r"-title$", "", i18n.scope_of(node)) or "top"
        base = f"{prefix}.{scope}.{node.tag}"
        counts[base] += 1
        key = f"{base}-{counts[base]}"
        while key in existing:
            counts[base] += 1
            key = f"{base}-{counts[base]}"
        existing.add(key)
        marks = []
        if start in text_roots:
            marks.append(f' data-t="{key}"')
        for attr in attr_nodes.get(start, (None, []))[1]:
            marks.append(f' data-t-{attr}="{key}.{attr}"')
        if marks:
            end = node.open_end - (2 if src[node.open_end - 2 : node.open_end] == "/>" else 1)
            inserts[end] = "".join(marks)
    for at in sorted(inserts, reverse=True):
        src = src[:at] + inserts[at] + src[at:]
    path.write_text(src, encoding="utf-8")
    print(f"{path}: {sum(m.count('data-t') for m in inserts.values())} markers added")
    return 0


def _inner_text(src: str, node: i18n.Node) -> str:
    if node.close_start < 0:
        return ""
    return re.sub(r"<[^>]+>", " ", src[node.open_end : node.close_start])


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "write-source":
        return write_source()
    if cmd == "check":
        parts = list(ALL_PARTS)
        if "--part" in args:
            i = args.index("--part")
            parts, args = [args[i + 1]], args[:i] + args[i + 2 :]
        return check(args, parts)
    if cmd == "todo" and len(args) in (1, 2):
        return todo(args[0], args[1:] or list(ALL_PARTS))
    if cmd == "pages" and not args:
        return pages()
    if cmd == "unmarked" and len(args) == 1:
        return show_unmarked(Path(args[0]))
    if cmd == "mark" and len(args) == 2:
        return mark(Path(args[0]), args[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
