#!/usr/bin/env python3
"""The public home page's translation pipeline (app/landing/i18n/README.md).

  write-source        write app/landing/i18n/en.json from the marked English
                      source (home.html, layout.html, render.py). Run after
                      every English edit; tests fail while it is out of date.
  check [LOCALE...]   per locale: missing, stale (English changed since the
                      translation), unknown keys, and broken placeholders.
                      Exit 1 on any finding.
  todo LOCALE         the segments LOCALE still needs, as JSON on stdout:
                      {key: {text, note, hash}} -- the hand-off to a
                      translation pass, which answers with {key: {text,
                      source_hash: hash}} entries for app/landing/i18n/LOCALE.json.
  unmarked FILE       visible English in FILE with no data-t marker.
  mark FILE PREFIX    add data-t markers to FILE's unmarked text, keyed
                      PREFIX.<section id>.<tag><n>. A drafting aid: review
                      the keys it picks before committing.

Run from the repository root with the project's Python:
  ../../.venv/bin/python scripts/landing_i18n.py check
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


def _dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_source() -> int:
    from app.landing.render import source_catalog

    path = i18n.DIR / "en.json"
    new = _dump(source_catalog())
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if new == old:
        print(f"{path.relative_to(ROOT)}: up to date")
        return 0
    path.write_text(new, encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: written ({len(json.loads(new))} segments)")
    return 0


def problems(locale: str) -> dict[str, list[str]]:
    """Every finding for one locale's catalog, by kind."""
    from app.landing.render import source_catalog

    path = i18n.DIR / f"{locale}.json"
    cat = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return i18n.catalog_problems(locale, cat, source_catalog())


def check(locales: list[str]) -> int:
    from app.landing.render import source_catalog

    src = source_catalog()
    en_path = i18n.DIR / "en.json"
    bad = 0
    if not en_path.exists() or json.loads(en_path.read_text(encoding="utf-8")) != src:
        print("en.json: out of date -- run: scripts/landing_i18n.py write-source")
        bad = 1
    tags = locales or [loc.tag for loc in LOCALES if loc.tag != "en"]
    for tag in tags:
        if not (i18n.DIR / f"{tag}.json").exists():
            print(f"{tag}: no catalog (not served)")
            continue
        found = {k: v for k, v in problems(tag).items() if v}
        if not found:
            print(f"{tag}: {len(src)} segments, complete")
            continue
        bad = 1
        print(f"{tag}:")
        for kind, keys in found.items():
            print(
                f"  {kind} ({len(keys)}): {', '.join(keys[:12])}{' ...' if len(keys) > 12 else ''}"
            )
    return bad


def todo(locale: str) -> int:
    from app.landing.render import source_catalog

    src = source_catalog()
    p = problems(locale)
    want = set(p["missing"]) | set(p["stale"])
    print(_dump({k: v for k, v in src.items() if k in want}), end="")
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
        return check(args)
    if cmd == "todo" and len(args) == 1:
        return todo(args[0])
    if cmd == "unmarked" and len(args) == 1:
        return show_unmarked(Path(args[0]))
    if cmd == "mark" and len(args) == 2:
        return mark(Path(args[0]), args[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
