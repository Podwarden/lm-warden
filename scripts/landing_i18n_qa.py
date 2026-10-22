#!/usr/bin/env python3
"""Translation QA for the public website's catalogs (app/landing/i18n/README.md,
"Translation QA").

The method: a blind back-translation (the agent sees only the target text),
a monolingual native-editor review (target only), a bilingual meaning review
(English vs back-translation), fixes, a blind re-check of the changed
segments, then an MQM-style score. This script moves the text between those
steps and records the result; the reviews themselves are agent briefs in the
README.

  extract LOCALE --out DIR [--keys FILE]
                      DIR/<LOCALE>.target.json ({key: target text}) and
                      DIR/<LOCALE>.source.json ({key: English}) for every
                      published segment of LOCALE (home + each complete
                      part). --keys (a JSON list, or an object whose keys
                      count, e.g. a fixes file) narrows it to those keys:
                      the re-verifier's input.
  diff LOCALE --back FILE [--back FILE...] --out FILE.md
                      side-by-side for the meaning reviewer: key, English,
                      target, back-translation, back-translator notes, as
                      Markdown (FILE.md) and JSON (FILE.json). Later --back
                      files override earlier ones key by key.
  apply LOCALE --fixes FILE
                      write {key: new target text} into the right catalog
                      file (home or part) with source_hash set to the
                      current English; refuses the whole set when a key is
                      unknown or a fix breaks tags, code, numbers or
                      glossary terms. Then runs landing_i18n.py check LOCALE.
  record LOCALE --issues FILE --back FILE [--back FILE...]
                      [--back-model NAME] [--fix-model NAME]
                      write app/landing/i18n/qa/<LOCALE>/report.json (date,
                      the source and target hashes covered, the issues,
                      the MQM score, pass/fail) and report.md.
  status              per locale: published segments, QA-covered, stale
                      (text changed since QA), score, pass.

Back-translation file: {key: "literal English"} plus optional
{"<key>#note": "why the target reads oddly"}; {key: {"text": ..., "note":
...}} also works. Issues file: a JSON list of {key, category, severity
(minor|major|critical), status (open|fixed|accepted), note}.

Pass rule: no open/accepted critical or major issue, and an MQM score
(minor 1, major 5, critical 25 penalty points per 1,000 English words
covered) of at most 10. Fixed issues are listed but do not score.

Run from the repository root with the project's Python:
  ../../.venv/bin/python scripts/landing_i18n_qa.py status
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import landing_i18n  # noqa: E402

from app.landing import i18n  # noqa: E402
from app.landing.pages import LOCALES  # noqa: E402

ALL_PARTS: tuple[str, ...] = ("home", *i18n.PARTS)
QA_DIR = i18n.DIR / "qa"
SEVERITY_POINTS = {"minor": 1, "major": 5, "critical": 25}
STATUSES = ("open", "fixed", "accepted")
CATEGORIES = (
    "omission",
    "addition",
    "mistranslation",
    "number",
    "untranslated",
    "terminology",
    "fluency",
    "grammar",
    "style",
    "locale-convention",
)
MAX_SCORE = 10.0
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*")
_TAG_RE = re.compile(r"<[^>]+>")


def _dump(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _locales() -> list[str]:
    return [loc.tag for loc in LOCALES if loc.tag != "en"]


def _catalog(part: str, loc: str) -> dict[str, dict[str, str]]:
    path = i18n.catalog_path(part, loc)
    if not path.is_file():
        return {}
    data: dict[str, dict[str, str]] = _read_json(path)
    return data


def _published_parts(loc: str) -> list[str]:
    return [p for p in ALL_PARTS if i18n.part_ships(p, loc)]


@dataclass(frozen=True)
class Segment:
    key: str
    part: str
    english: str
    note: str
    source_hash: str  # hash of the current English
    target: str

    @property
    def target_hash(self) -> str:
        return i18n.source_hash(self.target)


def published(loc: str) -> dict[str, Segment]:
    """Every segment LOCALE serves, in part and source order."""
    out: dict[str, Segment] = {}
    for part in _published_parts(loc):
        cat = _catalog(part, loc)
        for key, src in landing_i18n._source(part).items():
            if key in cat:
                out[key] = Segment(
                    key, part, src["text"], src.get("note", ""), src["hash"], cat[key]["text"]
                )
    return out


def english_words(text: str) -> int:
    return len(_WORD_RE.findall(_TAG_RE.sub(" ", text)))


def mqm_score(issues: list[dict[str, Any]], words: int) -> float:
    """Penalty points of the issues still in the text per 1,000 English words."""
    points = sum(SEVERITY_POINTS[i["severity"]] for i in issues if i["status"] != "fixed")
    return round(points * 1000 / max(words, 1), 2)


def _load_back(paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """(key -> back-translation, key -> note) merged over PATHS in order."""
    back: dict[str, str] = {}
    notes: dict[str, str] = {}
    for path in paths:
        data = _read_json(path)
        if not isinstance(data, dict):
            raise SystemExit(f"{path}: not a JSON object")
        for key, val in data.items():
            if key.endswith("#note"):
                notes[key.removesuffix("#note")] = str(val)
            elif isinstance(val, dict):
                back[key] = str(val.get("text", ""))
                if val.get("note"):
                    notes[key] = str(val["note"])
            else:
                back[key] = str(val)
    return back, notes


# --- extract -----------------------------------------------------------------------


def extract(loc: str, out: Path, keys_file: Path | None) -> int:
    segs = published(loc)
    if not segs:
        print(f"{loc}: nothing published", file=sys.stderr)
        return 1
    if keys_file is not None:
        wanted = set(_read_json(keys_file))
        unknown = sorted(wanted - set(segs))
        if unknown:
            print(f"{loc}: not published: {', '.join(unknown)}", file=sys.stderr)
            return 1
        segs = {k: s for k, s in segs.items() if k in wanted}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{loc}.target.json").write_text(
        _dump({k: s.target for k, s in segs.items()}), encoding="utf-8"
    )
    (out / f"{loc}.source.json").write_text(
        _dump({k: s.english for k, s in segs.items()}), encoding="utf-8"
    )
    words = sum(english_words(s.english) for s in segs.values())
    print(f"{loc}: {len(segs)} segments ({words} English words) -> {out}")
    return 0


# --- diff --------------------------------------------------------------------------


def _md_quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines() or [""])


def diff(loc: str, backs: list[Path], out: Path) -> int:
    segs = published(loc)
    back, notes = _load_back(backs)
    rows = []
    lines = [
        f"# {loc}: English vs blind back-translation",
        "",
        f"{len(back)} of {len(segs)} published segments back-translated.",
        "",
    ]
    missing = [k for k in segs if k not in back]
    if missing:
        lines += [f"Not back-translated ({len(missing)}): {', '.join(missing)}", ""]
    for key, seg in segs.items():
        if key not in back:
            continue
        row = {
            "key": key,
            "english": seg.english,
            "target": seg.target,
            "back": back[key],
            "note": notes.get(key, ""),
            "context": seg.note,
        }
        rows.append(row)
        lines += [f"## `{key}`", ""]
        if seg.note:
            lines += [f"*Context: {seg.note}*", ""]
        lines += ["**English**", "", _md_quote(seg.english), ""]
        lines += [f"**Target ({loc})**", "", _md_quote(seg.target), ""]
        lines += ["**Back-translation**", "", _md_quote(back[key]), ""]
        if key in notes:
            lines += [f"**Back-translator note:** {notes[key]}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(_dump(rows), encoding="utf-8")
    print(f"{loc}: {len(rows)} segments -> {out} and {out.with_suffix('.json')}")
    return 0


# --- apply -------------------------------------------------------------------------


def apply(loc: str, fixes_file: Path) -> int:
    fixes = _read_json(fixes_file)
    if not isinstance(fixes, dict) or not all(isinstance(v, str) for v in fixes.values()):
        print(f"{fixes_file}: want a JSON object {{key: new text}}", file=sys.stderr)
        return 2
    by_part: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for key, new in fixes.items():
        try:
            part = i18n.part_of(key)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if key not in landing_i18n._source(part):
            errors.append(f"{key}: not in the {part} source")
        elif key not in _catalog(part, loc):
            errors.append(f"{key}: no {loc} translation to fix ({part})")
        else:
            by_part.setdefault(part, {})[key] = new
    for part, part_fixes in by_part.items():
        src = landing_i18n._source(part)
        trial = {k: {"text": t, "source_hash": src[k]["hash"]} for k, t in part_fixes.items()}
        found = i18n.catalog_problems(loc, trial, {k: src[k] for k in part_fixes})
        for kind, keys in found.items():
            if kind not in {"missing", "stale", "unknown"} and keys:
                errors.append(f"{kind}: {', '.join(keys)}")
    if errors:
        print("refused, nothing written:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    for part, part_fixes in by_part.items():
        src = landing_i18n._source(part)
        cat = _catalog(part, loc)
        for key, new in part_fixes.items():
            cat[key] = {"text": new, "source_hash": src[key]["hash"]}
        path = i18n.catalog_path(part, loc)
        path.write_text(_dump(cat), encoding="utf-8")
        print(f"{_rel(path)}: {len(part_fixes)} segments updated")
    return int(landing_i18n.check([loc], list(ALL_PARTS)))


# --- record ------------------------------------------------------------------------


def _validate_issues(issues: object, keys: set[str]) -> list[str]:
    if not isinstance(issues, list):
        return ["issues: want a JSON list"]
    errs = []
    for n, issue in enumerate(issues):
        if not isinstance(issue, dict):
            errs.append(f"issue {n}: not an object")
            continue
        if issue.get("key") not in keys:
            errs.append(f"issue {n}: key {issue.get('key')!r} is not a published segment")
        if issue.get("severity") not in SEVERITY_POINTS:
            errs.append(f"issue {n}: severity must be one of {', '.join(SEVERITY_POINTS)}")
        if issue.get("status") not in STATUSES:
            errs.append(f"issue {n}: status must be one of {', '.join(STATUSES)}")
        if issue.get("category") not in CATEGORIES:
            errs.append(f"issue {n}: category must be one of {', '.join(CATEGORIES)}")
    return errs


def report_path(loc: str) -> Path:
    return QA_DIR / loc / "report.json"


def record(
    loc: str,
    issues_file: Path,
    backs: list[Path],
    back_model: str | None,
    fix_model: str | None,
) -> int:
    if back_model and fix_model and back_model == fix_model:
        print("the back-translator must be a different model from the fixer", file=sys.stderr)
        return 2
    segs = published(loc)
    back, notes = _load_back(backs)
    covered = {k: s for k, s in segs.items() if k in back}
    issues = _read_json(issues_file)
    errs = _validate_issues(issues, set(segs))
    if errs:
        print("refused:", *errs, sep="\n  ", file=sys.stderr)
        return 1
    stale_tr = [
        k for k, s in segs.items() if _catalog(s.part, loc)[k]["source_hash"] != s.source_hash
    ]
    words = sum(english_words(s.english) for s in covered.values())
    remaining = [i for i in issues if i["status"] != "fixed"]
    count = {sev: sum(1 for i in remaining if i["severity"] == sev) for sev in SEVERITY_POINTS}
    found = {sev: sum(1 for i in issues if i["severity"] == sev) for sev in SEVERITY_POINTS}
    score = mqm_score(issues, words)
    passed = (
        count["critical"] == 0
        and count["major"] == 0
        and score <= MAX_SCORE
        and len(covered) == len(segs)
        and not stale_tr
    )
    report = {
        "locale": loc,
        "date": dt.date.today().isoformat(),
        "back_model": back_model,
        "fix_model": fix_model,
        "published_segments": len(segs),
        "covered_segments": len(covered),
        "english_words": words,
        "issues_found": found,
        "issues_remaining": count,
        "mqm_score": score,
        "max_score": MAX_SCORE,
        "pass": passed,
        "segments": {
            k: {"part": s.part, "source_hash": s.source_hash, "target_hash": s.target_hash}
            for k, s in covered.items()
        },
        "back_notes": {k: notes[k] for k in covered if k in notes},
        "issues": issues,
    }
    path = report_path(loc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump(report), encoding="utf-8")
    md = [
        f"# Translation QA: {loc}",
        "",
        f"- Date: {report['date']}",
        f"- Back-translation model: {back_model or 'not recorded'}; fixer: {fix_model or 'not recorded'}",
        f"- Segments covered: {len(covered)} of {len(segs)} published ({words} English words)",
        "- Issues found (critical/major/minor): "
        + "/".join(str(found[s]) for s in ("critical", "major", "minor")),
        "- Issues remaining (critical/major/minor): "
        + "/".join(str(count[s]) for s in ("critical", "major", "minor")),
        f"- MQM score: {score} per 1,000 words (pass at <= {MAX_SCORE}, no critical or major)",
        f"- Result: **{'PASS' if passed else 'FAIL'}**",
        "",
    ]
    if len(covered) < len(segs):
        md += [f"Not back-translated: {', '.join(k for k in segs if k not in covered)}", ""]
    if stale_tr:
        md += [f"Stale translations (English changed): {', '.join(stale_tr)}", ""]
    if remaining:
        md += ["## Remaining issues", "", "| Key | Category | Severity | Status | Note |"]
        md += ["|---|---|---|---|---|"]
        for i in remaining:
            note = str(i.get("note", "")).replace("|", "\\|").replace("\n", " ")
            md.append(
                f"| `{i['key']}` | {i['category']} | {i['severity']} | {i['status']} | {note} |"
            )
        md.append("")
    path.with_name("report.md").write_text("\n".join(md), encoding="utf-8")
    print(
        f"{_rel(path)}: {len(covered)}/{len(segs)} segments, "
        f"score {score}, {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


# --- status ------------------------------------------------------------------------


@dataclass(frozen=True)
class QaState:
    locale: str
    published: int
    covered: int  # published segments whose source and target match the report
    stale: list[str]  # published segments QA'd before the text changed
    uncovered: list[str]  # published segments the report never saw
    score: float | None
    passed: bool | None  # None: no report

    @property
    def ok(self) -> bool:
        return bool(self.passed) and not self.stale and not self.uncovered


def qa_state(loc: str) -> QaState:
    segs = published(loc)
    path = report_path(loc)
    if not path.is_file():
        return QaState(loc, len(segs), 0, [], list(segs), None, None)
    report = _read_json(path)
    done: dict[str, dict[str, str]] = report.get("segments", {})
    stale, uncovered = [], []
    for key, seg in segs.items():
        rec = done.get(key)
        if rec is None:
            uncovered.append(key)
        elif rec["source_hash"] != seg.source_hash or rec["target_hash"] != seg.target_hash:
            stale.append(key)
    covered = len(segs) - len(stale) - len(uncovered)
    return QaState(
        loc, len(segs), covered, stale, uncovered, report.get("mqm_score"), report.get("pass")
    )


def status() -> int:
    print(
        f"{'locale':<7} {'published':>9} {'covered':>8} {'stale':>6} {'new':>5} {'score':>6}  result"
    )
    for loc in _locales():
        st = qa_state(loc)
        score = "-" if st.score is None else f"{st.score:g}"
        if st.passed is None:
            result = "no report" if st.published else "-"
        elif st.ok:
            result = "pass"
        elif not st.passed:
            result = "FAIL"
        else:
            result = "re-run QA"
        print(
            f"{loc:<7} {st.published:>9} {st.covered:>8} {len(st.stale):>6} "
            f"{len(st.uncovered):>5} {score:>6}  {result}"
        )
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="landing_i18n_qa.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("locale")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keys", type=Path)
    p = sub.add_parser("diff")
    p.add_argument("locale")
    p.add_argument("--back", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("apply")
    p.add_argument("locale")
    p.add_argument("--fixes", type=Path, required=True)
    p = sub.add_parser("record")
    p.add_argument("locale")
    p.add_argument("--issues", type=Path, required=True)
    p.add_argument("--back", type=Path, action="append", required=True)
    p.add_argument("--back-model")
    p.add_argument("--fix-model")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd != "status" and a.locale not in _locales():
        ap.error(f"unknown locale {a.locale!r} (one of {', '.join(_locales())})")
    if a.cmd == "extract":
        return extract(a.locale, a.out, a.keys)
    if a.cmd == "diff":
        return diff(a.locale, a.back, a.out)
    if a.cmd == "apply":
        return apply(a.locale, a.fixes)
    if a.cmd == "record":
        return record(a.locale, a.issues, a.back, a.back_model, a.fix_model)
    return status()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
