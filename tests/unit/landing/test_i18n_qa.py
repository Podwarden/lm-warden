"""Translation QA gate (app/landing/i18n/README.md, "Translation QA"; owner,
2026-10-21: "translate back and forth, see if back translation shows the same
meaning, use other approaches to control quality, for all languages").

A locale with app/landing/i18n/qa/<loc>/report.json must have passed and its
report must cover every segment it publishes, at the current English and the
current translation. A published locale without a report only warns while QA
is rolled out: set REQUIRE_QA_REPORT = True to make the report mandatory."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

from app.landing import i18n
from app.landing.pages import LOCALES

#: The one-line switch: True once every published locale has been through QA.
REQUIRE_QA_REPORT = False

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "landing_i18n_qa.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("landing_i18n_qa", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["landing_i18n_qa"] = mod
    spec.loader.exec_module(mod)
    return mod


qa = _load()
TAGS = [loc.tag for loc in LOCALES if loc.tag != "en"]


@pytest.mark.parametrize("tag", TAGS)
def test_published_locale_passed_translation_qa(tag: str) -> None:
    st = qa.qa_state(tag)
    if not st.published:
        return
    if st.passed is None:
        msg = f"{tag}: no translation QA report -- re-run translation QA for {tag}"
        if REQUIRE_QA_REPORT:
            pytest.fail(msg)
        warnings.warn(msg, stacklevel=1)
        return
    problems = []
    if not st.passed:
        problems.append(f"the report failed (score {st.score})")
    if st.stale:
        problems.append(f"{len(st.stale)} segments changed since QA: {', '.join(st.stale[:8])}")
    if st.uncovered:
        problems.append(
            f"{len(st.uncovered)} published segments never QA'd: {', '.join(st.uncovered[:8])}"
        )
    assert not problems, f"re-run translation QA for {tag}: " + "; ".join(problems)


def test_mqm_score_counts_only_what_is_left() -> None:
    issues = [
        {"severity": "minor", "status": "open"},
        {"severity": "major", "status": "fixed"},
        {"severity": "critical", "status": "accepted"},
    ]
    assert qa.mqm_score(issues, 2000) == 13.0
    assert qa.english_words("Run <code>lm-warden serve</code> on your GPU.") == 6


def _sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    tag = next(t for t in TAGS if i18n.part_ships("home", t))
    copy = tmp_path / f"{tag}.json"
    shutil.copy(i18n.catalog_path("home", tag), copy)
    real = i18n.catalog_path
    monkeypatch.setattr(
        i18n, "catalog_path", lambda part, t: copy if (part, t) == ("home", tag) else real(part, t)
    )
    return tag, copy


def test_apply_writes_a_fix_with_the_current_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag, copy = _sandbox(tmp_path, monkeypatch)
    src = json.loads(i18n.source_path("home").read_text(encoding="utf-8"))
    key = next(k for k, v in src.items() if k.startswith("home.") and "<" not in v["text"])
    new = qa.published(tag)[key].target + " (QA)"
    fixes = tmp_path / "fixes.json"
    fixes.write_text(json.dumps({key: new}), encoding="utf-8")
    qa.apply(tag, fixes)
    got = json.loads(copy.read_text(encoding="utf-8"))[key]
    assert got == {"text": new, "source_hash": src[key]["hash"]}


@pytest.mark.parametrize(
    "fix",
    [{"meta.title": "<b>no brand"}, {"home.no-such-key": "x"}, {"nopart.x": "x"}],
)
def test_apply_refuses_bad_fixes_and_writes_nothing(
    fix: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tag, copy = _sandbox(tmp_path, monkeypatch)
    before = copy.read_text(encoding="utf-8")
    fixes = tmp_path / "fixes.json"
    fixes.write_text(json.dumps(fix), encoding="utf-8")
    assert qa.apply(tag, fixes) == 1
    assert copy.read_text(encoding="utf-8") == before
