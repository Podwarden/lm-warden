"""The downloadable session report (analysis Task 4.4)."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from app.lens import report
from tests.conftest import bearer
from tests.unit.lens import seed
from tests.unit.lens.test_api import ADM, KEY_A, ME, _a, _admin
from tests.unit.lens.test_routes_ui import (  # noqa: F401
    A,
    B,
    Env,
    env,
    put_toolset,
    req,
    score_all,
    sid,
)


def validate(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$") -> None:
    """The subset of JSON Schema 2020-12 the report schema uses (no jsonschema in the venv)."""
    if "$ref" in schema:
        node: Any = root
        for part in schema["$ref"][2:].split("/"):
            node = node[part]
        return validate(value, node, root, path)
    if "anyOf" in schema:
        errs = []
        for sub in schema["anyOf"]:
            try:
                return validate(value, sub, root, path)
            except AssertionError as e:
                errs.append(str(e))
        raise AssertionError(f"{path}: no anyOf branch matched: {errs}")
    if "const" in schema:
        assert value == schema["const"], f"{path}: {value!r} != {schema['const']!r}"
    if "enum" in schema:
        assert value in schema["enum"], f"{path}: {value!r} not in {schema['enum']}"
    types = schema.get("type")
    if types is not None:
        types = [types] if isinstance(types, str) else types
        py = {
            "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
            "integer": int, "number": (int, float),
        }  # fmt: skip
        ok = any(
            isinstance(value, py[t])
            and not (t in ("integer", "number") and isinstance(value, bool))
            for t in types
        )
        assert ok, f"{path}: {value!r} is not {types}"
    if isinstance(value, str) and "pattern" in schema:
        assert re.search(schema["pattern"], value), f"{path}: {value!r} !~ {schema['pattern']}"
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema:
            assert value >= schema["minimum"], path
        if "maximum" in schema:
            assert value <= schema["maximum"], path
    if isinstance(value, dict):
        for k in schema.get("required", []):
            assert k in value, f"{path}: missing {k}"
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                validate(v, props[k], root, f"{path}.{k}")
            elif extra is False:
                raise AssertionError(f"{path}: unexpected {k}")
            elif isinstance(extra, dict):
                validate(v, extra, root, f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            validate(v, schema["items"], root, f"{path}[{i}]")


def jreport(env: Env, who: str = "a", **q: str) -> dict[str, Any]:  # noqa: F811
    qs = "&".join(f"{k}={v}" for k, v in q.items())
    r = _a(env, f"/sessions/{sid(env, A)}/report?format=json&{qs}")
    assert r.status_code == 200, r.text
    return json.loads(r.text)  # type: ignore[no-any-return]


def test_the_json_validates_against_the_schema(env: Env) -> None:  # noqa: F811
    score_all(env)
    data = jreport(env)
    validate(data, report.schema(), report.schema())
    assert data["scores"]["overall"]["scored"] is True
    assert {f["rule"] for f in data["findings"]} >= {"E-01", "R-04"}
    assert data["view"] == "key" and data["score_version"] == 1


def test_the_schema_rejects_a_wrong_report(env: Env) -> None:  # noqa: F811
    score_all(env)
    data = jreport(env)
    data["extra"] = 1
    with pytest.raises(AssertionError):
        validate(data, report.schema(), report.schema())


def test_no_id_no_key_name_no_canary_in_either_format(env: Env) -> None:  # noqa: F811
    score_all(env)
    put_toolset(env.db, A, {seed.tool_hash(i): f"a_tool_{i}" for i in range(4)}, tools=4)
    sess = sid(env, A)
    ref = sess[2:]
    for fmt in ("json", "html"):
        text = _a(env, f"/sessions/{sess}/report?format={fmt}").text
        for needle in (ref, sess, req("reqa"), "reqa-fake", A, "tok1", "my-token", "CANARY",
                       seed.tool_hash(3), "192.0.2", KEY_A):  # fmt: skip
            assert needle not in text, (fmt, needle)
        assert "a_tool_" not in text


def test_tool_names_only_when_asked(env: Env) -> None:  # noqa: F811
    score_all(env)
    put_toolset(env.db, A, {seed.tool_hash(i): f"a_tool_{i}" for i in range(4)}, tools=4)
    plain = _a(env, f"/sessions/{sid(env, A)}/report?format=json").text
    named = _a(env, f"/sessions/{sid(env, A)}/report?format=json&tool_names=1").text
    assert "a_tool_" not in plain and "tool #" in plain
    assert "a_tool_3" in named
    data = json.loads(named)
    assert any(
        f.get("tool_name") == "a_tool_3" and f["tool"].startswith("tool #")
        for f in data["findings"]
    )


def test_html_is_self_contained(env: Env) -> None:  # noqa: F811
    score_all(env)
    r = _a(env, f"/sessions/{sid(env, A)}/report?format=html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "attachment" in r.headers["content-disposition"]
    body = r.text
    assert "<script" not in body.lower() and "http" not in body.lower()
    assert "<link" not in body.lower() and "src=" not in body.lower() and "@import" not in body
    assert "<style>" in body and "Prompt Lens session report" in body
    assert "E-01" in body and "system prompt line 2" in body


def test_the_admin_view_says_so_on_its_first_line_only_when_chosen(env: Env) -> None:  # noqa: F811
    score_all(env)
    mine = _a(env, f"/sessions/{sid(env, A)}/report?format=html").text
    theirs = _admin(env, f"/sessions/{sid(env, A)}/report?format=html").text
    assert "admin view" not in mine
    assert re.search(r"<body><p class=mut>admin view</p>", theirs)
    assert json.loads(_admin(env, f"/sessions/{sid(env, A)}/report").text)["view"] == "admin"
    assert "key_id" not in theirs and A not in theirs


def test_two_sessions_share_no_id(env: Env) -> None:  # noqa: F811
    score_all(env)
    a = _a(env, f"/sessions/{sid(env, A)}/report").text
    b = env.client.get(
        f"{ME}/sessions/{sid(env, B)}/report", headers=bearer("vw_secondtoken1234567890abcdef1234")
    ).text
    ids = lambda t: set(re.findall(r"[0-9a-f]{16,}|reqa-\S+|reqb-\S+", t))  # noqa: E731
    assert not ids(a) and not ids(b)


def test_a_key_cannot_download_a_foreign_session(env: Env) -> None:  # noqa: F811
    score_all(env)
    r = _a(env, f"/sessions/{sid(env, B)}/report")
    assert r.status_code == 404
    assert (
        env.client.get(f"{ADM}/sessions/{sid(env, B)}/report", headers=env.admin).status_code == 200
    )


def test_the_review_is_rubric_scores_only() -> None:
    card = {
        "session": {"harness": "h", "model": "m", "route": "local", "turns": 1, "duration_s": 1.0},
        "score_version": 1, "weights": {}, "scores": {}, "top": [], "findings": [], "turns": [],
        "cost": {"prompt_tokens": 1, "reused_tokens": 0, "recomputed_tokens": 1, "fixable_lost_tokens": 0, "card": None},
        "review": {"score": 3.4, "scale": 5, "rubric": {"clarity": 4}, "text": "SECRET QUOTE", "quotes": ["x"]},
    }  # fmt: skip
    out = report.report_data(card)
    assert out["review"] == {"score": 3.4, "scale": 5, "rubric": {"clarity": 4}}
    assert "SECRET" not in json.dumps(out)


def test_the_page_routes_serve_the_same_report(env: Env) -> None:  # noqa: F811
    score_all(env)
    page = env.a.get(f"/api/me/prompt-lens/sessions/{sid(env, A)}/report?format=json")
    assert page.status_code == 200 and page.text == _a(env, f"/sessions/{sid(env, A)}/report").text
    assert env.a.get(f"/api/me/prompt-lens/sessions/{sid(env, B)}/report").status_code == 404
    assert env.as_admin(f"/api/me/prompt-lens/sessions/{sid(env, A)}/report").status_code == 403
    adm = env.as_admin(f"/api/admin/prompt-lens/sessions/{sid(env, B)}/report?format=html")
    assert adm.status_code == 200 and "admin view" in adm.text
    assert env.a.get(f"/api/admin/prompt-lens/sessions/{sid(env, A)}/report").status_code == 401
