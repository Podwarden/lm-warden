"""Tool schema lint (analysis spec 8.2, malformed-call causes; plan Task 1.4)."""

import json

from app.lens.schema_lint import (
    LINT_NESTED_NO_REQUIRED,
    LINT_NO_DESC,
    LINT_NO_TYPE,
    LINT_SHORT_DESC,
    lint_tool,
    lint_tools,
)

LONG = "Search the documentation index for pages that match a query and return the best ones."


def fn(description, parameters):
    f = {"name": "search_docs", "parameters": parameters}
    if description is not None:
        f["description"] = description
    return {"type": "function", "function": f}


def prop(t="string", d="a thing"):
    p = {}
    if t is not None:
        p["type"] = t
    if d is not None:
        p["description"] = d
    return p


def test_flag_values():
    assert (LINT_NO_TYPE, LINT_NO_DESC, LINT_NESTED_NO_REQUIRED, LINT_SHORT_DESC) == (1, 2, 4, 8)


def test_clean_tool_is_zero():
    t = fn(LONG, {"type": "object", "properties": {"q": prop(), "n": prop("integer")}})
    assert lint_tool(t) == 0


def test_lint_flags_missing_type_and_short_description():
    t = fn(
        "Search docs.",
        {"type": "object", "properties": {"q": prop(None), "n": prop("integer"), "m": prop()}},
    )
    assert lint_tool(t) == LINT_NO_TYPE | LINT_SHORT_DESC


def test_short_description_needs_three_parameters():
    t = fn("Search docs.", {"type": "object", "properties": {"q": prop(), "n": prop()}})
    assert lint_tool(t) == 0


def test_missing_descriptions():
    assert lint_tool(fn(None, {"type": "object", "properties": {}})) == LINT_NO_DESC
    t = fn(LONG, {"type": "object", "properties": {"q": prop(d=None)}})
    assert lint_tool(t) == LINT_NO_DESC
    # No description and three parameters: also short.
    t = fn(None, {"type": "object", "properties": {"a": prop(), "b": prop(), "c": prop()}})
    assert lint_tool(t) == LINT_NO_DESC | LINT_SHORT_DESC


def test_a_union_or_enum_counts_as_typed():
    props = {
        "a": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "x"},
        "b": {"enum": ["x", "y"], "description": "x"},
        "c": {"$ref": "#/defs/c", "description": "x"},
    }
    assert lint_tool(fn(LONG, {"type": "object", "properties": props})) == 0


def test_nested_object_without_required():
    nested = {
        "type": "object",
        "description": "the filter",
        "properties": {"from": prop(), "to": prop()},
    }
    t = fn(LONG, {"type": "object", "properties": {"f": nested}})
    assert lint_tool(t) == LINT_NESTED_NO_REQUIRED
    nested["required"] = ["from"]
    assert lint_tool(t) == 0


def test_nested_parameters_are_checked_inside_arrays():
    items = {"type": "object", "properties": {"x": prop(None)}, "required": ["x"]}
    t = fn(
        LONG,
        {
            "type": "object",
            "properties": {"rows": {"type": "array", "description": "r", "items": items}},
        },
    )
    assert lint_tool(t) == LINT_NO_TYPE


def test_anthropic_shape():
    t = {
        "name": "x",
        "description": LONG,
        "input_schema": {"type": "object", "properties": {"q": prop(None)}},
    }
    assert lint_tool(t) == LINT_NO_TYPE


def test_garbage_is_zero_and_never_raises():
    for t in (
        None,
        5,
        "x",
        {},
        {"function": 3},
        {"type": "function", "function": {"parameters": "x"}},
    ):
        assert isinstance(lint_tool(t), int)  # type: ignore[arg-type]
    deep: dict = {"type": "object", "properties": {}}
    cur = deep
    for _ in range(500):
        nxt: dict = {"type": "object", "description": "d", "properties": {}, "required": []}
        cur["properties"]["n"] = nxt
        cur = nxt
    assert isinstance(lint_tool(fn(LONG, deep)), int)


def test_lint_tools_one_int_per_tool_and_text_free():
    canary = "CANARY-1d4b7e-schema"
    tools = [
        fn(canary, {"type": "object", "properties": {canary: prop(None, canary)}}),
        fn(LONG, {"type": "object", "properties": {}}),
        "junk",
    ]
    got = lint_tools(tools)
    assert got == [LINT_NO_TYPE, 0, 0]
    assert canary not in json.dumps(got)
    assert lint_tools(None) == [] and lint_tools("x") == []
