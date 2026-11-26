"""Text-free lint of the offered tool schemas (analysis spec 8.2, malformed call).

One int of flags per tool, checked once per distinct toolset (the hot path
memoises it by the toolset's id). Reads names, types and lengths only and
returns bits: no schema text leaves here (A-3, A-10).
"""

from __future__ import annotations

from typing import Any

#: A parameter (at any depth) with no ``type`` (nor a union, enum, const or ref).
LINT_NO_TYPE = 1
#: The tool, or one of its parameters, has no description.
LINT_NO_DESC = 2
#: A nested object with properties but no ``required`` list.
LINT_NESTED_NO_REQUIRED = 4
#: A description under ``SHORT_DESC_CHARS`` for a tool with ``SHORT_DESC_PARAMS`` or more parameters.
LINT_SHORT_DESC = 8

SHORT_DESC_CHARS = 80
SHORT_DESC_PARAMS = 3
#: Schemas nested deeper than this are not walked further (bounded work).
MAX_DEPTH = 16
#: At most this many parameters are walked per tool.
MAX_PARAMS = 1_024

_TYPED = ("type", "anyOf", "oneOf", "allOf", "enum", "const", "$ref")


def _fn(tool: Any) -> tuple[Any, Any]:
    """``(description, parameters)`` of an OpenAI or Anthropic tool definition."""
    if not isinstance(tool, dict):
        return None, None
    f = tool.get("function")
    if isinstance(f, dict):
        return f.get("description"), f.get("parameters")
    if "input_schema" in tool or "name" in tool:
        return tool.get("description"), tool.get("input_schema", tool.get("parameters"))
    return None, None


def _walk(schema: Any, depth: int, budget: list[int]) -> int:
    """Flags of the properties under the object schema ``schema``."""
    if depth > MAX_DEPTH or not isinstance(schema, dict):
        return 0
    flags = 0
    props = schema.get("properties")
    if isinstance(props, dict):
        for p in props.values():
            budget[0] -= 1
            if budget[0] < 0:
                return flags
            if not isinstance(p, dict):
                flags |= LINT_NO_TYPE | LINT_NO_DESC
                continue
            if not any(k in p for k in _TYPED):
                flags |= LINT_NO_TYPE
            d = p.get("description")
            if not (isinstance(d, str) and d.strip()):
                flags |= LINT_NO_DESC
            flags |= _nested(p, depth + 1, budget)
    return flags


def _nested(p: dict[str, Any], depth: int, budget: list[int]) -> int:
    """Flags inside one parameter: a nested object, or an array's items."""
    flags = 0
    if isinstance(p.get("properties"), dict) and p["properties"]:
        if not isinstance(p.get("required"), list):
            flags |= LINT_NESTED_NO_REQUIRED
        flags |= _walk(p, depth, budget)
    items = p.get("items")
    if isinstance(items, dict):
        flags |= _nested(items, depth + 1, budget)
    return flags


def lint_tool(tool: dict[str, Any]) -> int:
    """The lint flags of one tool definition (0: nothing found or unreadable)."""
    desc, params = _fn(tool)
    if desc is None and params is None and not isinstance(tool, dict):
        return 0
    flags = 0
    if not (isinstance(desc, str) and desc.strip()):
        flags |= LINT_NO_DESC
    props = params.get("properties") if isinstance(params, dict) else None
    n = len(props) if isinstance(props, dict) else 0
    if (
        n >= SHORT_DESC_PARAMS
        and len(desc.strip() if isinstance(desc, str) else "") < SHORT_DESC_CHARS
    ):
        flags |= LINT_SHORT_DESC
    flags |= _walk(params, 0, [MAX_PARAMS])
    return flags


#: At most this many tools are linted per toolset (OpenAI's limit is 128).
MAX_TOOLS = 512


def lint_tools(tools: Any) -> list[int]:
    """One int per offered tool, in request order (an entry that is not a
    tool definition reads 0). Raises only on a bug; the caller fails open."""
    if not isinstance(tools, list):
        return []
    return [lint_tool(t) if isinstance(t, dict) else 0 for t in tools[:MAX_TOOLS]]
