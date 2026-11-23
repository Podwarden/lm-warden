"""What a key holder's Prompt Lens stream may carry (G7, spec 2026-11-22 section 4).

The stream is already narrowed to the caller's own key (``event_matches`` with
the key from the sign-in, a filtered hub subscription and the key's own replay
window). ``for_key_holder`` then removes what an own event still says about
anyone else:

* ``client_ip`` and ``token_label`` (isolation rule 2; the label is their own
  key's name, so it is redundant);
* every any-key or fleet cache figure: ``reusable_any`` (kvsig), the raw engine
  ``cached_tokens`` and the ``cache`` observation (request_end), and the
  cache-observation names that live inside it (``reusable_tokens``,
  ``reusable_tokens_fleet``, ``cache_outcome``, ``divergence_cause``) and an
  ``other_key_label`` wherever they appear;
* the kvsig ``other`` part state (ruling G-R11): a key holder's parts are
  ``parts_key``, built as if no other key had cached anything
  (``prompt_lens_kvsig.build_kvsig_pair``), so no state, boundary or size says
  where another key's span ended. Without ``parts_key`` (no part was
  ``other``) the parts pass as they are. Should an ``other`` part still
  arrive, it reads ``new`` and merges with its ``new`` node neighbours, as a
  backstop;
* the hub's global ``seq``, which counts every key's events, so the gap between
  two own events would tell how busy the other keys are. The key stream numbers
  its own frames instead (app/proxy/prompt_lens_stream.py).

Every other field passes. The result is a copy: the input is the ring's own
dict, shared with every other subscriber, and is never changed.
"""

from __future__ import annotations

from typing import Any

#: Dropped from every event a key holder receives.
_DROP = frozenset(
    {
        "client_ip",
        "token_label",
        "reusable_any",
        "cached_tokens",
        "cache",
        "reusable_tokens",
        "reusable_tokens_fleet",
        "cache_outcome",
        "divergence_cause",
        "other_key_label",
        "seq",
        "parts_key",
    }
)


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    roles = dict(a.get("roles") or {})
    for k, v in (b.get("roles") or {}).items():
        roles[k] = roles.get(k, 0) + v
    # A run's ids are its last node's (a node id fixes its whole prefix).
    return {
        **b,
        "from": a.get("from"),
        "chars": (a.get("chars") or 0) + (b.get("chars") or 0),
        "roles": roles,
    }


def _backstop(parts: Any) -> Any:
    if not isinstance(parts, list):
        return parts
    out: list[Any] = []
    for p in parts:
        if isinstance(p, dict) and p.get("state") == "other":
            p = {**p, "state": "new"}
        prev = out[-1] if out else None
        if (
            isinstance(p, dict)
            and isinstance(prev, dict)
            and p.get("p") == prev.get("p") == "nodes"
            and p.get("state") == prev.get("state") == "new"
        ):
            out[-1] = _merge(prev, p)
        else:
            out.append(p)
    return out


def admin_view(event: dict[str, Any]) -> dict[str, Any]:
    """An event as every admin route sends it (the stream and the kept
    capture): as published, without kvsig's ``parts_key`` (the key holder's
    parts, prompt_lens_kvsig.build_kvsig_pair), which never leaves the server."""
    if "parts_key" not in event:
        return event
    return {k: v for k, v in event.items() if k != "parts_key"}


def for_key_holder(event: dict[str, Any]) -> dict[str, Any]:
    """``event`` as its key's holder may see it (module docstring). A copy."""
    out = {k: v for k, v in event.items() if k not in _DROP}
    if event.get("type") == "kvsig":
        key_parts = event.get("parts_key")
        parts = key_parts if isinstance(key_parts, list) else out.get("parts")
        if isinstance(parts, list) and any(
            isinstance(p, dict) and p.get("state") == "other" for p in parts
        ):
            parts = _backstop(parts)
        if parts is not None:
            out["parts"] = parts
    return out
