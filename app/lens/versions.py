"""Prompt versions (analysis Task 3.1, spec 13.2).

A prompt version is detected per key and model from the system prompt's line
digests with the lines of volatile span classes (a date, a path, an id) left
out, plus the toolset digest: a change in either starts a new version, a
volatile-only change never does (otherwise every worktree would be a version).
A client can name versions with ``X-Prompt-Version`` (at most 64 printable
characters, cleaned like ``x-batch-id``), which then wins.

The stored value is ``"h:" + the cleaned header`` for a named version and the
8 hex of ``version_id`` for a detected one, so a client's name can never be
mistaken for a detected id. Nothing here holds prompt text (A-10): the id is a
keyed hash of keyed line digests.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Any

from app.cobs2.keys import NetworkKey
from app.lens import probes
from app.proxy.forest_fields import clean_batch_id

#: A line longer than this is not classified (the regexes would scan up to a
#: megabyte): it counts as stable, so an edit to it starts a version.
MAX_CLASSIFIED_CHARS = 2_048
HEADER_PREFIX = "h:"
#: Rotation-chain hops followed at most (a hand-edited cycle ends).
MAX_LINEAGE = 64


def volatile_lines(text: str) -> frozenset[int]:
    """0-based indexes (into ``probes.sys_line_digests``) of the system prompt
    lines whose span class is volatile. The chunks past the line cap are
    stable."""
    lines, _rest = probes.sys_pieces(text)
    return frozenset(
        i
        for i, ln in enumerate(lines)
        if len(ln) <= MAX_CLASSIFIED_CHARS and probes.span_class(ln) in probes.VOLATILE_CLASSES
    )


def version_id(
    sys_digests: Sequence[bytes],
    volatile: frozenset[int],
    toolset_digest: str | None,
    key: NetworkKey,
    salt: bytes,
) -> str:
    """8 hex of the stable system prompt lines and the toolset (``None``: no
    tools)."""
    stable = b"".join(d for i, d in enumerate(sys_digests) if i not in volatile)
    payload = stable + b"\x00" + (toolset_digest or "").encode("ascii", "replace")
    return probes._d8(key, b"lens-version", salt, payload).hex()[:8]


def client_version(raw: str | None) -> str | None:
    """The stored value of an ``X-Prompt-Version`` header, or None when absent,
    empty, over 64 characters or holding a control character."""
    v = clean_batch_id(raw)
    return HEADER_PREFIX + v if v else None


def lineage_ids(db: sqlite3.Connection, token_id: str) -> list[str]:
    """The ids of ``token_id``'s whole rotation chain, oldest first (the same
    walk as ``TokenRepo.lineage``, on a worker thread's connection). A key that
    was never rotated is its own chain; an unknown id answers just itself."""
    seen = [token_id]
    cur = token_id
    for _ in range(MAX_LINEAGE):
        r = db.execute("SELECT rotated_from FROM api_tokens WHERE id = ?", (cur,)).fetchone()
        prev = r[0] if r is not None else None
        if not prev or prev in seen:
            break
        seen.insert(0, prev)
        cur = prev
    cur = token_id
    for _ in range(MAX_LINEAGE):
        r = db.execute(
            "SELECT id FROM api_tokens WHERE rotated_from = ? ORDER BY created_at ASC, id ASC"
            " LIMIT 1",
            (cur,),
        ).fetchone()
        if r is None or r[0] in seen:
            break
        seen.append(r[0])
        cur = r[0]
    return seen


def version_label(
    db: sqlite3.Connection,
    key_ids: Sequence[str],
    model: str,
    version: str,
    *,
    first_seen: float | None = None,
) -> dict[str, Any]:
    """``{"label": "v4", "since": <epoch s>}``: the ordinal of ``version`` among
    the versions the chain's scored sessions on ``model`` show, by first seen (a
    version the client named is labelled with its own name and not counted).
    A version no scored session carries yet (the session is still active) is
    the next ordinal, since ``first_seen``."""
    marks = ",".join("?" * len(key_ids))
    got = db.execute(
        f"SELECT version, MIN(started) AS s FROM lens_session_scores WHERE key_id IN ({marks})"
        " AND model = ? AND version IS NOT NULL GROUP BY version ORDER BY s ASC, version ASC",
        (*key_ids, model),
    ).fetchall()
    since = {str(v): float(s) for v, s in got}
    if version.startswith(HEADER_PREFIX):  # the client's own name is its label
        return {"label": version[len(HEADER_PREFIX) :], "since": since.get(version, first_seen)}
    order = [v for v in since if not v.startswith(HEADER_PREFIX)]  # detected versions only
    if version in since:
        return {"label": f"v{order.index(version) + 1}", "since": since[version]}
    return {"label": f"v{len(order) + 1}", "since": first_seen}
