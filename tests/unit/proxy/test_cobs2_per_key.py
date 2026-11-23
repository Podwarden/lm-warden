"""Lineage, reply anchors, heads and the prefix memory are per API key.

Client session ids are not secret and not per key (an OpenAI ``user``, a
harness session header), so two keys sending the same one must never share
lineage nodes, tips, reply anchors, heads or prefix estimates: that would
tell one key whether another sent the same prefix (key_scope)."""

import sqlite3

from app.cobs2.lineage import key_scope
from app.db.repos.tokens import hash_token
from tests.unit.proxy.test_cache_estimate_forward import (
    _body,
    _live_rows,
    _spy_history,
    _spy_registry,
)
from tests.unit.proxy.test_cobs2_forward import _ids_of, _turn
from tests.unit.proxy.test_dp_routing_forward import AUTH, _post, _ready

OTHER = "vw_othertoken0987654321fedcba98765"
AUTH2 = {"Authorization": f"Bearer {OTHER}"}
SESSION = {"X-Claude-Code-Session-Id": "0b1c2d3e-1111-4222-8333-444455556666"}


def _second_key(tmp_data_dir):
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok2", "other", OTHER[:8], hash_token(OTHER), "inference"),
        )
        db.commit()


def test_key_scope_separates_keys_and_salts():
    assert key_scope(b"", "a") != key_scope(b"", "b")
    assert key_scope(b"x", "a") != key_scope(b"y", "a")
    assert key_scope(b"x", "a") == key_scope(b"x", "a")


def test_two_keys_with_one_session_id_never_link(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _second_key(tmp_data_dir)
    recs = _spy_history(client)
    a = [{"role": "user", "content": "a"}]
    ra = [*a, {"role": "assistant", "content": "ra"}]
    b = [*ra, {"role": "user", "content": "b"}]
    c = [*ra, {"role": "user", "content": "c"}]
    t1 = _turn(client, recs, a, reply="ra", headers={**SESSION, **AUTH})
    t2 = _turn(client, recs, b, headers={**SESSION, **AUTH})
    # Key 1's own lineage links.
    assert (t2["cobs2_edge"], t2["cobs2_parent_node"]) == ("continue", t1["cobs2_node"])
    # Key 2, same client session id, rewrites key 1's turn: with a shared lineage
    # this would be a fork of key 1's conversation; per key it is a new body.
    o = _turn(client, recs, c, headers={**SESSION, **AUTH2})
    assert o["token_id"] == "tok2"
    assert (o["cobs2_edge"], o["cobs2_parent_node"], o.get("cobs2_parent_req")) == (
        "new",
        None,
        None,
    )
    assert not o["cobs2_loose_beyond_exact"]
    # Key 2 continuing key 1's exact conversation is new too.
    o2 = _turn(
        client,
        recs,
        [*b, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "d"}],
        headers={**SESSION, **AUTH2},
    )
    assert o2["cobs2_parent_node"] != t2["cobs2_node"]
    # (cobs2_conv is a content id: equal content gives an equal id under any
    # key and is no link; only the lineage edge and parent are.)


def test_compaction_of_another_key_is_not_compacted_from(tmp_data_dir, client):
    _ready(client, tmp_data_dir, dp=1)
    _second_key(tmp_data_dir)
    recs = _spy_history(client)
    sys = {"role": "system", "content": "be brief"}
    msgs = [sys, {"role": "user", "content": "Write a summary of the conversation so far."}]
    _turn(client, recs, msgs, reply="summary text", headers={**SESSION, **AUTH})
    collapsed = [sys, {"role": "user", "content": "summary text, carry on"}]
    o = _turn(client, recs, collapsed, headers={**SESSION, **AUTH2})
    assert o["cobs2_edge"] != "compacted_from" and o["cobs2_parent_node"] is None
    # The same collapsed body from key 1 is its compaction.
    own = _turn(client, recs, collapsed, headers={**SESSION, **AUTH})
    assert own["cobs2_edge"] == "compacted_from"


def test_heads_are_per_key(tmp_data_dir, client):
    """The subagent rule asks "is this head new?": another key's system prompt
    and tools must not answer it."""
    _ready(client, tmp_data_dir, dp=1)
    _second_key(tmp_data_dir)
    recs = _spy_history(client)
    msgs = [{"role": "system", "content": "you are a helper"}, {"role": "user", "content": "x"}]
    _turn(client, recs, msgs, headers={**SESSION, **AUTH})
    child = {
        "X-Claude-Code-Session-Id": "9f8e7d6c-1111-4222-8333-444455556666",
        "x-parent-session-id": SESSION["X-Claude-Code-Session-Id"],
    }
    o = _turn(client, recs, msgs, headers={**child, **AUTH2})
    # Key 2 never saw this head: a subagent edge, with no parent in key 1's session.
    assert (o["cobs2_edge"], o["cobs2_parent_node"]) == ("subagent", None)
    heads = {k[1] for k in client.app.state.cobs2_lineage._heads}
    assert len(heads) == 2
    assert _ids_of(client, msgs).head in {k[2] for k in client.app.state.cobs2_lineage._heads}


def test_prefix_memory_is_per_key(tmp_data_dir, client):
    _ready(client, tmp_data_dir)
    _second_key(tmp_data_dir)
    _spy_registry(client)
    _post(client, _body(5))
    _post(client, _body(5), headers=AUTH2)
    _post(client, _body(5))
    first, other, again = _live_rows(client)
    assert first.cache_est_tokens is None
    # Same ``user`` (the client session id) and prompt from another key: no estimate.
    assert other.token_id == "tok2" and other.cache_est_tokens is None
    assert again.cache_est_tokens == first.prompt_tokens > 0
