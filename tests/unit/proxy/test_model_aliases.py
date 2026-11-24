"""Model aliases in the proxy (D4a-D4e).

An enabled alias wins over a model with the same served name; a request is
routed to the first LOADED target in the alias's ordered list; /v1/models
lists each alias with a loaded target under its name and hides a model whose
served name is shadowed by an alias.
"""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

from app.db.repos.tokens import hash_token

_MODEL_SQL = (
    "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
    "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
    "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
    "VALUES (?,?,'r','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,NULL)"
)


def _seed(db_path, models, tokens=None):
    """models: [(id, served_model_name, status), ...]. tokens: list of
    (token_id, allowed_models). Returns {token_id: plaintext}."""
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        for mid, served, status in models:
            db.execute(_MODEL_SQL, (mid, served, json.dumps([0]), status))
        plaintexts = {}
        for token_id, allowed_models in tokens or [("tok1", None)]:
            plaintext = f"vw_alias_test_{token_id}_1234567890abcdef"
            db.execute(
                "INSERT INTO api_tokens(id, name, prefix, hash, scope, allowed_models, priority) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    token_id,
                    token_id,
                    plaintext[:8],
                    hash_token(plaintext),
                    "inference",
                    allowed_models,
                    0,
                ),
            )
            plaintexts[token_id] = plaintext
        db.commit()
    return plaintexts


def _add_alias(db_path, name, targets, enabled=1):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO model_aliases(id, name, targets, enabled) VALUES (?, ?, ?, ?)",
            (f"alias-{name}", name, json.dumps(targets), enabled),
        )
        db.commit()


def _fake_resp():
    fake = MagicMock()
    fake.status_code = 200
    fake.headers = {"content-type": "application/json"}
    fake.aread = AsyncMock(
        return_value=b'{"id":"x","model":"x","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}'
    )
    fake.aclose = AsyncMock()
    return fake


def _forward(client, token, payload):
    fake = _fake_resp()
    with (
        patch("httpx.AsyncClient.send", new=AsyncMock(return_value=fake)) as send,
        patch("app.proxy.routes._record_counters", new=AsyncMock()),
    ):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
    return r, send


def test_alias_wins_over_same_named_model(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "loaded"), ("m2", "model2", "loaded")])
    _add_alias(db_path, "model1", ["m2"])
    client.app.state.supervisor._ports["m1"] = 19001
    client.app.state.supervisor._ports["m2"] = 19002

    r, send = _forward(
        client,
        plaintexts["tok1"],
        {"model": "model1", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    sent_url = str(send.call_args.args[0].url)
    assert "127.0.0.1:19002" in sent_url
    forwarded = json.loads(send.call_args.args[0].content)
    assert forwarded["model"] == "model2"


def test_fallback_to_next_loaded_target(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "pulled"), ("m2", "model2", "loaded")])
    _add_alias(db_path, "alias-name", ["m1", "m2"])
    client.app.state.supervisor._ports["m2"] = 19002

    r, send = _forward(
        client,
        plaintexts["tok1"],
        {"model": "alias-name", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    assert "127.0.0.1:19002" in str(send.call_args.args[0].url)
    assert json.loads(send.call_args.args[0].content)["model"] == "model2"


def test_all_targets_unloaded_is_404(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "pulled"), ("m2", "model2", "pulled")])
    _add_alias(db_path, "dead", ["m1", "m2"])

    r, send = _forward(
        client,
        plaintexts["tok1"],
        {"model": "dead", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 404
    assert r.json()["detail"] == "no model behind alias 'dead' is loaded"
    assert send.call_count == 0


def test_deleted_target_model_is_404(tmp_data_dir, client):
    """A target whose row is gone (deleted) must 404, never 500."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "pulled")])
    _add_alias(db_path, "dead", ["ghost-id", "m1"])

    r, send = _forward(
        client,
        plaintexts["tok1"],
        {"model": "dead", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 404
    assert r.json()["detail"] == "no model behind alias 'dead' is loaded"
    assert send.call_count == 0


def test_disabled_alias_is_ignored(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "loaded"), ("m2", "model2", "loaded")])
    _add_alias(db_path, "model1", ["m2"], enabled=0)
    client.app.state.supervisor._ports["m1"] = 19001
    client.app.state.supervisor._ports["m2"] = 19002

    r, send = _forward(
        client,
        plaintexts["tok1"],
        {"model": "model1", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert r.status_code == 200
    assert "127.0.0.1:19001" in str(send.call_args.args[0].url)
    assert json.loads(send.call_args.args[0].content)["model"] == "model1"


def test_allow_list_alias_rules(tmp_data_dir, client):
    """D4d: the key may see the request via the alias name OR via the serving
    model's served name."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(
        db_path,
        [("m1", "model1", "loaded"), ("m2", "model2", "loaded")],
        tokens=[("t-alias", "model1"), ("t-serving", "model2"), ("t-other", "other")],
    )
    _add_alias(db_path, "model1", ["m2"])
    client.app.state.supervisor._ports["m1"] = 19001
    client.app.state.supervisor._ports["m2"] = 19002
    payload = {"model": "model1", "messages": [{"role": "user", "content": "hi"}]}

    r, send = _forward(client, plaintexts["t-alias"], payload)
    assert r.status_code == 200
    assert json.loads(send.call_args.args[0].content)["model"] == "model2"

    r, send = _forward(client, plaintexts["t-serving"], payload)
    assert r.status_code == 200
    assert json.loads(send.call_args.args[0].content)["model"] == "model2"

    r, send = _forward(client, plaintexts["t-other"], payload)
    assert r.status_code == 403
    assert r.json()["detail"] == "token not allowed for model 'model1'"
    assert send.call_count == 0


def test_v1_models_lists_alias_and_hides_shadowed_model(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(
        db_path,
        [
            ("m1", "model1", "loaded"),
            ("m2", "model2", "loaded"),
            ("m3", "model3", "pulled"),
        ],
    )
    _add_alias(db_path, "model1", ["m2"])
    _add_alias(db_path, "dead", ["m3"])

    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintexts['tok1']}"})

    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    # model1 is hidden (shadowed by the alias); the alias is listed instead.
    assert ids.count("model1") == 1
    assert ids.count("model2") == 1
    alias_entry = next(m for m in r.json()["data"] if m["id"] == "model1")
    assert alias_entry["object"] == "model"
    assert alias_entry["owned_by"] == "vllm-warden"
    # The alias entry carries the serving model's fields (clients size their
    # context from max_model_len), with the alias name as id.
    model_entry = next(m for m in r.json()["data"] if m["id"] == "model2")
    assert model_entry["max_model_len"] == 4096
    assert alias_entry == {**model_entry, "id": "model1"}
    # An alias with no loaded target is not listed.
    assert "dead" not in ids
    assert "model3" not in ids


def test_no_alias_unchanged(tmp_data_dir, client):
    """A plain model request (no alias) forwards byte-identical bytes."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "loaded")])
    client.app.state.supervisor._ports["m1"] = 19001

    payload = {"model": "model1", "messages": [{"role": "user", "content": "hi"}]}
    r, send = _forward(client, plaintexts["tok1"], payload)

    assert r.status_code == 200
    # The client's wire bytes (httpx serializes json= with separators=(",", ":"))
    # must reach the engine untouched.
    assert send.call_args.args[0].content == json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


# --- alias settings applied through the forward path (Task 6) --------------------


def _add_full_alias(
    db_path,
    name,
    targets,
    *,
    guard="inherit",
    thinking_policy="inherit",
    params=None,
    enabled=1,
):
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO model_aliases(id, name, targets, guard, thinking_policy, params, enabled) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                f"alias-{name}",
                name,
                json.dumps(targets),
                guard,
                thinking_policy,
                json.dumps(params or {}),
                enabled,
            ),
        )
        db.commit()


def test_alias_params_and_thinking_applied_in_forward_path(tmp_data_dir, client):
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "loaded")])
    _add_full_alias(
        db_path,
        "alias",
        ["m1"],
        thinking_policy="off",
        params={
            "temperature": {"value": 0.1, "mode": "force"},
            "max_tokens": {"value": 128, "mode": "cap"},
        },
    )
    client.app.state.supervisor._ports["m1"] = 19001

    r, send = _forward(
        client,
        plaintexts["tok1"],
        {
            "model": "alias",
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert r.status_code == 200
    forwarded = json.loads(send.call_args.args[0].content)
    assert forwarded["model"] == "model1"
    assert forwarded["chat_template_kwargs"]["enable_thinking"] is False
    assert forwarded["temperature"] == 0.1
    assert forwarded["max_tokens"] == 128


def test_alias_guard_sanitizes_when_key_has_none(tmp_data_dir, client):
    from tests.unit.guard.corpus import github_token

    value = github_token()
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    plaintexts = _seed(db_path, [("m1", "model1", "loaded")])
    _add_full_alias(db_path, "alias", ["m1"], guard="sanitize")
    client.app.state.supervisor._ports["m1"] = 19001

    r, send = _forward(
        client,
        plaintexts["tok1"],
        {
            "model": "alias",
            "messages": [{"role": "user", "content": f"deploy with {value}"}],
        },
    )

    assert r.status_code == 200
    forwarded = send.call_args.args[0].content.decode()
    assert value not in forwarded
