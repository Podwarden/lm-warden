import sqlite3

from tests.conftest import csrf_header, jwt_login, seed_admin_user


def _auth(tmp_data_dir, client):
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    return {**jwt_login(client), **csrf_header(client)}


def _insert_loaded_model(db_path, served="qwen", window=8192):
    c = sqlite3.connect(db_path)
    c.execute(
        "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices, tensor_parallel_size, "
        "status, max_model_len, supports_tools, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
        (served + "-id", served, "org/" + served, "[0]", 1, "loaded", window, 1),
    )
    c.commit()


def test_chat_crud_roundtrip(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    r = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"})
    assert r.status_code == 201, r.text
    chat = r.json()
    assert chat["title"] == "New chat" and chat["settings"]["temperature"] == 0.7
    # the reassessment feature's undo affordance: always present, null until a
    # generated title has actually replaced an earlier one.
    assert chat["title_prev"] is None
    listed = client.get("/api/chat2/chats", headers=h).json()["chats"][0]
    assert listed["id"] == chat["id"] and listed["title_prev"] is None
    g = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()
    assert g["messages"] == [] and g["context"] == {
        "type": "context",
        "promptTokens": 0,
        "window": None,
        "full": False,
    }
    p = client.patch(
        f"/api/chat2/chats/{chat['id']}",
        headers=h,
        json={"title": "Mine", "settings": {"temperature": 0.1}},
    )
    assert p.json()["title"] == "Mine" and p.json()["title_source"] == "user"
    assert p.json()["settings"]["temperature"] == 0.1 and p.json()["settings"]["max_tokens"] == 1024
    assert client.delete(f"/api/chat2/chats/{chat['id']}", headers=h).status_code == 204
    gone = client.get(f"/api/chat2/chats/{chat['id']}", headers=h)
    # every chat2 non-2xx uses the shared {code, message} envelope (spec §4.5)
    assert gone.status_code == 404 and gone.json()["detail"]["code"] == "not_found"
    for call in (
        lambda: client.patch(f"/api/chat2/chats/{chat['id']}", headers=h, json={}),
        lambda: client.delete(f"/api/chat2/chats/{chat['id']}", headers=h),
        lambda: client.post(f"/api/chat2/chats/{chat['id']}/fork", headers=h, json={"at_seq": 1}),
    ):
        r = call()
        assert r.status_code == 404 and r.json()["detail"]["code"] == "not_found", r.text


def test_defaults_seed_new_chats_and_models_budget_whoami(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    _insert_loaded_model(tmp_data_dir / "vllm-warden.db")
    put = client.put(
        "/api/chat2/defaults", headers=h, json={"model": "qwen", "settings": {"temperature": 0.3}}
    )
    assert put.status_code == 200 and put.json()["settings"]["temperature"] == 0.3
    chat = client.post("/api/chat2/chats", headers=h, json={}).json()
    assert chat["model"] == "qwen" and chat["settings"]["temperature"] == 0.3
    models = client.get("/api/chat2/models", headers=h).json()["models"]
    assert models[0]["id"] == "qwen" and models[0]["context_window"] == 8192
    assert client.get("/api/chat2/budget", headers=h).json() == {"budget": None}
    assert isinstance(client.get("/api/chat2/_whoami", headers=h).json()["user_id"], int)
    g = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()
    assert g["context"]["window"] == 8192


def test_fork_and_delete_all(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    chat = client.post("/api/chat2/chats", headers=h, json={}).json()
    # seed two messages directly through the repo-backed SQL for speed
    c = sqlite3.connect(tmp_data_dir / "vllm-warden.db")
    for seq, role, text in ((1, "user", "q"), (2, "assistant", "a")):
        c.execute(
            "INSERT INTO messages(id,chat_id,seq,role,parts_json,settings_snapshot_json,"
            "created_at) VALUES (?,?,?,?,?,?,'2026-01-01T00:00:00.000Z')",
            (f"m{seq}", chat["id"], seq, role, f'[{{"type":"text","text":"{text}"}}]', "{}"),
        )
    c.commit()
    f = client.post(
        f"/api/chat2/chats/{chat['id']}/fork", headers=h, json={"at_seq": 1, "edited_text": "q2"}
    )
    assert f.status_code == 201 and f.json()["forked_from_chat_id"] == chat["id"]
    msgs = client.get(f"/api/chat2/chats/{f.json()['id']}", headers=h).json()["messages"]
    assert [m["parts"][0]["text"] for m in msgs] == ["q2"]
    assert client.delete("/api/chat2/chats", headers=h).json() == {"deleted": 2}


def test_enable_thinking_patch_round_trips(tmp_data_dir, client) -> None:
    """`False` must survive the settings merge -- `exclude_none` keeps it, but a
    truthiness check anywhere on the path would silently drop the only value
    that does anything."""
    h = _auth(tmp_data_dir, client)
    chat = client.post("/api/chat2/chats", headers=h, json={}).json()
    assert chat["settings"]["enable_thinking"] is True
    off = client.patch(
        f"/api/chat2/chats/{chat['id']}", headers=h, json={"settings": {"enable_thinking": False}}
    )
    assert off.status_code == 200 and off.json()["settings"]["enable_thinking"] is False
    again = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()
    assert again["chat"]["settings"]["enable_thinking"] is False
    on = client.patch(
        f"/api/chat2/chats/{chat['id']}", headers=h, json={"settings": {"enable_thinking": True}}
    )
    assert on.json()["settings"]["enable_thinking"] is True


def test_patching_the_title_clears_the_undo_target(tmp_data_dir, client) -> None:
    """A user rename retires the auto-title history: `title_prev` pointed at a
    title the machine picked, and offering to "undo" back to it after the user
    has said what they want would be nonsense."""
    h = _auth(tmp_data_dir, client)
    chat = client.post("/api/chat2/chats", headers=h, json={}).json()
    c = sqlite3.connect(tmp_data_dir / "vllm-warden.db")
    c.execute(
        "UPDATE chats SET title = 'Auto two', title_prev = 'Auto one' WHERE id = ?", (chat["id"],)
    )
    c.commit()
    assert (
        client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()["chat"]["title_prev"]
        == "Auto one"
    )

    p = client.patch(f"/api/chat2/chats/{chat['id']}", headers=h, json={"title": "Mine"})
    assert p.status_code == 200 and p.json()["title"] == "Mine"
    assert p.json()["title_prev"] is None

    # a patch that does NOT touch the title leaves the undo target alone
    c.execute("UPDATE chats SET title_prev = 'Auto one' WHERE id = ?", (chat["id"],))
    c.commit()
    p = client.patch(
        f"/api/chat2/chats/{chat['id']}", headers=h, json={"settings": {"temperature": 0.4}}
    )
    assert p.json()["title_prev"] == "Auto one"


def test_whoami_carries_the_contract_version(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    r = client.get("/api/chat2/_whoami", headers=h)
    assert r.status_code == 200
    assert r.json() == {"user_id": r.json()["user_id"], "contract": "1.0"}


def test_settings_scope_is_stored_and_echoed_opaque(tmp_data_dir, client) -> None:
    h = _auth(tmp_data_dir, client)
    r = client.post(
        "/api/chat2/chats",
        json={"settings": {"scope": {"instance_id": "i-1", "extra": [1]}}},
        headers=h,
    )
    assert r.status_code == 201
    assert r.json()["settings"]["scope"] == {"instance_id": "i-1", "extra": [1]}
    cid = r.json()["id"]
    r = client.patch(f"/api/chat2/chats/{cid}", json={"settings": {"temperature": 0.1}}, headers=h)
    assert r.json()["settings"]["scope"] == {
        "instance_id": "i-1",
        "extra": [1],
    }  # a PATCH of another key must not erase it


def test_oversized_opaque_settings_dicts_are_rejected(tmp_data_dir, client) -> None:
    """`scope` and `tool_policy` are stored opaque, so nothing downstream bounds
    them — a client could park megabytes of host-defined JSON in the settings
    blob of every chat. 4 KiB of serialised JSON is far above any real binding
    ({instance_id} is ~30 bytes) and far below a storage problem."""
    h = _auth(tmp_data_dir, client)
    for key in ("scope", "tool_policy"):
        big = client.post(
            "/api/chat2/chats", headers=h, json={"settings": {key: {"pad": "x" * 5000}}}
        )
        assert big.status_code == 422, big.text
        assert "too large" in big.text

        ok = client.post(
            "/api/chat2/chats", headers=h, json={"settings": {key: {"pad": "x" * 1000}}}
        )
        assert ok.status_code == 201, ok.text
        assert ok.json()["settings"][key] == {"pad": "x" * 1000}


def test_chat_reports_whether_its_model_is_loaded(tmp_data_dir, client) -> None:
    """#240: a chat pins its model by name and nothing keeps that name inside the
    loaded set. Every chat envelope says whether the turn route would serve it,
    so a client never has to infer the answer from a catalog that lists loaded
    models only."""
    h = _auth(tmp_data_dir, client)
    db_path = tmp_data_dir / "vllm-warden.db"
    _insert_loaded_model(db_path)
    chat = client.post("/api/chat2/chats", headers=h, json={"model": "qwen"}).json()
    assert chat["model_loaded"] is True
    # a chat with no model at all is not servable either
    assert client.post("/api/chat2/chats", headers=h, json={}).json()["model_loaded"] is False
    # the model gets unloaded underneath the chat (the observed failure)
    c = sqlite3.connect(db_path)
    c.execute("UPDATE models SET status = 'pulled' WHERE served_model_name = 'qwen'")
    c.commit()
    detail = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()["chat"]
    assert detail["model"] == "qwen" and detail["model_loaded"] is False
    listed = {r["id"]: r for r in client.get("/api/chat2/chats", headers=h).json()["chats"]}
    assert listed[chat["id"]]["model_loaded"] is False
    # the recovery: pin the chat to a loaded model
    _insert_loaded_model(db_path, served="other")
    p = client.patch(f"/api/chat2/chats/{chat['id']}", headers=h, json={"model": "other"})
    assert (
        p.status_code == 200 and p.json()["model"] == "other" and p.json()["model_loaded"] is True
    )
    f = client.post(f"/api/chat2/chats/{chat['id']}/fork", headers=h, json={"at_seq": 1})
    assert f.status_code == 201 and f.json()["model_loaded"] is True


def test_reasoning_effort_patch_round_trips_and_is_bounded(tmp_data_dir, client) -> None:
    """#241: the value is stored as given (it is the model template's own
    vocabulary, not ours) inside a tight charset, and `""` is how a chat goes
    back to the engine default -- a null would be dropped by `exclude_none`."""
    h = _auth(tmp_data_dir, client)
    chat = client.post("/api/chat2/chats", headers=h, json={}).json()
    assert "reasoning_effort" not in chat["settings"]
    r = client.patch(
        f"/api/chat2/chats/{chat['id']}", headers=h, json={"settings": {"reasoning_effort": "low"}}
    )
    assert r.status_code == 200 and r.json()["settings"]["reasoning_effort"] == "low"
    again = client.get(f"/api/chat2/chats/{chat['id']}", headers=h).json()
    assert again["chat"]["settings"]["reasoning_effort"] == "low"
    off = client.patch(
        f"/api/chat2/chats/{chat['id']}", headers=h, json={"settings": {"reasoning_effort": ""}}
    )
    assert off.json()["settings"]["reasoning_effort"] == ""
    for bad in ("x" * 33, "High Effort", "x;drop"):
        r = client.patch(
            f"/api/chat2/chats/{chat['id']}",
            headers=h,
            json={"settings": {"reasoning_effort": bad}},
        )
        assert r.status_code == 422, bad
