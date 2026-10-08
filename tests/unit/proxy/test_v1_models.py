import json
import sqlite3

from app.db.repos.tokens import hash_token


def _seed(db_path):
    plaintext = "vw_validtoken1234567890abcdef12345"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        # 1 loaded, 1 pulled, 1 failed
        for mid, served, status in [
            ("qwen", "qwen", "loaded"),
            ("other", "other", "pulled"),
            ("dead", "dead", "failed"),
        ]:
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
                "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
                "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
                "VALUES (?,?,'r','main',?,1,'auto',4096,0.9,0,'[]',?,0,NULL,NULL)",
                (mid, served, json.dumps([0]), status),
            )
        db.commit()
        return plaintext


def test_v1_models_lists_only_loaded(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed(tmp_data_dir / "vllm-warden.db")
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == ["qwen"]
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["owned_by"] == "vllm-warden"


def test_v1_models_requires_bearer(tmp_data_dir, client):
    client.get("/healthz")
    _seed(tmp_data_dir / "vllm-warden.db")
    r = client.get("/v1/models")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# `max_model_len` (#4643-compatible): the window a client may actually use.
#
# The point of publishing it is that a client stops guessing, so the tests
# below pin the two ways a guess would still be forced: a NULL row (the
# operator never pinned a ceiling, so the model's own config decides), and a
# live override (the engine was launched with something the row does not say).
# ---------------------------------------------------------------------------


def _seed_one(db_path, *, max_model_len, hf_repo="r"):
    """One loaded model, with `max_model_len` written exactly as given (may be None)."""
    plaintext = "vw_validtoken1234567890abcdef12345"
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": [0]}),),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", plaintext[:8], hash_token(plaintext), "inference"),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error) "
            "VALUES (?,?,?,'main',?,1,'auto',?,0.9,0,'[]','loaded',0,NULL,NULL)",
            ("qwen", "qwen", hf_repo, json.dumps([0]), max_model_len),
        )
        db.commit()
    return plaintext


def _write_hf_config(hf_cache_dir, hf_repo, config):
    snap = hf_cache_dir / f"models--{hf_repo.replace('/', '--')}" / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(json.dumps(config))


def _models(client, plaintext):
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {plaintext}"})
    assert r.status_code == 200
    return r.json()["data"][0]


def test_v1_models_reports_the_rows_max_model_len(tmp_data_dir, client):
    client.get("/healthz")
    plaintext = _seed_one(tmp_data_dir / "vllm-warden.db", max_model_len=4096)
    assert _models(client, plaintext)["max_model_len"] == 4096


def test_v1_models_max_model_len_follows_a_live_override(tmp_data_dir, client):
    """A reload-with-config leaves the row alone, so the row is not the truth."""
    client.get("/healthz")
    plaintext = _seed_one(tmp_data_dir / "vllm-warden.db", max_model_len=4096)
    client.app.state.supervisor.get_overrides = lambda _mid: {"max_model_len": 8192}
    assert _models(client, plaintext)["max_model_len"] == 8192


def test_v1_models_falls_back_to_max_position_embeddings(tmp_data_dir, client):
    """A NULL row is not "unknown": both engines derive the window from the config."""
    client.get("/healthz")
    plaintext = _seed_one(tmp_data_dir / "vllm-warden.db", max_model_len=None)
    _write_hf_config(tmp_data_dir / "hf-cache", "r", {"max_position_embeddings": 32768})
    assert _models(client, plaintext)["max_model_len"] == 32768


def test_v1_models_override_of_none_drops_to_the_config(tmp_data_dir, client):
    """`{"max_model_len": None}` means "drop the flag", not "no override"."""
    client.get("/healthz")
    plaintext = _seed_one(tmp_data_dir / "vllm-warden.db", max_model_len=4096)
    _write_hf_config(tmp_data_dir / "hf-cache", "r", {"max_position_embeddings": 32768})
    client.app.state.supervisor.get_overrides = lambda _mid: {"max_model_len": None}
    assert _models(client, plaintext)["max_model_len"] == 32768


def test_v1_models_omits_max_model_len_when_nothing_states_one(tmp_data_dir, client):
    """Omitted, never null or 0 -- a client must be able to tell "unknown" apart."""
    client.get("/healthz")
    plaintext = _seed_one(tmp_data_dir / "vllm-warden.db", max_model_len=None)
    entry = _models(client, plaintext)
    assert "max_model_len" not in entry
    assert entry["id"] == "qwen"  # the rest of the entry is unaffected
