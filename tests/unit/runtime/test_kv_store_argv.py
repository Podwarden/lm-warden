"""``kv_store`` -> ``--kv-transfer-config`` argv, ``LMW_KV_KEY_FILE`` env, socket path."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.runtime.backends.vllm import VllmBackend
from app.runtime.backends.vllm.args import build_vllm_args
from app.runtime.backends.vllm.env import (
    ENGINE_SHM_PATH,
    HARD_LOCKED_ENV_KEYS,
    build_subprocess_env,
)
from app.runtime.kv_store import KvTierUnsupported, kv_launch_kwargs, socket_path

SOCK = "/dev/shm/lmw-kv.sock"
KEY = "/data/cobs2_network_key"
SHA = "a" * 40
SHA2 = "b" * 40
REVS = {"revision": SHA, "tokenizer_revision": SHA}


def _model(**over):
    base = dict(
        id="m1",
        served_model_name="x",
        hf_repo="a/b",
        hf_revision=None,
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        extra_env={},
    )
    base.update(over)
    return SimpleNamespace(**base)


def _flag(argv: list[str]) -> str | None:
    return argv[argv.index("--kv-transfer-config") + 1] if "--kv-transfer-config" in argv else None


def test_socket_path_is_pod_local_next_to_the_ram_pool():
    """R25: never on the data volume, which two api pods share during a rolling update."""
    s = SimpleNamespace(data_dir=Path("/var/lib/lmw"))
    assert socket_path(s) == f"{ENGINE_SHM_PATH}/lmw-kv.sock"
    assert socket_path(s, "/tmp/shm") == "/tmp/shm/lmw-kv.sock"
    assert not socket_path(s).startswith("/var/lib/lmw")


def test_off_and_missing_attribute_emit_nothing():
    assert _flag(build_vllm_args(_model(kv_store="off"), port=1)) is None
    assert _flag(build_vllm_args(_model(), port=1)) is None  # stand-in row without the column


@pytest.mark.parametrize("tiers", ["ram", "ram+nvme", "nvme"])
def test_tiers_emit_exact_connector_config(tiers):
    argv = build_vllm_args(_model(kv_store=tiers), port=1, kv_socket=SOCK, kv_revisions=REVS)
    assert json.loads(_flag(argv)) == {
        "kv_connector": "LmwKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "lmw_kv.connector",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "socket": SOCK,
            "tiers": tiers,
            "min_tokens": 2048,
            "revision": SHA,
            "tokenizer_revision": SHA,
            # The model row id: the daemon's owner of every namespace this engine registers,
            # released when the engine goes away (I3).
            "owner": "m1",
        },
    }


def test_flag_precedes_extra_args():
    argv = build_vllm_args(
        _model(kv_store="ram", extra_args=["--foo", "1"]), port=1, kv_socket=SOCK, kv_revisions=REVS
    )
    assert argv.index("--kv-transfer-config") < argv.index("--foo")


@pytest.mark.parametrize("tiers", ["ram", "ram+nvme", "nvme"])
def test_docker_driver_never_gets_the_connector_argv(tiers):
    """F1: a docker-driver engine is a sibling container; it cannot reach the daemon."""
    with pytest.raises(KvTierUnsupported, match="VW_ENGINE_DRIVER=docker"):
        build_vllm_args(
            _model(kv_store=tiers), port=1, driver="docker", kv_socket=SOCK, kv_revisions=REVS
        )
    # the local subprocess driver gets it
    local = build_vllm_args(
        _model(kv_store=tiers), port=1, driver="local", kv_socket=SOCK, kv_revisions=REVS
    )
    assert _flag(local)


def test_plan_passes_the_driver_so_a_docker_plan_is_refused():
    backend = VllmBackend()
    with pytest.raises(KvTierUnsupported):
        backend.plan(
            _model(kv_store="ram"),
            port=1,
            bind_host="0.0.0.0",
            engine_driver="docker",
            kv_socket=SOCK,
            kv_key_file=KEY,
            kv_revisions=REVS,
        )
    off = backend.plan(_model(kv_store="off"), port=1, bind_host="0.0.0.0", engine_driver="docker")
    assert _flag(off.argv) is None


def test_launch_kwargs_refuse_a_docker_driver(tmp_path):
    docker = SimpleNamespace(engine_driver="docker", db_path=tmp_path / "w.db")
    with pytest.raises(KvTierUnsupported):
        kv_launch_kwargs(docker, _model(kv_store="ram"))
    assert kv_launch_kwargs(docker, _model(kv_store="off")) == {}
    local = SimpleNamespace(engine_driver="local", db_path=tmp_path / "w.db", hf_cache_dir=None)
    assert kv_launch_kwargs(local, _model(kv_store="ram", hf_revision=SHA))["kv_socket"]


def test_enabled_without_socket_is_a_bug():
    with pytest.raises(ValueError, match="kv_socket"):
        build_vllm_args(_model(kv_store="ram"), port=1)


def test_enabled_without_revisions_is_a_bug():
    with pytest.raises(ValueError, match="kv_revisions"):
        build_vllm_args(_model(kv_store="ram"), port=1, kv_socket=SOCK)


# ---- F6: the namespace is keyed by the commit the engine loads -----------------------


def _cache(tmp_path, repo: str, ref: str, sha: str, *, hub: bool = False) -> Path:
    root = tmp_path / "hf"
    base = root / "hub" if hub else root
    refs = base / f"models--{repo.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / ref).write_text(sha + "\n")
    return root


def _settings(tmp_path, cache):
    return SimpleNamespace(engine_driver="local", db_path=tmp_path / "w.db", hf_cache_dir=cache)


@pytest.mark.parametrize("hub", [False, True])
def test_main_resolves_to_the_commit_in_the_cache(tmp_path, hub):
    cache = _cache(tmp_path, "a/b", "main", SHA, hub=hub)
    kw = kv_launch_kwargs(_settings(tmp_path, cache), _model(kv_store="ram", hf_revision="main"))
    assert kw["kv_revisions"] == {"revision": SHA, "tokenizer_revision": SHA}
    argv = build_vllm_args(_model(kv_store="ram", hf_revision="main"), port=1, **{
        k: v for k, v in kw.items() if k != "kv_key_file"
    })  # fmt: skip
    extra = json.loads(_flag(argv))["kv_connector_extra_config"]
    assert extra["revision"] == SHA and "main" not in extra.values()
    # an empty revision is vLLM's default branch, main
    kw = kv_launch_kwargs(_settings(tmp_path, cache), _model(kv_store="ram", hf_revision=None))
    assert kw["kv_revisions"]["revision"] == SHA


def test_a_moved_ref_changes_the_commit(tmp_path):
    cache = _cache(tmp_path, "a/b", "main", SHA)
    model = _model(kv_store="ram", hf_revision="main")
    assert kv_launch_kwargs(_settings(tmp_path, cache), model)["kv_revisions"]["revision"] == SHA
    _cache(tmp_path, "a/b", "main", SHA2)  # a re-pull moved main
    assert kv_launch_kwargs(_settings(tmp_path, cache), model)["kv_revisions"]["revision"] == SHA2


@pytest.mark.parametrize(
    "over",
    [
        {"hf_revision": "main"},  # nothing in the cache
        {"hf_revision": "dev"},  # a branch the cache has no ref for
        {"hf_revision": "main", "extra_args": ["--revision", SHA2]},
        {"hf_revision": "main", "extra_args": ["--tokenizer_revision=x"]},
        {"hf_revision": "main", "extra_args": ["--tokenizer", "o/t"]},
        {"hf_revision": "main", "tokenizer_repo": "o/tok"},  # tokenizer not in the cache
    ],
)
def test_an_unresolvable_revision_refuses_the_tier_for_the_load(tmp_path, over):
    from app.runtime.kv_store import KvRevisionUnresolved

    cache = _cache(tmp_path, "x/y", "main", SHA)  # another repo only
    if over.get("hf_revision") == "main" and "extra_args" in over or "tokenizer_repo" in over:
        cache = _cache(tmp_path, "a/b", "main", SHA)
    with pytest.raises(KvRevisionUnresolved):
        kv_launch_kwargs(_settings(tmp_path, cache), _model(kv_store="ram", **over))


def _snapshot(cache: Path, repo: str, sha: str) -> None:
    (cache / f"models--{repo.replace('/', '--')}" / "snapshots" / sha).mkdir(parents=True)


def test_a_separate_tokenizer_repo_resolves_its_own_commit(tmp_path):
    cache = _cache(tmp_path, "a/b", "main", SHA)
    _cache(tmp_path, "o/tok", "main", SHA2)
    _snapshot(cache, "o/tok", SHA2)
    kw = kv_launch_kwargs(
        _settings(tmp_path, cache),
        _model(kv_store="ram", hf_revision="main", tokenizer_repo="o/tok"),
    )
    assert kw["kv_revisions"] == {"revision": SHA, "tokenizer_revision": SHA2}


def test_env_points_at_key_file_never_the_key():
    assert "LMW_KV_KEY_FILE" in HARD_LOCKED_ENV_KEYS
    env = build_subprocess_env(
        _model(kv_store="ram"), hf_token="t", hf_cache_dir="/c", kv_key_file=KEY
    )
    assert env["LMW_KV_KEY_FILE"] == KEY


def test_env_absent_when_off():
    env = build_subprocess_env(_model(kv_store="off"), hf_token="t", hf_cache_dir="/c")
    assert "LMW_KV_KEY_FILE" not in env


def test_extra_env_cannot_set_it():
    with pytest.raises(ValueError, match="hard-locked"):
        build_subprocess_env(
            _model(extra_env={"LMW_KV_KEY_FILE": "/x"}), hf_token="t", hf_cache_dir="/c"
        )


def _opt(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_engine_starts_at_the_resolved_commits_when_the_tier_is_on(tmp_path):
    """F6: the engine loads exactly the commit the namespace is keyed by (no re-pull race)."""
    cache = _cache(tmp_path, "a/b", "main", SHA)
    model = _model(kv_store="ram", hf_revision="main")
    kw = kv_launch_kwargs(_settings(tmp_path, cache), model)
    argv = build_vllm_args(
        model, port=1, kv_socket=kw["kv_socket"], kv_revisions=kw["kv_revisions"]
    )
    assert _opt(argv, "--revision") == [SHA]
    assert _opt(argv, "--tokenizer-revision") == [SHA]
    assert "main" not in argv


def test_a_separate_tokenizer_repo_starts_at_its_own_commit(tmp_path):
    cache = _cache(tmp_path, "a/b", "main", SHA)
    _cache(tmp_path, "o/tok", "main", SHA2)
    _snapshot(cache, "o/tok", SHA2)
    model = _model(kv_store="ram", hf_revision="main", tokenizer_repo="o/tok")
    kw = kv_launch_kwargs(_settings(tmp_path, cache), model)
    argv = build_vllm_args(
        model, port=1, kv_socket=kw["kv_socket"], kv_revisions=kw["kv_revisions"]
    )
    assert _opt(argv, "--revision") == [SHA]
    assert _opt(argv, "--tokenizer-revision") == [SHA2]
    assert _opt(argv, "--tokenizer") == ["o/tok"]


@pytest.mark.parametrize("rev", ["main", None, SHA])
def test_kv_off_argv_is_unchanged(rev):
    """kv_store off: the row's own revision, no --tokenizer-revision, as before."""
    argv = build_vllm_args(_model(kv_store="off", hf_revision=rev), port=1)
    assert _opt(argv, "--revision") == ([rev] if rev else [])
    assert "--tokenizer-revision" not in argv
    # revisions handed in by mistake for an off row change nothing
    assert (
        build_vllm_args(_model(kv_store="off", hf_revision=rev), port=1, kv_revisions=REVS) == argv
    )


def test_a_pinned_model_commit_never_becomes_the_tokenizer_repos_revision(tmp_path):
    """I1: hf_revision is a commit of the model repo; the tokenizer repo uses its own ref."""
    cache = _cache(tmp_path, "o/tok", "main", SHA2)
    _snapshot(cache, "o/tok", SHA2)
    model = _model(kv_store="ram", hf_revision=SHA, tokenizer_repo="o/tok")
    kw = kv_launch_kwargs(_settings(tmp_path, cache), model)
    assert kw["kv_revisions"] == {"revision": SHA, "tokenizer_revision": SHA2}
    argv = build_vllm_args(
        model, port=1, kv_socket=kw["kv_socket"], kv_revisions=kw["kv_revisions"]
    )
    assert _opt(argv, "--tokenizer-revision") == [SHA2]


@pytest.mark.parametrize("with_ref", [False, True])
def test_a_tokenizer_repo_without_its_ref_and_snapshot_turns_the_tier_off(tmp_path, with_ref):
    """I1: no ref, or a ref whose snapshot is missing: tier off, never the model's commit."""
    from app.runtime.kv_store import KvRevisionUnresolved

    cache = _cache(tmp_path, "a/b", "main", SHA)
    if with_ref:
        _cache(tmp_path, "o/tok", "main", SHA2)  # ref only, the snapshot is not there
    model = _model(kv_store="ram", hf_revision=SHA, tokenizer_repo="o/tok")
    with pytest.raises(KvRevisionUnresolved):
        kv_launch_kwargs(_settings(tmp_path, cache), model)


# ---- Task 12: the owner reaches the connector; effective-argv holds no secret ---------


def test_effective_argv_carries_the_owner_and_no_secret(client, tmp_data_dir):
    import sqlite3

    from tests.unit.models.test_effective_argv import _auth, _seed_done

    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_done(db_path)
    refs = tmp_data_dir / "hf-cache" / "models--org--model" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text(SHA)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, dtype, max_model_len, gpu_memory_utilization, "
            "trust_remote_code, extra_args, status, pulled_bytes, pulled_total, last_error, "
            "kv_store) VALUES ('kvm', 'kvm', 'org/model', 'main', '[0]', 1, 'auto', 4096, 0.9, "
            "0, '[]', 'registered', 0, NULL, NULL, 'ram+nvme')"
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope, kv_isolation, kv_tenant_id) "
            "VALUES ('t1', 't1', 'vw_t1___', 'h', 'inference', 'isolated', 't1')"
        )
        db.commit()
    r = client.get("/api/models/kvm/effective-argv", headers=_auth(client))
    assert r.status_code == 200, r.text
    argv = r.json()["argv"]
    extra = json.loads(_flag(argv))["kv_connector_extra_config"]
    assert extra["owner"] == "kvm"
    from lmw_kv.scopes import isolation_scope, scope_tag, tenant_tag

    nk = client.app.state.cobs2_key.key
    joined = " ".join(argv)
    for secret in (
        nk.hex(),
        tenant_tag(nk, "t1").hex(),
        isolation_scope(nk, "t1", 0).hex(),
        scope_tag(isolation_scope(nk, "t1", 0)).hex(),
    ):
        assert secret not in joined
