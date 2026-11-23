"""Live GPU tests for per-key KV isolation (Plan 2), against a running LM Warden.

Spec 2026-11-21 Plan 2, section 8 (see tests/live/kv/README.md for how to run them):

* ``test_g7_isolation_each_tier``: keys A (isolated), B (sharing) and C (isolated). A sends a
  long prompt with a client salt. B and C send it with no salt and with A's client salt, B
  also with a reserved ``lmw1:`` value (A's own derived salt when the run can compute it).
  None of them gets a cached token or a tier token; A's repeat hits. Done three times: with
  A's blocks on the GPU, after an engine restart (RAM path) and after a daemon restart with
  RAM empty and the engine restarted (NVMe path).
* ``test_g8_gpu_isolation_without_daemon``: the GPU case with ``kv_store`` off and no daemon
  process at all.
* ``test_g11_parity_mixed_keys``: step 1's greedy parity rule, with A, B and a key D that is
  switched sharing to isolated to sharing in the middle of its session.
* ``test_g12_two_rows_one_namespace``: two rows on the same small weights, one ``ram``, one
  ``ram+nvme``, both loaded: no segment file is created or grows under their namespace.
* ``test_fair_flood_smoke``: ``kv_evict_policy`` ``fair``; a flooder sends 10x its share of the
  RAM tier (tier / active keys, Task 15 N1); the victim's resumed session still loads from RAM
  or NVMe.
* ``test_g7c_no_secret_in_logs`` (last, so it greps the whole run): the scope, scope tag,
  tenant tag and derived salt of every key used appear in no api log line, engine log line,
  history row, capture or ``effective-argv``.

Every value the tests derive (salts, scopes, tags) is a secret of the key holder: it is never
printed, written to the report or put in an assertion message.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import secrets
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.live.kv.conftest import KvLiveConfig
from tests.live.kv.test_kv_live import (
    Shot,
    Warden,
    new_log_lines,
    prompt_ids,
    run_cmd,
    same_within_determinism,
)

_UNSET: Any = object()


# ---------------------------------------------------------------------------
# Keys and requests
# ---------------------------------------------------------------------------


@dataclass
class Key:
    label: str
    id: str
    token: str = field(repr=False)


def mint(w: Warden, label: str, *, isolated: bool, capture: bool) -> Key:
    body = {"name": f"kv-iso-{label}-{secrets.token_hex(3)}", "expires_in_days": 1, "priority": 5}
    out = w.ok("POST", "/api/tokens", json=body)
    key = Key(label=label, id=out["id"], token=out["plaintext"])
    patch: dict[str, Any] = {}
    if isolated:
        patch["kv_isolation"] = "isolated"
    if capture:
        patch["capture_hours"] = 6
    if patch:
        w.ok("PATCH", f"/api/tokens/{key.id}", json=patch)
    return key


def send_as(
    w: Warden,
    key: Key,
    salt: str,
    size: int,
    phase: str,
    *,
    cache_salt: Any = _UNSET,
    model: str | None = None,
    log: list[str] | None = None,
) -> Shot:
    """``test_kv_live.send`` with a chosen key, a client ``cache_salt`` and model."""
    cfg = w.cfg
    body: dict[str, Any] = {
        "model": model or cfg.model,
        "prompt": prompt_ids(salt, size),
        "max_tokens": cfg.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "logprobs": 2,
    }
    if cache_salt is not _UNSET:
        body["cache_salt"] = cache_salt
    shot = Shot(size=size, phase=phase)
    t0 = time.perf_counter()
    with httpx.stream(
        "POST",
        f"{cfg.base_url}/v1/completions",
        headers={"Authorization": f"Bearer {key.token}"},
        json=body,
        verify=cfg.verify_tls,
        timeout=cfg.request_timeout_s,
    ) as r:
        assert r.status_code == 200, f"{phase} {size}: HTTP {r.status_code}"
        shot.request_id = r.headers.get("X-LMW-Request-Id", "")
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or {}
            if usage:
                shot.usage_cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            for ch in d.get("choices", []):
                if not shot.ttft_s:
                    shot.ttft_s = time.perf_counter() - t0
                shot.token_ids += ch.get("token_ids") or []
                for top in (ch.get("logprobs") or {}).get("top_logprobs") or []:
                    vals = sorted((top or {}).values(), reverse=True)
                    shot.margins.append(vals[0] - vals[1] if len(vals) > 1 else float("inf"))
    shot.total_s = time.perf_counter() - t0
    assert len(shot.token_ids) == cfg.max_tokens, f"{phase} {size}: {len(shot.token_ids)} tokens"
    if shot.request_id:
        if log is not None:
            log.append(shot.request_id)
        shot.turn = w.turn(shot.request_id)
    print(
        f"[kv-iso] {phase:>16} {key.label} {size:>6}: ttft {shot.ttft_s:.3f}s "
        f"cached {shot.usage_cached} hist {shot.turn.get('cached_tokens')} tiers {shot.tiers}"
    )
    return shot


def nothing_reused(s: Shot) -> bool:
    t = s.tiers
    return (
        not s.usage_cached
        and not s.turn.get("cached_tokens")
        and not t.get("gpu")
        and not t.get("ram")
        and not t.get("nvme")
    )


def load_row(w: Warden, model_id: str, served: str) -> None:
    """``Warden.load`` for a row other than the configured model."""
    w.ok("POST", f"/api/models/{model_id}/load")
    try:
        w.wait_status(model_id, "loaded")
    except AssertionError as e:
        why = str(w.row(model_id).get("last_error") or "")[:800]
        raise AssertionError(f"{served} did not load: {e}; last_error: {why}") from None
    model_ok(w, served)


def model_ok(w: Warden, model: str) -> None:
    """``Warden.wait_serving`` for any served name."""
    deadline = time.monotonic() + 600
    while True:
        r = httpx.post(
            f"{w.cfg.base_url}/v1/completions",
            headers={"Authorization": f"Bearer {w.key}"},
            json={"model": model, "prompt": "Hello", "max_tokens": 1},
            verify=w.cfg.verify_tls,
            timeout=120,
        )
        if r.status_code == 200:
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"{model} not serving: HTTP {r.status_code}")
        time.sleep(5.0)


# ---------------------------------------------------------------------------
# Derived secrets, for G7 (B presents A's salt) and G7c (grep)
# ---------------------------------------------------------------------------


def _hmac(nk: bytes, msg: bytes) -> bytes:
    return hmac.digest(nk, msg, "sha256")


def tenant_tag(nk: bytes, tenant_id: str) -> bytes:
    return _hmac(nk, b"lmw-tenant\x00v1\x00" + tenant_id.encode())[:16]


def iso_scope(nk: bytes, tenant_id: str, epoch: int) -> bytes:
    return _hmac(nk, b"lmw-scope\x00v1\x00" + tenant_id.encode() + b"\x00" + str(epoch).encode())


def scope_tag(scope: bytes) -> bytes:
    return hashlib.sha256(b"lmw-scope-tag\x00" + scope).digest()[:16] if scope else b""


def gpu_salt(nk: bytes, scope: bytes, client_salt: str) -> str:
    cs = client_salt.encode()
    msg = b"lmw-gpu\x00" + bytes([len(scope)]) + scope + struct.pack("<I", len(cs)) + cs
    return "lmw1:" + _hmac(nk, msg).hex()


def _cross_check(nk: bytes) -> None:
    """The encodings above equal lmw_kv.scopes' (when importable), so a grep for them is not
    vacuous. They are re-implemented here so the test does not trust the code under test."""
    try:
        from lmw_kv import scopes
    except ImportError:
        return
    tid = "x-check"
    sc = iso_scope(nk, tid, 3)
    assert scopes.tenant_tag(nk, tid) == tenant_tag(nk, tid)
    assert scopes.isolation_scope(nk, tid, 3) == sc
    assert scopes.scope_tag(sc) == scope_tag(sc)
    assert scopes.gpu_cache_salt(nk, sc, "s") == gpu_salt(nk, sc, "s")
    assert scopes.gpu_cache_salt(nk, b"", "lmw1:x") == gpu_salt(nk, b"", "lmw1:x")


@dataclass(repr=False)
class Secrets:
    nk: bytes
    rows: dict[str, tuple[str, int]]  # key id -> (kv_tenant_id, kv_epoch)

    def __repr__(self) -> str:
        return "Secrets(<redacted>)"

    def scope_of(self, key: Key) -> bytes:
        tid, epoch = self.rows[key.id]
        return iso_scope(self.nk, tid, epoch)


def load_secrets(cfg: KvLiveConfig, keys: list[Key]) -> Secrets | None:
    if not (cfg.network_key_file and cfg.db_path):
        return None
    line = Path(cfg.network_key_file).read_text().strip().splitlines()[0]
    version, _, hexkey = line.partition(":")
    assert version == "v1", "unexpected network key file format"
    nk = bytes.fromhex(hexkey)
    assert len(nk) == 32
    _cross_check(nk)
    con = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    try:
        rows = {}
        for k in keys:
            got = con.execute(
                "SELECT kv_tenant_id, kv_epoch FROM api_tokens WHERE id = ?", (k.id,)
            ).fetchone()
            assert got, f"key {k.label} not in the database"
            rows[k.id] = (str(got[0]), int(got[1]))
    finally:
        con.close()
    return Secrets(nk=nk, rows=rows)


# ---------------------------------------------------------------------------
# The shared run
# ---------------------------------------------------------------------------


@dataclass
class Iso:
    w: Warden
    model_id: str
    salt: str
    a: Key
    b: Key
    c: Key
    d: Key
    request_ids: list[str] = field(default_factory=list)
    client_salts: set[str] = field(default_factory=set)
    #: every key that ever had an isolated scope, and the reserved salts B sent
    reserved_sent: list[str] = field(default_factory=list, repr=False)
    #: request id -> the reserved salt B itself sent in it (a client value, kept in its
    #: capture by design, so exempt there)
    sent_by_rid: dict[str, str] = field(default_factory=dict, repr=False)
    log_before: list[str] = field(default_factory=list, repr=False)
    api_log_before: list[str] = field(default_factory=list, repr=False)
    extra: dict[str, Any] = field(default_factory=dict)

    def send(self, key: Key, salt: str, size: int, phase: str, **kw: Any) -> Shot:
        if isinstance(kw.get("cache_salt"), str):
            self.client_salts.add(kw["cache_salt"])
        return send_as(self.w, key, salt, size, phase, log=self.request_ids, **kw)

    def keys(self) -> list[Key]:
        return [self.a, self.b, self.c, self.d]

    def report(self) -> None:
        cfg = self.w.cfg
        cfg.report_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "salt": self.salt,
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "requests": len(self.request_ids),
            "extra": self.extra,
        }
        path = cfg.report_dir / f"kv-iso-{self.salt}.json"
        path.write_text(json.dumps(out, indent=1, default=str))
        print(f"[kv-iso] report: {path}")


def daemon_alive(cfg: KvLiveConfig) -> bool | None:
    if not cfg.daemon_alive_cmd:
        return None
    return run_cmd(cfg.daemon_alive_cmd).returncode == 0


def kill_daemon(cfg: KvLiveConfig) -> None:
    p = run_cmd(cfg.kill_daemon_cmd)
    assert p.returncode == 0, "the kill command found no daemon to kill"
    time.sleep(cfg.daemon_wait_s)


def set_store(iso: Iso, value: str) -> None:
    w, mid = iso.w, iso.model_id
    w.unload(mid)
    w.patch(mid, kv_store=value)
    w.load(mid)


@pytest.fixture(scope="module")
def iso(kv_cfg: KvLiveConfig):
    w = Warden(kv_cfg)
    w.mint_key()
    mid = w.model_id(kv_cfg.model)
    start = w.row(mid)
    runtime = w.ok("GET", "/api/settings/runtime")
    assert runtime.get("kv_store_nvme_dir"), "set kv_store_nvme_dir (the NVMe mount) first"
    keys: list[Key] = []
    salt = secrets.token_hex(4)
    try:
        a = mint(w, "A", isolated=True, capture=True)
        keys.append(a)
        b = mint(w, "B", isolated=False, capture=True)
        keys.append(b)
        c = mint(w, "C", isolated=True, capture=True)
        keys.append(c)
        d = mint(w, "D", isolated=False, capture=True)
        keys.append(d)
        r = Iso(w=w, model_id=mid, salt=salt, a=a, b=b, c=c, d=d)
        r.extra["start"] = {k: start.get(k) for k in ("kv_store", "status", "dtype")}
        r.extra["runtime"] = {k: runtime.get(k) for k in runtime if str(k).startswith("kv_")}
        if kv_cfg.log_cmd:
            r.log_before = run_cmd(kv_cfg.log_cmd).stdout.splitlines()
        set_store(r, "ram+nvme")
        yield r
    finally:
        try:
            if "r" in locals():
                r.report()
            policy = runtime.get("kv_evict_policy") or "lru"
            w.api("PATCH", "/api/settings/runtime", json={"kv_evict_policy": policy})
            w.unload(mid)
            w.patch(mid, kv_store=start["kv_store"])
            if start.get("status") == "loaded":
                w.load(mid)
        finally:
            for k in keys:
                w.api("DELETE", f"/api/tokens/{k.id}")
            w.revoke_key()
            w.close()


def iso_size(cfg: KvLiveConfig) -> int:
    return next((n for n in sorted(cfg.sizes) if n >= 20000), max(cfg.sizes))


# ---------------------------------------------------------------------------
# G7 and G8: no cross-key hit in any tier
# ---------------------------------------------------------------------------


def gpu_case(iso: Iso, tier: str) -> dict[str, Any]:
    """A writes; (reset for ``tier``); B and C try every salt form; A repeats and hits."""
    cfg = iso.w.cfg
    # The GPU case keeps all six prompts in the GPU prefix cache at once, so A's blocks are
    # still there for every try and for A's repeat (20k x 6 overflows a 16 GB card's cache).
    n = cfg.gpu_case_size if tier == "gpu" else iso_size(cfg)
    salt = f"{iso.salt}-g7-{tier}"
    a_cs = f"client-{tier}-{iso.salt}"
    iso.send(iso.a, salt, n, f"g7-{tier}-A", cache_salt=a_cs)
    if tier == "ram":
        iso.w.restart(iso.model_id)
    elif tier == "nvme":
        time.sleep(cfg.nvme_flush_s)  # the write-through to NVMe is asynchronous
        kill_daemon(cfg)
        iso.w.restart(iso.model_id)
    sec = load_secrets(cfg, [iso.a])
    if sec is not None:
        reserved = gpu_salt(sec.nk, sec.scope_of(iso.a), a_cs)  # A's own engine salt
        how = "A's derived salt"
    else:
        reserved = "lmw1:" + secrets.token_hex(32)
        how = "a random lmw1: value"
    iso.reserved_sent.append(reserved)
    tries = [
        (iso.b, _UNSET, "B no salt"),
        (iso.b, a_cs, "B with A's client salt"),
        (iso.b, reserved, f"B with {how}"),
        (iso.c, _UNSET, "C no salt"),
        (iso.c, a_cs, "C with A's client salt"),
    ]
    leaks = []
    result: dict[str, Any] = {"reserved_form": how}
    for key, cs, what in tries:
        s = iso.send(key, salt, n, f"g7-{tier}-{key.label}", cache_salt=cs)
        if cs is reserved:
            iso.sent_by_rid[s.request_id] = reserved
        result[what] = {"cached": s.usage_cached, "tiers": s.tiers}
        if not nothing_reused(s):
            leaks.append(what)
    again = iso.send(iso.a, salt, n, f"g7-{tier}-A-again", cache_salt=a_cs)
    result["A again"] = {"cached": again.usage_cached, "tiers": again.tiers}
    assert not leaks, f"{tier}: another key reused A's blocks: {leaks}"
    got = int(again.tiers.get(tier) or 0)
    if tier == "gpu" and again.tiers.get("gpu") is None:  # kv_store off: no tier columns
        got = int(again.turn.get("cached_tokens") or again.usage_cached or 0)
    assert got > 0, f"{tier}: A's repeat did not hit its own {tier} blocks: {again.tiers}"
    return result


def test_g7_isolation_each_tier(iso: Iso) -> None:
    cfg = iso.w.cfg
    if not cfg.kill_daemon_cmd:
        pytest.skip("set VW_KV_LIVE_KILL_DAEMON_CMD (the NVMe path restarts the daemon)")
    iso.w.restart(iso.model_id)
    for tier in ("gpu", "ram", "nvme"):
        iso.extra[f"g7_{tier}"] = gpu_case(iso, tier)


def test_g8_gpu_isolation_without_daemon(iso: Iso) -> None:
    """No daemon process at all: kv_store off (so the api does not start one), daemon killed.
    The proxy's GPU salt does not depend on the store."""
    cfg = iso.w.cfg
    if not (cfg.kill_daemon_cmd and cfg.daemon_alive_cmd):
        pytest.skip("set VW_KV_LIVE_KILL_DAEMON_CMD and VW_KV_LIVE_DAEMON_ALIVE_CMD")
    w, mid = iso.w, iso.model_id
    w.unload(mid)
    w.patch(mid, kv_store="off")
    if daemon_alive(cfg):
        kill_daemon(cfg)
    try:
        assert daemon_alive(cfg) is False, "a store daemon is still running"
        w.load(mid)
        iso.extra["g8"] = gpu_case(iso, "gpu")
        assert daemon_alive(cfg) is False, "the daemon came back during the test"
    finally:
        set_store(iso, "ram+nvme")


# ---------------------------------------------------------------------------
# G11: parity with mixed keys and a key switched mid-session
# ---------------------------------------------------------------------------


def test_g11_parity_mixed_keys(iso: Iso) -> None:
    cfg = iso.w.cfg
    w, mid = iso.w, iso.model_id
    n = iso_size(cfg)
    p, q = f"{iso.salt}-g11-p", f"{iso.salt}-g11-q"
    # Vanilla: kv_store off, fresh engine.
    set_store(iso, "off")
    vp = iso.send(iso.b, p, n, "g11-vanilla-p")
    vq = iso.send(iso.b, q, n, "g11-vanilla-q")
    set_store(iso, "ram+nvme")
    res: dict[str, Any] = {}
    bad: list[str] = []

    def check(name: str, ref: Shot, s: Shot) -> None:
        ok, why = same_within_determinism(ref, s, cfg.tie_margin)
        res[name] = {"parity": why, "cached": s.usage_cached, "tiers": s.tiers}
        if not ok:
            bad.append(f"{name}: {why}")

    check("A save", vp, iso.send(iso.a, p, n, "g11-A-save"))
    check("B save", vp, iso.send(iso.b, p, n, "g11-B-save"))
    check("D turn 1 sharing", vq, iso.send(iso.d, q, n, "g11-D-1"))
    w.ok("PATCH", f"/api/tokens/{iso.d.id}", json={"kv_isolation": "isolated"})
    d2 = iso.send(iso.d, q, n, "g11-D-2-isolated")
    check("D turn 2 isolated", vq, d2)
    assert nothing_reused(d2), f"D's isolated turn reused its shared blocks: {d2.tiers}"
    w.restart(mid)
    restored = {
        "A restore": iso.send(iso.a, p, n, "g11-A-restore"),
        "B restore": iso.send(iso.b, p, n, "g11-B-restore"),
        "D turn 3 isolated restore": iso.send(iso.d, q, n, "g11-D-3"),
    }
    for name, s in restored.items():
        check(name, vp if name[0] in "AB" else vq, s)
        assert s.external > 0, f"{name}: nothing came from the store: {s.tiers}"
    w.ok("PATCH", f"/api/tokens/{iso.d.id}", json={"kv_isolation": "sharing"})
    d4 = iso.send(iso.d, q, n, "g11-D-4-sharing")
    check("D turn 4 sharing again", vq, d4)
    res["D turn 4 reused"] = d4.external + int(d4.tiers.get("gpu") or 0)
    iso.extra["g11"] = res
    assert not bad, f"parity: {bad}"


# ---------------------------------------------------------------------------
# G12: two rows on one namespace, one ram and one ram+nvme
# ---------------------------------------------------------------------------


def segments(cfg: KvLiveConfig) -> dict[str, int]:
    """``path -> size`` of every NVMe segment file (VW_KV_LIVE_SEGMENTS_CMD prints
    ``<path> <size>`` lines)."""
    out: dict[str, int] = {}
    for line in run_cmd(cfg.segments_cmd).stdout.splitlines():
        path, _, size = line.strip().rpartition(" ")
        if path:
            out[path] = int(size)
    return out


def grown(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {p: s for p, s in after.items() if s > before.get(p, -1)}


def _unload_row(w: Warden, model_id: str) -> None:
    """Unload a row whatever state it is in (a failed row is left as it is)."""
    if w.row(model_id).get("status") in ("loaded", "loading"):
        w.ok("POST", f"/api/models/{model_id}/unload")
    deadline = time.monotonic() + 300
    while w.row(model_id).get("status") in ("loaded", "loading", "unloading"):
        assert time.monotonic() < deadline, f"row {model_id} never unloaded"
        time.sleep(3.0)


def test_g12_two_rows_one_namespace(iso: Iso) -> None:
    cfg = iso.w.cfg
    if not (cfg.g12_hf_repo and cfg.g12_revision and cfg.segments_cmd):
        pytest.skip("set VW_KV_LIVE_G12_HF_REPO, VW_KV_LIVE_G12_REVISION, VW_KV_LIVE_SEGMENTS_CMD")
    if len(cfg.g12_gpus) < 2:
        pytest.skip("set VW_KV_LIVE_G12_GPUS to two GPU indices: one loaded model per GPU")
    w = iso.w
    w.unload(iso.model_id)  # its GPU may be one of the two
    tag = secrets.token_hex(3)
    base = {
        "hf_repo": cfg.g12_hf_repo,
        "hf_revision": cfg.g12_revision,
        "tensor_parallel_size": 1,
        "dtype": cfg.g12_dtype,
        "max_model_len": 8192,
        "gpu_memory_utilization": cfg.g12_gpu_util,
        "extra_args": ["--max-num-seqs", "2"],
    }
    names = {"ram+nvme": f"kv-g12-nvme-{tag}", "ram": f"kv-g12-ram-{tag}"}
    gpu_of = {"ram+nvme": cfg.g12_gpus[0], "ram": cfg.g12_gpus[1]}
    ids: dict[str, str] = {}
    res: dict[str, Any] = {}
    try:
        for store, name in names.items():
            body = {**base, "served_model_name": name, "kv_store": store}
            out = w.ok("POST", "/api/models", json={**body, "gpu_indices": [gpu_of[store]]})
            ids[store] = out["id"]
            if w.row(ids[store])["status"] != "pulled":
                w.pull(ids[store])
        s0 = segments(cfg)
        load_row(w, ids["ram+nvme"], names["ram+nvme"])
        iso.send(iso.b, f"{iso.salt}-g12-0", 4096, "g12-nvme-control", model=names["ram+nvme"])
        time.sleep(cfg.nvme_flush_s)
        s1 = segments(cfg)
        control = grown(s0, s1)
        res["control_grown_files"] = len(control)
        assert control, "control: the ram+nvme row alone wrote nothing to NVMe"
        ns_dirs = {str(Path(p).parent) for p in control}
        load_row(w, ids["ram"], names["ram"])  # both rows loaded now
        s2 = segments(cfg)
        for store, name in names.items():
            for i in range(2):
                iso.send(iso.b, f"{iso.salt}-g12-{store}-{i}", 4096, f"g12-{store}", model=name)
        time.sleep(cfg.nvme_flush_s)
        s3 = segments(cfg)
        late = {p: s for p, s in grown(s2, s3).items() if str(Path(p).parent) in ns_dirs}
        res["files_grown_after_ram_row_loaded"] = len(late)
        res["namespace_files"] = len([p for p in s3 if str(Path(p).parent) in ns_dirs])
        assert not late, f"NVMe written under the shared namespace with a ram row loaded: {late}"
        # Second control: with the ram row gone, the ram+nvme row writes to NVMe again, so
        # the silence above came from the ram row's registration and nothing else.
        _unload_row(w, ids["ram"])
        iso.send(iso.b, f"{iso.salt}-g12-after", 4096, "g12-nvme-after", model=names["ram+nvme"])
        time.sleep(cfg.nvme_flush_s)
        again = {
            p: s for p, s in grown(s3, segments(cfg)).items() if str(Path(p).parent) in ns_dirs
        }
        res["files_grown_after_ram_row_unloaded"] = len(again)
        assert again, "control: NVMe did not resume after the ram row unloaded"
    finally:
        iso.extra["g12"] = res
        for mid in ids.values():
            _unload_row(w, mid)
            w.api("DELETE", f"/api/models/{mid}")
        w.load(iso.model_id)


# ---------------------------------------------------------------------------
# Fair flood smoke
# ---------------------------------------------------------------------------


def usage(w: Warden) -> dict[str, Any]:
    return dict(w.ok("GET", "/api/kv/usage"))


def test_fair_flood_smoke(iso: Iso) -> None:
    cfg = iso.w.cfg
    w, mid = iso.w, iso.model_id
    w.ok("PATCH", "/api/settings/runtime", json={"kv_evict_policy": "fair"})
    deadline = time.monotonic() + 30
    while usage(w).get("policy") != "fair":
        assert time.monotonic() < deadline, "the store never reported policy fair"
        time.sleep(2.0)
    victim = mint(w, "V", isolated=False, capture=False)
    flooder = mint(w, "F", isolated=False, capture=False)
    res: dict[str, Any] = {}
    try:
        n = iso_size(cfg)
        u0 = usage(w)
        iso.send(victim, f"{iso.salt}-flood-v", n, "flood-victim-save")
        time.sleep(cfg.nvme_flush_s)
        u1 = usage(w)
        ram_cap = int(u1["tiers"]["ram"]["capacity"])
        grew = int(u1["tiers"]["ram"]["used"]) - int(u0["tiers"]["ram"]["used"])
        assert grew > 0, "the victim's save added nothing to the RAM tier"
        per_token = max(1, grew // n)
        active = 2  # the victim and the flooder hold blocks during the flood (Task 15 N1)
        flood_tokens = int(cfg.flood_factor * ram_cap / active / per_token)
        if cfg.flood_max_tokens:
            flood_tokens = min(flood_tokens, cfg.flood_max_tokens)
        prompts = max(1, flood_tokens // n)
        res.update(
            ram_capacity=ram_cap,
            bytes_per_token=per_token,
            flood_tokens=prompts * n,
            flood_prompts=prompts,
            flood_tiers=round(prompts * n * per_token / ram_cap, 2),
        )
        t0 = time.monotonic()
        for i in range(prompts):
            send_as(w, flooder, f"{iso.salt}-flood-{i}", n, "flood")
        res["flood_s"] = round(time.monotonic() - t0, 1)
        u2 = usage(w)
        res["tiers_after_flood"] = u2["tiers"]
        w.restart(mid)  # the GPU cache is empty: only the store can serve the resume
        back = iso.send(victim, f"{iso.salt}-flood-v", n, "flood-victim-resume")
        res["resume"] = {"cached": back.usage_cached, "tiers": back.tiers}
        res["resume_store_fraction"] = round(back.external / n, 3)
        assert back.external > 0, f"the resumed session got nothing from RAM or NVMe: {back.tiers}"
    finally:
        iso.extra["fair_flood"] = res
        w.ok("PATCH", "/api/settings/runtime", json={"kv_evict_policy": "lru"})
        for k in (victim, flooder):
            w.api("DELETE", f"/api/tokens/{k.id}")


# ---------------------------------------------------------------------------
# G7c: no secret anywhere (last: it greps the whole run)
# ---------------------------------------------------------------------------


def needles(iso: Iso, sec: Secrets) -> dict[str, set[str]]:
    """Every derived value of this run, by kind, in the forms it could be written."""
    out: dict[str, set[str]] = {"tenant": set(), "scope": set(), "scope_tag": set(), "salt": set()}
    salts = {*iso.client_salts, ""}
    for k in iso.keys():
        tid, epoch = sec.rows[k.id]
        out["tenant"].add(tenant_tag(sec.nk, tid).hex())
        for e in range(epoch + 1):
            sc = iso_scope(sec.nk, tid, e)
            out["scope"].add(sc.hex())
            out["scope_tag"].add(scope_tag(sc).hex())
            for cs in salts:
                out["salt"].add(gpu_salt(sec.nk, sc, cs))
    for r in iso.reserved_sent:  # B's reserved salt as the proxy rewrites it
        out["salt"].add(gpu_salt(sec.nk, b"", r))
    for kind in out:
        out[kind] |= {v.upper() for v in out[kind]}
        out[kind] |= {v[5:] for v in out[kind] if v.lower().startswith("lmw1:")}
    return out


def haystacks(iso: Iso) -> dict[str, list[tuple[str, str]]]:
    """``source -> [(request id or "", text)]`` for every place a secret must not be."""
    cfg, w = iso.w.cfg, iso.w
    hay: dict[str, list[tuple[str, str]]] = {}
    if cfg.api_log_cmd:
        hay["api log"] = [("", run_cmd(cfg.api_log_cmd, timeout=300).stdout)]
    if cfg.log_dir:  # every engine's whole log (each row's, every run), rotated ones too
        hay["engine logs"] = [
            ("", p.read_text("utf-8", "replace"))
            for p in sorted(Path(cfg.log_dir).iterdir())
            if p.is_file() and ".log" in p.name
        ]
    elif cfg.log_cmd:
        lines = new_log_lines(iso.log_before, run_cmd(cfg.log_cmd).stdout.splitlines())
        hay["engine log"] = [("", "\n".join(lines))]
    hay["history"], hay["captures"] = [], []
    for rid in iso.request_ids:
        hay["history"].append((rid, w.api("GET", f"/api/requests/{rid}").text))
        c = w.api("GET", f"/api/capture/{rid}")
        if c.status_code == 200:
            hay["captures"].append((rid, c.text))
    if cfg.capture_dir:
        hay["capture files"] = []
        for p in Path(cfg.capture_dir).rglob("*.json.gz"):
            try:
                text = gzip.decompress(p.read_bytes()).decode("utf-8", "replace")
            except OSError:
                continue
            hay["capture files"].append((p.name.split(".", 1)[0], text))
    argv = w.ok("GET", f"/api/models/{iso.model_id}/effective-argv")
    hay["effective-argv"] = [("", json.dumps(argv))]
    return hay


def exempt(iso: Iso, rid: str) -> set[str]:
    """What the client itself sent in request ``rid``: its own body is kept on purpose."""
    v = iso.sent_by_rid.get(rid)
    if not v:
        return set()
    return {v, v.upper(), v[5:], v[5:].upper()}


def test_g7c_no_secret_in_logs(iso: Iso) -> None:
    cfg = iso.w.cfg
    sec = load_secrets(cfg, iso.keys())
    if sec is None:
        pytest.skip("set VW_KV_LIVE_NETWORK_KEY_FILE and VW_KV_LIVE_DB_PATH")
    if not iso.request_ids:
        pytest.skip("no request of this run to check")
    found = needles(iso, sec)
    hay = haystacks(iso)
    sizes = {src: sum(len(t) for _, t in items) for src, items in hay.items()}
    iso.extra["g7c"] = {
        "requests": len(iso.request_ids),
        "needles": {k: len(v) for k, v in found.items()},
        "haystack_chars": sizes,
        "captures": len(hay["captures"]),
        "capture_files": len(hay.get("capture files", [])),
        "exempt_client_values": len(iso.sent_by_rid),
    }
    # Each source must hold something, or the grep proves nothing.
    assert sizes["history"] > 0, "no history rows read"
    assert hay["captures"], "no capture of this run to check"
    if cfg.api_log_cmd:
        assert sizes["api log"] > 0, "the api log command printed nothing"
    hits = sorted(
        {
            f"{kind} in {src}"
            for src, items in hay.items()
            for rid, text in items
            for kind, values in found.items()
            if any(v and v in text for v in values - exempt(iso, rid))
        }
    )
    assert not hits, f"derived secrets found (values withheld): {hits}"
