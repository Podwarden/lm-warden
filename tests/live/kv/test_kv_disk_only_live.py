"""Live GPU tests for the disk only KV cache store (``kv_store`` ``nvme``).

Spec 2026-11-23, sections 4 and 10.2 (see tests/live/kv/README.md for how to run them):

* ``test_disk_only_argv_and_pool``: the engine argv asks for ``"tiers": "nvme"``, the daemon's
  pool is 1 GiB and at least one namespace is staging.
* ``test_long_prompt_resumes_from_disk``: a long prompt saved, flushed to disk, then sent to
  a restarted engine loads exactly ``floor((P - 1) / B) * B`` tokens from NVMe, at least
  ``VW_KV_LIVE_MIN_SPEEDUP`` times faster than vanilla cold, with the save pass's output.
* ``test_ram_returns_to_empty``: RAM in use and the staging slots go back to 0 after the
  writes land and after the resume.
* ``test_segments_stay_out_of_the_page_cache``: at most 64 MiB of the segment files resident.
* ``test_weights_stay_in_the_page_cache``: the weight files' resident share drops by at most
  2 percentage points over the run.
* ``test_state_snapshots_saved_with_match_unit_16``: each GDN state namespace holds exactly
  the snapshots at vLLM's block aligned chunk ends and at ``floor((P - 1) / B) * B``.
* ``test_partial_head_block_is_private``: a GPU hit that ends inside a block, then a longer
  prompt that resumes from the store: the shared block is never written.
* ``test_write_volume_and_counters``: ``nvme_bytes_written`` matches the records written,
  ``/api/kv/usage`` agrees, and a resume counts one reload per position, not per group.

The tests run inside the api container: they read the daemon's socket, its NVMe manifest
and the page cache of the segment and weight files. The model's ``kv_store`` is put back
when the module ends. Keys, scopes and salts are never printed.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import socket
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.live.kv.conftest import KvLiveConfig
from tests.live.kv.pagecache import resident_bytes
from tests.live.kv.test_kv_live import (
    Shot,
    Warden,
    new_log_lines,
    prompt_ids,
    run_cmd,
    same_within_determinism,
    send,
)

GIB = 1 << 30
MIB = 1 << 20
#: Segment bytes allowed in the page cache (spec 10.2).
SEGMENT_RESIDENT_MAX = 64 * MIB
#: Percentage points the weights' resident share may drop (spec 10.2).
WEIGHTS_DROP_MAX_PP = 2.0
#: Seconds the writes of one prompt may take to land on disk.
FLUSH_TIMEOUT_S = 180.0
#: The partial head block case (spike, spec 4.6).
PARTIAL_X = 20000
PARTIAL_W = 10000


# ---------------------------------------------------------------------------
# The daemon, its manifest and the page cache
# ---------------------------------------------------------------------------


def daemon(cfg: KvLiveConfig, op: str, **fields: Any) -> dict[str, Any]:
    """One request on the daemon's socket (PROTOCOL.md, one JSON object per line)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(20.0)
        s.connect(cfg.socket)
        s.sendall((json.dumps({"op": op, "id": 1, **fields}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    r = json.loads(buf)
    assert r.get("ok"), f"daemon {op}: {r.get('error')}"
    return dict(r)


def stats(cfg: KvLiveConfig) -> dict[str, Any]:
    return dict(daemon(cfg, "stats")["stats"])


def wait_flushed(cfg: KvLiveConfig) -> tuple[dict[str, Any], float]:
    """Wait for every staging slot to leave RAM (the writes landed); (stats, seconds)."""
    t0 = time.monotonic()
    while True:
        st = stats(cfg)
        if int((st.get("staging") or {}).get("in_flight", -1)) == 0:
            return st, round(time.monotonic() - t0, 2)
        if time.monotonic() - t0 > FLUSH_TIMEOUT_S:
            return st, round(time.monotonic() - t0, 2)
        time.sleep(0.5)


@dataclass(frozen=True)
class Row:
    key: bytes
    parent: bytes
    grp: str
    length: int


def manifest(cfg: KvLiveConfig) -> dict[str, list[Row]]:
    """``ns hex -> rows`` of every namespace's NVMe manifest, read only."""
    out: dict[str, list[Row]] = {}
    for db in sorted(Path(cfg.segment_dir).glob("*/manifest.db")):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        try:
            rows = con.execute("SELECT key, parent, grp, len FROM blocks").fetchall()
        finally:
            con.close()
        out[db.parent.name] = [Row(bytes(k), bytes(p), str(g), int(n)) for k, p, g, n in rows]
    return out


def new_rows(before: dict[str, list[Row]], after: dict[str, list[Row]]) -> dict[str, list[Row]]:
    out = {}
    for ns, rows in after.items():
        old = {(r.key, r.grp) for r in before.get(ns, [])}
        out[ns] = [r for r in rows if (r.key, r.grp) not in old]
    return out


def chain(rows: list[Row]) -> list[bytes]:
    """The block keys of one new prefix chain, from block 0 on (its parent links)."""
    keys = {r.key for r in rows}
    child = {r.parent: r.key for r in rows}
    roots = [r.key for r in rows if r.parent not in keys]
    assert len(roots) == 1, f"want one new prefix chain, found {len(roots)}"
    out = [roots[0]]
    while out[-1] in child:
        out.append(child[out[-1]])
    return out


def segment_resident(cfg: KvLiveConfig) -> dict[str, int]:
    files = sorted(Path(cfg.segment_dir).rglob("*.seg"))
    return {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "resident": sum(resident_bytes(p) for p in files),
    }


def weights_resident(cfg: KvLiveConfig) -> dict[str, float]:
    files = {p.resolve() for p in Path(cfg.weights_dir).rglob("*.safetensors")}
    size = sum(p.stat().st_size for p in files)
    res = sum(resident_bytes(p) for p in files)
    return {"files": len(files), "bytes": size, "resident": res, "pct": 100.0 * res / max(1, size)}


# ---------------------------------------------------------------------------
# Expected positions
# ---------------------------------------------------------------------------


def last_cache_position(p: int, b: int) -> int:
    """Store tokens of a resume on a cold GPU (MTP nodrop or no MTP): ``floor((P-1)/B)*B``."""
    return (p - 1) // b * b


def snapshot_positions(p: int, b: int, m: int) -> list[int]:
    """Where vLLM's aligned split materialises GDN state for a cold prompt of ``p`` tokens
    with ``--max-num-batched-tokens m``: each non-final chunk ends at its end floored to
    ``b`` (the next chunk starts there), and the prefill stops at the last cache position."""
    assert m >= b, f"max-num-batched-tokens {m} is below the block size {b}"
    lcp = last_cache_position(p, b)
    out, start = [], 0
    while lcp > 0:
        end = start + m
        if end >= lcp:
            out.append(lcp)
            break
        start = end // b * b
        out.append(start)
    return out


def argv_value(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv[:-1]):
        if a == flag:
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_BLOCK_RES = (
    re.compile(r"LM Warden KV store on: block size (\d+)"),
    re.compile(r"[Ss]etting attention block size to (\d+) tokens"),
)


def find_block_size(cfg: KvLiveConfig) -> int:
    if cfg.block_size:
        return cfg.block_size
    if cfg.log_cmd:
        for line in reversed(run_cmd(cfg.log_cmd).stdout.splitlines()):
            for rx in _BLOCK_RES:
                m = rx.search(line)
                if m:
                    return int(m.group(1))
    pytest.skip("no block size: set VW_KV_LIVE_BLOCK_SIZE or VW_KV_LIVE_LOG_CMD")


# ---------------------------------------------------------------------------
# The shared run
# ---------------------------------------------------------------------------


@dataclass
class Disk:
    w: Warden
    model_id: str
    salt: str
    block: int = 0
    mbt: int = 0
    argv: list[str] = field(default_factory=list)
    vanilla: Shot | None = None
    saved: Shot | None = None
    resumed: Shot | None = None
    stats0: dict[str, Any] = field(default_factory=dict)
    man0: dict[str, list[Row]] = field(default_factory=dict)
    saved_rows: dict[str, list[Row]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    log_before: list[str] = field(default_factory=list, repr=False)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.w.cfg.long_size

    def holder(self) -> dict[str, Any]:
        """The run key's own report (``GET /api/me/key/cache``), signed in as its holder."""
        import httpx

        cfg = self.w.cfg
        login = httpx.post(
            f"{cfg.base_url}/api/forest/login",
            headers={"Authorization": f"Bearer {self.w.key}", "Origin": cfg.base_url},
            verify=cfg.verify_tls,
            timeout=30,
        )
        assert login.status_code == 200, f"/api/forest/login: HTTP {login.status_code}"
        r = httpx.get(
            f"{cfg.base_url}/api/me/key/cache",
            headers={"Authorization": f"Bearer {login.json()['token']}"},
            verify=cfg.verify_tls,
            timeout=30,
        )
        assert r.status_code == 200, f"/api/me/key/cache: HTTP {r.status_code}"
        return dict(r.json())

    def report(self) -> None:
        cfg = self.w.cfg
        cfg.report_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "salt": self.salt,
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "block": self.block,
            "max_num_batched_tokens": self.mbt,
            "shots": {
                k: s.summary()
                for k, s in (
                    ("vanilla", self.vanilla),
                    ("saved", self.saved),
                    ("resumed", self.resumed),
                )
                if s is not None
            },
            "extra": self.extra,
        }
        path = cfg.report_dir / f"kv-disk-{self.salt}.json"
        path.write_text(json.dumps(out, indent=1, default=str))
        print(f"[kv-disk] report: {path}")


def _reloads(h: dict[str, Any]) -> int:
    reuse = (h.get("store") or {}).get("reuse")
    assert reuse is not None, "the key's report has no reuse figures"
    return int(reuse["reloads"])


@pytest.fixture(scope="module")
def disk(kv_cfg: KvLiveConfig):
    cfg = kv_cfg
    missing = [
        v
        for v, have in (
            ("VW_KV_LIVE_SOCKET", cfg.socket),
            ("VW_KV_LIVE_SEGMENT_DIR", cfg.segment_dir),
            ("VW_KV_LIVE_WEIGHTS_DIR", cfg.weights_dir),
        )
        if not have
    ]
    if missing:
        pytest.skip(f"disk only tests need {', '.join(missing)} (run inside the api container)")
    w = Warden(cfg)
    w.mint_key()
    mid = w.model_id(cfg.model)
    start = w.row(mid)
    runtime = w.ok("GET", "/api/settings/runtime")
    d = Disk(w=w, model_id=mid, salt=secrets.token_hex(4))
    d.extra["start"] = {k: start.get(k) for k in ("kv_store", "status", "dtype")}
    d.extra["runtime"] = {k: runtime.get(k) for k in runtime if str(k).startswith("kv_store")}
    assert runtime.get("kv_store_nvme_dir"), "set kv_store_nvme_dir (the NVMe mount) first"
    try:
        if cfg.log_cmd:
            d.log_before = run_cmd(cfg.log_cmd).stdout.splitlines()
        # 1. Vanilla: kv_store off, fresh engine, the long prompt cold.
        w.unload(mid)
        w.patch(mid, kv_store="off")
        w.load(mid)
        d.vanilla = send(w, d.salt, d.size, "vanilla")
        # 2. Save pass: disk only, fresh engine, the same prompt.
        w.unload(mid)
        w.patch(mid, kv_store="nvme")
        w.load(mid)
        d.argv = list(w.ok("GET", f"/api/models/{mid}/effective-argv")["argv"])
        d.mbt = int(argv_value(d.argv, "--max-num-batched-tokens") or 0)
        d.block = find_block_size(cfg)
        d.extra["weights_after_load"] = weights_resident(cfg)
        d.extra["system_kv_store"] = w.ok("GET", "/api/system/kv-store")
        d.stats0 = stats(cfg)
        d.man0 = manifest(cfg)
        d.roles = {ns: str(v.get("role")) for ns, v in (d.stats0.get("by_namespace") or {}).items()}
        d.saved = send(w, d.salt, d.size, "save")
        st, secs = wait_flushed(cfg)
        d.extra["after_save"] = {
            "flush_s": secs,
            "ram_used": st.get("ram_used"),
            "staging": st.get("staging"),
            "segments": segment_resident(cfg),
        }
        man1 = manifest(cfg)
        d.saved_rows = new_rows(d.man0, man1)
        d.extra["stats_after_save"] = st
        # 3. Resume: engine restart, so only the disk can serve the prefix.
        w.restart(mid)
        time.sleep(6)  # the reports cache the daemon's stats for 5 s
        before = _reloads(d.holder())
        d.resumed = send(w, d.salt, d.size, "resume")
        st, secs = wait_flushed(cfg)
        time.sleep(6)
        after = _reloads(d.holder())
        d.extra["after_resume"] = {
            "flush_s": secs,
            "ram_used": st.get("ram_used"),
            "staging": st.get("staging"),
            "segments": segment_resident(cfg),
            "holder_reloads": {"before": before, "after": after},
        }
        d.extra["weights_after_resume"] = weights_resident(cfg)
        d.report()
        yield d
    finally:
        d.report()
        try:
            w.unload(mid)
            w.patch(mid, kv_store=start["kv_store"])
            if start.get("status") == "loaded":
                w.load(mid)
        finally:
            w.revoke_key()
            w.close()


# ---------------------------------------------------------------------------
# Tests (file order matters: the partial head case and the counters come last)
# ---------------------------------------------------------------------------


def test_disk_only_argv_and_pool(disk: Disk) -> None:
    kv = json.loads(argv_value(disk.argv, "--kv-transfer-config") or "{}")
    assert kv.get("kv_connector") == "LmwKVConnector"
    assert kv["kv_connector_extra_config"]["tiers"] == "nvme"
    d = disk.extra["system_kv_store"]["daemon"]
    disk.extra["pool"] = {
        "ram_granted_bytes": d.get("ram_granted_bytes"),
        "staging": d.get("staging"),
    }
    assert d.get("ram_granted_bytes") == GIB, f"pool {d.get('ram_granted_bytes')}, want 1 GiB"
    assert int((d.get("staging") or {}).get("namespaces") or 0) > 0, f"staging {d.get('staging')}"


def test_long_prompt_resumes_from_disk(disk: Disk) -> None:
    cfg = disk.w.cfg
    v, s, r = disk.vanilla, disk.saved, disk.resumed
    assert v and s and r
    want = last_cache_position(disk.size, disk.block)
    speedup = v.ttft_s / r.ttft_s
    ok_v, why_v = same_within_determinism(v, r, cfg.tie_margin)
    disk.extra["resume"] = {
        "want_nvme": want,
        "tiers": r.tiers,
        "ttft_cold_vanilla": round(v.ttft_s, 3),
        "ttft_save": round(s.ttft_s, 3),
        "ttft_resume": round(r.ttft_s, 3),
        "speedup": round(speedup, 2),
        "parity_save": "IDENTICAL" if s.token_ids == r.token_ids else "DIFFER",
        "parity_vanilla": why_v,
    }
    print(f"[kv-disk] resume: {disk.extra['resume']}")
    assert int(r.tiers.get("nvme") or 0) == want, f"nvme {r.tiers}, want exactly {want}"
    assert not r.tiers.get("ram"), f"a disk only resume served RAM tokens: {r.tiers}"
    assert not r.tiers.get("gpu"), f"the restarted engine had GPU tokens: {r.tiers}"
    assert r.token_ids == s.token_ids, "the resume differs from the save pass's cold output"
    assert ok_v, f"the resume differs from vanilla: {why_v}"
    assert (
        speedup >= cfg.min_speedup
    ), f"TTFT {r.ttft_s:.3f}s vs vanilla cold {v.ttft_s:.3f}s ({speedup:.1f}x)"


def test_ram_returns_to_empty(disk: Disk) -> None:
    for when in ("after_save", "after_resume"):
        x = disk.extra[when]
        staging = x.get("staging") or {}
        assert x.get("ram_used") == 0, f"{when}: ram_used {x.get('ram_used')}"
        assert staging.get("in_flight") == 0, f"{when}: staging {staging}"


def test_segments_stay_out_of_the_page_cache(disk: Disk) -> None:
    for when in ("after_save", "after_resume"):
        seg = disk.extra[when]["segments"]
        assert seg["files"] > 0, f"{when}: no segment file under the segment dir"
        assert (
            seg["resident"] <= SEGMENT_RESIDENT_MAX
        ), f"{when}: {seg['resident'] / MIB:.1f} MiB of segments in the page cache"


def test_weights_stay_in_the_page_cache(disk: Disk) -> None:
    a, b = disk.extra["weights_after_load"], disk.extra["weights_after_resume"]
    assert a["files"] > 0, "no *.safetensors under VW_KV_LIVE_WEIGHTS_DIR"
    drop = a["pct"] - b["pct"]
    disk.extra["weights_drop_pp"] = round(drop, 2)
    assert (
        drop <= WEIGHTS_DROP_MAX_PP
    ), f"weights resident {a['pct']:.1f}% -> {b['pct']:.1f}% (drop {drop:.1f} pp)"


def test_state_snapshots_saved_with_match_unit_16(disk: Disk) -> None:
    cfg = disk.w.cfg
    assert argv_value(disk.argv, "--prefix-match-unit") == str(cfg.match_unit)
    b, p = disk.block, disk.size
    assert disk.mbt, "set --max-num-batched-tokens in the model's extra_args"
    want = snapshot_positions(p, b, disk.mbt)
    chains = sorted(
        (ns for ns, role in disk.roles.items() if role in ("attention", "mtp")),
        key=lambda ns: disk.roles[ns] != "attention",
    )
    states = [ns for ns, role in disk.roles.items() if role == "gdn_state"]
    assert chains and states, f"roles {sorted(set(disk.roles.values()))}"
    keys = chain(disk.saved_rows[chains[0]])
    assert len(keys) == p // b, f"{len(keys)} attention blocks saved, want {p // b}"
    st0 = disk.stats0.get("by_namespace") or {}
    st1 = disk.extra["stats_after_save"].get("by_namespace") or {}
    got: dict[str, Any] = {}
    bad: list[str] = []
    for ns in states:
        rows = disk.saved_rows.get(ns, [])
        index = {k: i for i, k in enumerate(keys)}
        at = sorted((index[r.key] + 1) * b for r in rows if r.key in index)
        stray = sum(1 for r in rows if r.key not in index)
        lengths = {r.length for r in rows}
        grew = int((st1.get(ns) or {}).get("nvme_bytes") or 0) - int(
            (st0.get(ns) or {}).get("nvme_bytes") or 0
        )
        count = grew // max(lengths) if lengths else 0
        found = [
            int(daemon(cfg, "lookup", ns=ns, keys=[keys[q // b - 1].hex()])["n"]) for q in want
        ]
        got[ns[:8]] = {"positions": at, "stats_blocks": count, "lookup": found}
        if at != want or stray or count != len(want) or found != [1] * len(want):
            bad.append(f"{ns[:8]}: at {at}, stray {stray}, stats {count}, lookup {found}")
    disk.extra["snapshots"] = {"want": want, "got": got}
    print(f"[kv-disk] snapshots: want {want}, got {got}")
    assert not bad, f"state snapshots, want {want} in every GDN namespace: {bad}"


def flush_gpu(disk: Disk, tag: str) -> dict[str, Any]:
    """Evict vLLM's GPU prefix cache with unique prompts larger than it."""
    cfg = disk.w.cfg
    n = 20000
    count = max(1, math.ceil(cfg.gpu_flush_tokens / n))
    t0 = time.monotonic()
    for i in range(count):
        send(disk.w, f"{disk.salt}-flush-{tag}-{i}", n, "flush", max_tokens=1)
    wait_flushed(cfg)
    return {"prompts": count, "tokens": count * n, "s": round(time.monotonic() - t0, 1)}


def test_partial_head_block_is_private(disk: Disk) -> None:
    """Review Focus 4 (spec 4.6): W, a 16 aligned prefix of X whose tail checkpoint sits
    inside a block, is in the GPU cache; X then resumes with the GPU up to the block floor
    and the store for the rest; W's next answer is unchanged."""
    cfg, w = disk.w.cfg, disk.w
    b, u = disk.block, cfg.match_unit
    assert PARTIAL_W % u == 0 and PARTIAL_W % b != 0, (PARTIAL_W, u, b)
    x_ids = prompt_ids(f"{disk.salt}-x", PARTIAL_X)
    w_ids = x_ids[:PARTIAL_W]
    res: dict[str, Any] = {}
    try:
        x1 = send(w, disk.salt, PARTIAL_X, "x-cold", ids=x_ids)
        res["x_cold"] = {"tiers": x1.tiers, "ttft": round(x1.ttft_s, 3)}
        res["flush"] = flush_gpu(disk, "x")
        w1 = send(w, disk.salt, PARTIAL_W, "w-first", ids=w_ids)
        res["w_first"] = {"tiers": w1.tiers, "ttft": round(w1.ttft_s, 3)}
        assert not w1.tiers.get("gpu"), f"the flush left W on the GPU: {w1.tiers}"
        x2 = send(w, disk.salt, PARTIAL_X, "x-after-w", ids=x_ids)
        w2 = send(w, disk.salt, PARTIAL_W, "w-again", ids=w_ids)
        c_w = int(w2.tiers.get("gpu") or 0)
        want_gpu = c_w // b * b
        want_store = last_cache_position(PARTIAL_X, b) - want_gpu
        res.update(
            c_w=c_w,
            x_after_w={"tiers": x2.tiers, "ttft": round(x2.ttft_s, 3)},
            w_again={"tiers": w2.tiers, "ttft": round(w2.ttft_s, 3)},
            want_gpu=want_gpu,
            want_store=want_store,
            x_parity="IDENTICAL" if x2.token_ids == x1.token_ids else "DIFFER",
            w_parity="IDENTICAL" if w2.token_ids == w1.token_ids else "DIFFER",
        )
        print(f"[kv-disk] partial head: {res}")
        assert c_w == (PARTIAL_W - 1) // u * u, f"W's GPU hit {c_w}: its tail checkpoint is lost"
        assert c_w % b != 0, f"W's checkpoint {c_w} is on a block boundary: no partial block"
        assert int(x2.tiers.get("gpu") or 0) == want_gpu, f"X: {x2.tiers}, want gpu {want_gpu}"
        assert cfg.store_tokens(x2.tiers) == want_store, f"X: {x2.tiers}, want store {want_store}"
        assert x2.token_ids == x1.token_ids, "X after W differs from X's cold output"
        assert w2.token_ids == w1.token_ids, "W's next answer changed after X's load"
    finally:
        disk.extra["partial_head"] = res


def test_write_volume_and_counters(disk: Disk) -> None:
    cfg = disk.w.cfg
    st, _ = wait_flushed(cfg)
    rows = new_rows(disk.man0, manifest(cfg))
    blocks = sum(len(v) for v in rows.values())
    record_bytes = sum(r.length for v in rows.values() for r in v)
    sizes = sorted({r.length for v in rows.values() for r in v})
    w0 = int((disk.stats0.get("reuse") or {}).get("nvme_bytes_written") or 0)
    w1 = int((st.get("reuse") or {}).get("nvme_bytes_written") or 0)
    written = w1 - w0
    time.sleep(6)  # the reports cache the daemon's stats for 5 s
    usage = disk.w.ok("GET", "/api/kv/usage")
    by_model = {
        m["model_id"]: int((m.get("reuse") or {}).get("nvme_bytes_written") or 0)
        for m in usage.get("models", [])
    }
    st2 = stats(cfg)
    total = int((st2.get("reuse") or {}).get("nvme_bytes_written") or 0)
    hr = disk.extra["after_resume"]["holder_reloads"]
    positions = int(disk.resumed.tiers.get("nvme") or 0) // disk.block if disk.resumed else 0
    disk.extra["write_volume"] = {
        "blocks_written": blocks,
        "record_bytes_manifest": record_bytes,
        "record_sizes": sizes,
        "nvme_bytes_written_delta": written,
        "nvme_bytes_written_total": total,
        "usage_by_model": by_model,
        "holder_reloads_delta": hr["after"] - hr["before"],
        "positions_loaded": positions,
        "load_failed_lines": load_failed_lines(disk),
        "load_s": {
            k: (s.tiers.get("load_s") if s else None) for k, s in (("resume", disk.resumed),)
        },
    }
    print(f"[kv-disk] write volume: {disk.extra['write_volume']}")
    assert blocks > 0 and written > 0, "nothing was written to disk"
    assert (
        abs(written - record_bytes) <= cfg.write_tolerance * record_bytes
    ), f"nvme_bytes_written {written} vs {blocks} records of {sizes} B = {record_bytes}"
    assert (
        abs(sum(by_model.values()) - total) <= cfg.write_tolerance * total
    ), f"/api/kv/usage {by_model} vs stats {total}"
    assert positions > 0, "the resume loaded nothing"
    assert (
        hr["after"] - hr["before"] == positions
    ), f"holder reloads grew by {hr['after'] - hr['before']}, want one per position ({positions})"


def load_failed_lines(disk: Disk) -> int | None:
    """Failed loads in the engine log during the run (an empty window that ran out of
    retries, a block gone, a bad slot). Successful empty windows are not logged."""
    cfg = disk.w.cfg
    if not cfg.log_cmd:
        return None
    new = new_log_lines(disk.log_before, run_cmd(cfg.log_cmd).stdout.splitlines())
    return sum(1 for ln in new if "LM Warden KV load failed" in ln)
