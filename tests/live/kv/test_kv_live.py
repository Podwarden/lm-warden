"""Live GPU tests for the KV-cache tiers (lmw_kv), against a running LM Warden.

What they prove on real hardware (see tests/live/kv/README.md for how to run them):

* ``test_parity_after_restore``: greedy 64-token output after a restore from the store
  equals a vanilla run (kv_store off), within vLLM's own run-to-run determinism.
* ``test_restore_after_engine_restart``: after an engine restart the turn is served from
  the RAM or NVMe tier (history shows it) and TTFT is at least 5x lower than vanilla cold
  at 20k+ tokens.
* ``test_cached_tokens_include_connector_tokens``: ruling R24's assumption that vLLM's
  ``cached_tokens`` includes the tokens the connector loaded.
* ``test_fail_open_when_daemon_killed_mid_load``: the store daemon is killed while a long
  prompt loads; the request completes with vanilla output and the engine keeps serving.
* ``test_nvme_tier_serves_after_daemon_restart``: a fresh daemon (empty RAM pool) serves
  the prompts from NVMe.
* ``test_old_namespace_never_loads``: after the model's revision changes, nothing stored
  under the old revision is loaded.

The tests share one module-scoped run (vanilla pass, save pass, engine restart, restore
pass) and run in file order. Each pass uses prompts salted with a fresh run id, so blocks
left on NVMe by an earlier run never count. The model's ``kv_store`` and revision are put
back when the module ends.

Nothing is hard-coded: the URL, the admin token and the model come from VW_KV_LIVE_* env
vars (tests/live/kv/conftest.py). The admin token is never printed.
"""

from __future__ import annotations

import json
import random
import secrets
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import pytest

from tests.live.kv.conftest import KvLiveConfig

#: Model statuses a load or unload passes through on its way to a stable one.
_TRANSIENT = {"loading", "unloading", "pulling"}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class Warden:
    """The control API (admin token) and the data plane (a short-lived inference key)."""

    def __init__(self, cfg: KvLiveConfig) -> None:
        self.cfg = cfg
        self.http = httpx.Client(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {cfg.admin_token}"},
            verify=cfg.verify_tls,
            timeout=60.0,
        )
        self.key: str | None = None
        self.key_id: str | None = None

    def close(self) -> None:
        self.http.close()

    def api(self, method: str, path: str, **kw: Any) -> httpx.Response:
        return self.http.request(method, path, **kw)

    def ok(self, method: str, path: str, **kw: Any) -> Any:
        r = self.api(method, path, **kw)
        if r.status_code >= 400:
            raise AssertionError(f"{method} {path}: HTTP {r.status_code} {r.text[:400]}")
        return r.json() if r.content else None

    # -- inference key ------------------------------------------------------

    def mint_key(self) -> None:
        body = {"name": f"kv-live-{secrets.token_hex(3)}", "expires_in_days": 1, "priority": 9}
        out = self.ok("POST", "/api/tokens", json=body)
        self.key, self.key_id = out["plaintext"], out["id"]

    def revoke_key(self) -> None:
        if self.key_id:
            self.api("DELETE", f"/api/tokens/{self.key_id}")
            self.key = self.key_id = None

    # -- model control ------------------------------------------------------

    def model_id(self, served: str) -> str:
        for m in self.ok("GET", "/api/models")["models"]:
            if m["served_model_name"] == served:
                return str(m["id"])
        raise AssertionError(f"no model with served name {served!r}")

    def row(self, model_id: str) -> dict[str, Any]:
        return dict(self.ok("GET", f"/api/models/{model_id}/settings"))

    def wait_status(self, model_id: str, want: str) -> None:
        deadline = time.monotonic() + self.cfg.load_timeout_s
        while True:
            st = self.row(model_id)["status"]
            if st == want:
                return
            if st == "failed":
                raise AssertionError(f"model {model_id} failed while waiting for {want!r}")
            if st not in _TRANSIENT and time.monotonic() > deadline - self.cfg.load_timeout_s + 60:
                # Not moving towards ``want`` a minute after the call: say so early.
                raise AssertionError(f"model {model_id}: status {st!r}, wanted {want!r}")
            if time.monotonic() > deadline:
                raise AssertionError(f"model {model_id}: still {st!r} after timeout")
            time.sleep(3.0)

    def unload(self, model_id: str) -> None:
        if self.row(model_id)["status"] in ("loaded", "loading"):
            self.ok("POST", f"/api/models/{model_id}/unload")
        self.wait_status(model_id, "pulled")

    def load(self, model_id: str) -> None:
        self.ok("POST", f"/api/models/{model_id}/load")
        self.wait_status(model_id, "loaded")
        self.wait_serving()

    def restart(self, model_id: str) -> None:
        """Unload and load: a fresh engine, so vLLM's GPU prefix cache is empty."""
        self.unload(model_id)
        self.load(model_id)

    def patch(self, model_id: str, **fields: Any) -> None:
        self.ok("PATCH", f"/api/models/{model_id}/settings", json=fields)

    def pull(self, model_id: str) -> None:
        self.ok("POST", f"/api/models/{model_id}/pull")
        self.wait_status(model_id, "pulled")

    def wait_serving(self) -> None:
        """The router can answer 529 for a while after 'loaded'; wait for a real reply."""
        deadline = time.monotonic() + 600
        while True:
            r = httpx.post(
                f"{self.cfg.base_url}/v1/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.cfg.model, "prompt": "Hello", "max_tokens": 1},
                verify=self.cfg.verify_tls,
                timeout=120,
            )
            if r.status_code == 200:
                return
            if time.monotonic() > deadline:
                raise AssertionError(f"engine not serving: HTTP {r.status_code} {r.text[:200]}")
            time.sleep(5.0)

    # -- history ------------------------------------------------------------

    def turn(self, request_id: str) -> dict[str, Any]:
        """The turn's history row. The tier columns land after pop_hits, so retry a bit."""
        for _ in range(20):
            r = self.api("GET", f"/api/requests/{request_id}", params={"wait_ms": 5000})
            if r.status_code == 200:
                return dict(r.json())
            time.sleep(1.0)
        raise AssertionError(f"request {request_id} never reached the history: {r.status_code}")


# ---------------------------------------------------------------------------
# One prompt, streamed
# ---------------------------------------------------------------------------


@dataclass
class Shot:
    size: int
    phase: str
    request_id: str = ""
    ttft_s: float = 0.0
    total_s: float = 0.0
    token_ids: list[int] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    usage_cached: int | None = None
    turn: dict[str, Any] = field(default_factory=dict)

    @property
    def tiers(self) -> dict[str, Any]:
        return dict(self.turn.get("tiers") or {})

    @property
    def external(self) -> int:
        t = self.tiers
        return int(t.get("ram") or 0) + int(t.get("nvme") or 0)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("margins")
        d["token_ids"] = len(self.token_ids)
        d["turn"] = {
            k: self.turn.get(k) for k in ("ttft_s", "cached_tokens", "prompt_tokens", "tiers")
        }
        return d


def prompt_ids(salt: str, n: int) -> list[int]:
    """Seeded random token ids, as in the spike's tools/client.py, salted per run."""
    rng = random.Random(f"{salt}:{n}")
    return [rng.randrange(1000, 140000) for _ in range(n)]


def send(w: Warden, salt: str, size: int, phase: str) -> Shot:
    cfg = w.cfg
    body = {
        "model": cfg.model,
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
    shot = Shot(size=size, phase=phase)
    t0 = time.perf_counter()
    with httpx.stream(
        "POST",
        f"{cfg.base_url}/v1/completions",
        headers={"Authorization": f"Bearer {w.key}"},
        json=body,
        verify=cfg.verify_tls,
        timeout=cfg.request_timeout_s,
    ) as r:
        assert r.status_code == 200, f"{phase} {size}: HTTP {r.status_code} {r.read()[:300]!r}"
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
        shot.turn = w.turn(shot.request_id)
    print(
        f"[kv-live] {phase:>10} {size:>6}: ttft {shot.ttft_s:.3f}s total {shot.total_s:.2f}s "
        f"cached {shot.usage_cached} tiers {shot.tiers}"
    )
    return shot


def same_within_determinism(a: Shot, b: Shot, tie_margin: float) -> tuple[bool, str]:
    """Identical tokens, or a first divergence at a bf16 near-tie (spike Q2 rule)."""
    if a.token_ids == b.token_ids:
        return True, "IDENTICAL"
    i = next(i for i, (x, y) in enumerate(zip(a.token_ids, b.token_ids, strict=False)) if x != y)
    m = min(
        a.margins[i] if i < len(a.margins) else 1.0, b.margins[i] if i < len(b.margins) else 1.0
    )
    if m < tie_margin:
        return True, f"DIFFER at {i} on a near-tie (top-2 margin {m:.4f})"
    return False, f"DIFFER at {i} (top-2 margin {m:.4f})"


def run_cmd(cmd: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S602 -- an operator-supplied command, by design
        cmd, shell=True, capture_output=True, text=True, timeout=timeout, check=False
    )


# ---------------------------------------------------------------------------
# The shared run
# ---------------------------------------------------------------------------


@dataclass
class Run:
    w: Warden
    model_id: str
    salt: str
    vanilla: dict[int, Shot] = field(default_factory=dict)
    saved: dict[int, Shot] = field(default_factory=dict)
    restored: dict[int, Shot] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def report(self) -> None:
        cfg = self.w.cfg
        cfg.report_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "salt": self.salt,
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "vanilla": {n: s.summary() for n, s in self.vanilla.items()},
            "saved": {n: s.summary() for n, s in self.saved.items()},
            "restored": {n: s.summary() for n, s in self.restored.items()},
            "extra": self.extra,
        }
        path = cfg.report_dir / f"kv-live-{self.salt}.json"
        path.write_text(json.dumps(out, indent=1, default=str))
        print(f"[kv-live] report: {path}")


@pytest.fixture(scope="module")
def run(kv_cfg: KvLiveConfig):
    w = Warden(kv_cfg)
    w.mint_key()
    mid = w.model_id(kv_cfg.model)
    start = w.row(mid)
    runtime = w.ok("GET", "/api/settings/runtime")
    r = Run(w=w, model_id=mid, salt=secrets.token_hex(4))
    r.extra["start"] = {k: start.get(k) for k in ("kv_store", "hf_revision", "status", "dtype")}
    r.extra["runtime"] = {k: runtime.get(k) for k in runtime if str(k).startswith("kv_store")}
    assert runtime.get("kv_store_nvme_dir"), "set kv_store_nvme_dir (the NVMe mount) first"
    try:
        # 1. Vanilla: kv_store off, fresh engine, cold prefill for every size.
        w.unload(mid)
        w.patch(mid, kv_store="off")
        w.load(mid)
        for n in kv_cfg.sizes:
            r.vanilla[n] = send(w, r.salt, n, "vanilla")
        # 2. Save pass: ram+nvme, fresh engine, the same prompts (stored write-through).
        w.unload(mid)
        w.patch(mid, kv_store="ram+nvme")
        w.load(mid)
        argv = w.ok("GET", f"/api/models/{mid}/effective-argv")["argv"]
        r.extra["argv_kv"] = next(
            (argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--kv-transfer-config"), None
        )
        for n in kv_cfg.sizes:
            r.saved[n] = send(w, r.salt, n, "save")
        # 3. Restore pass: engine restart, so only the store can serve the prefix.
        w.restart(mid)
        for n in kv_cfg.sizes:
            r.restored[n] = send(w, r.salt, n, "restore")
        r.report()
        yield r
    finally:
        r.report()
        try:
            w.unload(mid)
            w.patch(mid, kv_store=start["kv_store"], hf_revision=start["hf_revision"])
            if start.get("status") == "loaded":
                w.load(mid)
        finally:
            w.revoke_key()
            w.close()


# ---------------------------------------------------------------------------
# Tests (file order matters: later tests restart the engine and the daemon)
# ---------------------------------------------------------------------------


def test_kv_transfer_config_asks_for_recompute(run: Run) -> None:
    cfg = json.loads(run.extra["argv_kv"])
    assert cfg["kv_connector"] == "LmwKVConnector"
    assert cfg["kv_load_failure_policy"] == "recompute"
    assert cfg["kv_connector_extra_config"]["tiers"] == "ram+nvme"


def test_parity_after_restore(run: Run) -> None:
    # Parity is only evidence if the long prompts really came from the store.
    for n in run.w.cfg.sizes:
        if n >= 20000:
            assert run.restored[n].external > 0, f"{n}: nothing restored: {run.restored[n].tiers}"
    bad = []
    for n in run.w.cfg.sizes:
        ok, why = same_within_determinism(run.vanilla[n], run.restored[n], run.w.cfg.tie_margin)
        run.extra[f"parity_{n}"] = why
        print(f"[kv-live] parity {n}: {why}")
        if not ok:
            bad.append(f"{n}: {why}")
    assert not bad, f"restore differs from vanilla: {bad}"


def test_restore_after_engine_restart(run: Run) -> None:
    cfg = run.w.cfg
    long_sizes = [n for n in cfg.sizes if n >= 20000]
    assert long_sizes, "VW_KV_LIVE_SIZES needs a size of 20000 or more"
    for n in long_sizes:
        v, s = run.vanilla[n], run.restored[n]
        assert s.external > 0, f"{n}: no RAM/NVMe tokens in history: {s.tiers}"
        speedup = v.ttft_s / s.ttft_s
        run.extra[f"speedup_{n}"] = round(speedup, 2)
        assert (
            speedup >= cfg.min_speedup
        ), f"{n}: TTFT {s.ttft_s:.3f}s vs vanilla cold {v.ttft_s:.3f}s ({speedup:.1f}x)"


def test_cached_tokens_include_connector_tokens(run: Run) -> None:
    """R24: vLLM's cached_tokens counts connector-loaded tokens, so after an engine
    restart (empty GPU cache) cached >= ram + nvme and served_gpu = cached - ram - nvme."""
    checked = 0
    for n in run.w.cfg.sizes:
        s = run.restored[n]
        if s.external == 0:
            continue
        checked += 1
        cached = s.turn.get("cached_tokens")
        t = s.tiers
        run.extra[f"r24_{n}"] = {"cached": cached, "usage_cached": s.usage_cached, **t}
        assert cached is not None and cached >= s.external, f"{n}: cached {cached}, tiers {t}"
        assert t.get("gpu") == cached - s.external, f"{n}: gpu {t.get('gpu')}, cached {cached}"
        total = int(t.get("gpu") or 0) + s.external
        assert total == cached, f"{n}: gpu+ram+nvme {total} != cached {cached}"
    assert checked, "no restored turn to check"


def new_log_lines(before: list[str], after: list[str]) -> list[str]:
    """The lines of ``after`` written since ``before`` was read (both are log tails)."""
    if not before:
        return after
    for i in range(len(after) - 1, -1, -1):
        if after[i] == before[-1]:
            return after[i + 1 :]
    return after


def test_fail_open_when_daemon_killed_mid_load(run: Run) -> None:
    """Review Focus 1: the daemon dies while a long prompt loads.

    Two outcomes are correct. A kill before ``begin_load`` fails the load and vLLM
    recomputes (that path is pinned down by ``test_failed_load_recomputes``). A kill
    after it lets the copy finish, because the RAM file stays mapped in the worker;
    only the hit record is lost. Either way the output is vanilla's and the engine
    keeps serving, with no engine traceback.
    """
    cfg = run.w.cfg
    if not cfg.kill_daemon_cmd:
        pytest.skip("set VW_KV_LIVE_KILL_DAEMON_CMD")
    n = max(cfg.sizes)
    run.w.restart(run.model_id)  # empty GPU cache: the 50k prompt must load from the store
    before = run_cmd(cfg.log_cmd).stdout.splitlines() if cfg.log_cmd else []
    killed: dict[str, Any] = {}

    def kill() -> None:
        time.sleep(cfg.kill_delay_s)
        p = run_cmd(cfg.kill_daemon_cmd)
        killed.update(rc=p.returncode, out=(p.stdout + p.stderr)[-300:])

    t = threading.Thread(target=kill)
    t.start()
    shot = send(run.w, run.salt, n, "kill")
    t.join()
    run.extra["kill"] = {"cmd_result": killed, "shot": shot.summary()}
    assert killed.get("rc") == 0, f"the kill command found no daemon to kill: {killed}"
    ok, why = same_within_determinism(run.vanilla[n], shot, cfg.tie_margin)
    assert ok, f"output after a daemon kill differs from vanilla: {why}"
    # The engine is still up and serving: the next request completes too.
    after = send(run.w, run.salt + "-after", 4096, "after-kill")
    run.extra["after_kill"] = after.summary()
    assert run.w.row(run.model_id)["status"] == "loaded"
    if cfg.log_cmd:
        new = new_log_lines(before, run_cmd(cfg.log_cmd).stdout.splitlines())
        run.extra["kill_log"] = [ln for ln in new if "LM Warden KV" in ln][-20:]
        run.extra["kill_path"] = (
            "kill before begin_load: load failed, recomputed"
            if any("load failed" in ln for ln in run.extra["kill_log"])
            else "kill after begin_load: the copy finished from the mapped RAM file"
        )
        print(f"[kv-live] kill path: {run.extra['kill_path']}")
        bad = [ln for ln in new if "Traceback" in ln or "EngineDeadError" in ln]
        assert not bad, f"engine error after the kill: {bad[:3]}"


def test_nvme_tier_serves_after_daemon_restart(run: Run) -> None:
    cfg = run.w.cfg
    if not cfg.kill_daemon_cmd:
        pytest.skip("set VW_KV_LIVE_KILL_DAEMON_CMD")
    run_cmd(cfg.kill_daemon_cmd)  # the api restarts it with a new, empty RAM pool
    time.sleep(cfg.daemon_wait_s)
    run.w.restart(run.model_id)  # a fresh engine maps the new pool
    n = next(n for n in sorted(cfg.sizes) if n >= 20000)
    shot = send(run.w, run.salt, n, "nvme")
    run.extra["nvme"] = shot.summary()
    assert int(shot.tiers.get("nvme") or 0) > 0, f"no NVMe tokens: {shot.tiers}"
    ok, why = same_within_determinism(run.vanilla[n], shot, cfg.tie_margin)
    assert ok, f"NVMe restore differs from vanilla: {why}"


def test_failed_load_recomputes(run: Run) -> None:
    """A load that fails after the lookup matched is computed by vLLM instead
    (kv_load_failure_policy=recompute, one load attempt per request)."""
    cfg = run.w.cfg
    if not (cfg.kill_daemon_cmd and cfg.break_nvme_cmd):
        pytest.skip("set VW_KV_LIVE_KILL_DAEMON_CMD and VW_KV_LIVE_BREAK_NVME_CMD")
    run_cmd(cfg.kill_daemon_cmd)  # a fresh daemon: RAM empty, the blocks only on NVMe
    time.sleep(cfg.daemon_wait_s)
    broke = run_cmd(cfg.break_nvme_cmd)  # damage the records; the daemon's index still lists them
    assert broke.returncode == 0, f"break command failed: {(broke.stdout + broke.stderr)[-300:]}"
    run.w.restart(run.model_id)
    before = run_cmd(cfg.log_cmd).stdout.splitlines() if cfg.log_cmd else []
    n = max(cfg.sizes)
    shot = send(run.w, run.salt, n, "broken")
    run.extra["broken"] = {"break_rc": broke.returncode, "shot": shot.summary()}
    ok, why = same_within_determinism(run.vanilla[n], shot, cfg.tie_margin)
    assert ok, f"output after a failed load differs from vanilla: {why}"
    assert shot.external == 0, f"a damaged block was served: {shot.tiers}"
    assert not shot.turn.get("cached_tokens"), f"cached {shot.turn.get('cached_tokens')}"
    if cfg.log_cmd:
        new = new_log_lines(before, run_cmd(cfg.log_cmd).stdout.splitlines())
        lines = [ln for ln in new if "LM Warden KV" in ln]
        run.extra["broken_log"] = lines[-5:]
        failed = [ln for ln in lines if "load failed for" in ln]
        assert len(failed) == 1, f"want exactly one failed load during this test: {lines}"


def test_old_namespace_never_loads(run: Run) -> None:
    """After the revision changes, nothing stored under the old one loads.

    Positive control first: a fresh prompt saved and restored under the original
    revision proves the store serves on this engine, so a zero later means the
    namespace changed, not that the worker was off.
    """
    cfg = run.w.cfg
    if not cfg.other_revision:
        pytest.skip("set VW_KV_LIVE_OTHER_REVISION (another commit of the same model)")
    mid = run.model_id
    salt = run.salt + "-ns"
    n = next(n for n in sorted(cfg.sizes) if n >= 20000)
    run.w.restart(mid)  # a fresh engine on the running daemon (earlier tests restarted it)
    send(run.w, salt, n, "ns-save")
    run.w.restart(mid)
    control = send(run.w, salt, n, "ns-control")
    run.extra["ns_control"] = control.summary()
    assert control.external > 0, f"control: the store served nothing: {control.tiers}"
    run.w.unload(mid)
    run.w.patch(mid, hf_revision=cfg.other_revision)
    if run.w.row(mid)["status"] != "pulled":
        run.w.pull(mid)
    run.w.load(mid)
    shot = send(run.w, salt, n, "other-rev")
    run.extra["other_rev"] = shot.summary()
    assert shot.external == 0, f"blocks from the old revision were loaded: {shot.tiers}"
    assert not shot.turn.get("cached_tokens"), f"cached {shot.turn.get('cached_tokens')}"
