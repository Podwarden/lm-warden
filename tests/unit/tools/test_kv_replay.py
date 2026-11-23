"""tools/kv_replay.py: trace build, the in-process replay and the flood (gates G9, G10).

Every trace here is synthetic (random token ids) and every recording is a hand-written
fixture: no number in this file is a measurement of real traffic.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from lmw_kv import ram as ram_mod
from lmw_kv.fair import Policy
from tests.unit.lmw_kv.test_store import ROOT, SCOPE
from tests.unit.lmw_kv.test_store_lru_exact import (
    GOLDEN,
    MAX_CHAIN,
    MAX_HELD,
    OPS,
    golden_store,
)

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "kv_replay.py"
B = 16


@pytest.fixture(scope="module")
def kr() -> ModuleType:
    spec = importlib.util.spec_from_file_location("kv_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kv_replay"] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


# ---- synthetic traces ----------------------------------------------------------------


def _session(
    key: str, *, start: int, grow: int, turns: int, t0: float, gap: float, isolated: bool = False
) -> list[dict[str, Any]]:
    """One key's session: every turn repeats the last prompt and appends ``grow`` tokens."""
    rng = random.Random(key)
    toks = [rng.randrange(1000, 30000) for _ in range(start)]
    out = []
    for j in range(turns):
        out.append(
            {
                "t": t0 + j * gap,
                "key": key,
                "isolated": isolated,
                "priority": 5,
                "tokens": list(toks),
            }
        )
        toks += [rng.randrange(1000, 30000) for _ in range(grow)]
    return out


def _three_keys() -> list[dict[str, Any]]:
    trace = (
        _session("a", start=12 * B, grow=2 * B + 5, turns=10, t0=0.0, gap=60.0)
        + _session("b", start=6 * B, grow=B, turns=8, t0=7.0, gap=75.0)
        + _session("c", start=5 * B + 3, grow=B, turns=8, t0=13.0, gap=70.0, isolated=True)
    )
    return sorted(trace, key=lambda e: (e["t"], e["key"]))


#: The RAM pool a replay needs is a few MiB; ask for more so the check has margin.
_POOL_ROOM = 64 << 20


def fast_workdir(shm: Path = Path("/dev/shm")) -> Path | None:
    """A memory file system when it has room for the RAM pool, else None (the system's
    temporary directory). The store syncs every write, so tmpfs is faster; but a default
    docker container has a 64 MiB /dev/shm, under the store's 2 GiB reserve, and there
    the RAM tier would be off and every RAM count 0."""
    if not shm.is_dir():
        return None
    return shm if ram_mod.fit_shm(shm / "kv-replay.probe", _POOL_ROOM) >= _POOL_ROOM else None


FAST = fast_workdir()
SIM: dict[str, Any] = {
    "ram_blocks": 24, "nvme_segments": 32, "segment_blocks": 1, "block_size": B, "workdir": FAST,
}  # fmt: skip


def test_fast_workdir_skips_a_small_shm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A 64 MiB /dev/shm (docker's default) is under the store's reserve: use the disk."""
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(ram_mod, "fit_shm", lambda path, wanted: 0)
    assert fast_workdir(shm) is None
    monkeypatch.setattr(ram_mod, "fit_shm", lambda path, wanted: wanted)
    assert fast_workdir(shm) == shm
    assert fast_workdir(tmp_path / "missing") is None


def test_sim_is_deterministic(kr: ModuleType) -> None:
    trace = _three_keys()
    one = kr.sim(trace, policy="fair", **SIM)
    two = kr.sim(trace, policy="fair", **SIM)
    assert one == two
    assert set(one) >= {"ram_tokens", "nvme_tokens", "recomputed_tokens", "by_key"}
    assert set(one["by_key"]) == {"a", "b", "c"}
    total = sum(len(e["tokens"]) for e in trace)
    assert one["ram_tokens"] + one["nvme_tokens"] + one["recomputed_tokens"] == total
    assert one["ram_tokens"] + one["nvme_tokens"] > 0
    assert kr.sim(trace, policy="lru", **SIM) == kr.sim(trace, policy="lru", **SIM)
    for _ in range(3):  # the writer thread's pace never shows
        assert kr.sim(trace, policy="fair", **SIM) == one


def test_sim_counts_whole_blocks_and_keeps_isolation(kr: ModuleType) -> None:
    """Two keys send the same prompt: a sharing pair reuses it, an isolated one never."""
    toks = list(range(100, 100 + 4 * B + 3))
    base = {"priority": 5, "tokens": toks}
    shared = [
        {"t": 0.0, "key": "x", "isolated": False, **base},
        {"t": 1.0, "key": "y", "isolated": False, **base},
    ]
    got = kr.sim(shared, policy="lru", **SIM)
    assert got["by_key"]["y"]["ram_tokens"] == 4 * B  # every whole block, from RAM
    assert got["by_key"]["y"]["recomputed_tokens"] == 3
    apart = [dict(shared[0]), {**shared[1], "isolated": True}]
    got = kr.sim(apart, policy="lru", **SIM)
    assert got["by_key"]["y"]["ram_tokens"] == 0
    assert got["by_key"]["y"]["recomputed_tokens"] == len(toks)


def test_sim_last_token_is_always_computed(kr: ModuleType) -> None:
    toks = list(range(10, 10 + 2 * B))  # exactly two blocks: only one is loadable
    trace = [
        {"t": 0.0, "key": "x", "isolated": False, "priority": 5, "tokens": toks},
        {"t": 1.0, "key": "x", "isolated": False, "priority": 5, "tokens": toks},
    ]
    got = kr.sim(trace, policy="lru", **SIM)["by_key"]["x"]
    assert got["ram_tokens"] == B
    assert got["repeat_turns"] == got["repeat_served"] == 1
    assert got["resumed_turns"] == 0  # 1 s later: the GPU cache's, not a resume


def test_only_a_pause_over_gap_s_is_a_resume(kr: ModuleType) -> None:
    toks = list(range(10, 10 + 3 * B))
    trace = [
        {"t": t, "key": "x", "isolated": False, "priority": 5, "tokens": toks}
        for t in (0.0, 30.0, 400.0, 401.0)
    ]
    got = kr.sim(trace, policy="lru", **SIM)["by_key"]["x"]
    assert (got["turns"], got["repeat_turns"], got["resumed_turns"]) == (4, 3, 1)
    assert got["served_turns"] == 1
    assert (
        kr.sim(trace, policy="lru", **{**SIM, "gap_s": 10.0})["by_key"]["x"]["resumed_turns"] == 2
    )


# ---- the golden workload through replay_ops -------------------------------------------


def _golden_ops(nss: list[bytes], results: list[Any]) -> Iterator[tuple[Any, ...]]:
    """The step 1 golden workload (test_store_lru_exact), as operations.

    Lazy: a save's outcome decides the next random draws, as in the original, so each
    operation is yielded only after the one before it ran (its result in ``results``).
    """
    rng = random.Random(7)
    chains: list[list[list[tuple[bytes, bytes]]]] = [[] for _ in nss]
    held: list[str] = []
    counter = 0
    holds = 0
    for _ in range(OPS):
        i = rng.randrange(len(nss))
        ns_id = nss[i]
        op = rng.random()
        if op < 0.55 or not chains[i]:
            if not chains[i] or rng.random() < 0.3:
                chain: list[tuple[bytes, bytes]] = []
                chains[i].append(chain)
            else:
                chain = rng.choice(chains[i])
                if len(chain) >= MAX_CHAIN:
                    chain = []
                    chains[i].append(chain)
            parent = chain[-1][0] if chain else ROOT
            counter += 1
            key = hashlib.sha256(f"{i}:{counter}".encode()).digest()
            yield ("save", ns_id, key, parent, SCOPE, {})
            if results[-1] is None:
                continue
            if rng.random() < 0.85:
                yield ("commit", ns_id, key)
                chain.append((key, parent))
                yield ("flush",)
            else:
                yield ("abort", ns_id, key)
        elif op < 0.85:
            chain = rng.choice(chains[i])
            if rng.random() < 0.3:
                holds += 1
                name = f"h{holds}"
                yield ("load", ns_id, [k for k, _ in chain], {"hold": name})
                held.append(name)
                if len(held) > MAX_HELD:
                    yield ("end", held.pop(0))
            else:
                yield ("load", ns_id, [k for k, _ in chain], {})
        elif op < 0.95:
            if held:
                yield ("end", held.pop(rng.randrange(len(held))))
        else:
            chain = rng.choice(chains[i])
            yield ("lookup", ns_id, [k for k, _ in chain])
    for name in held:
        yield ("end", name)
    yield ("flush",)


def test_replay_ops_lru_evicts_the_step1_golden_sequence(
    kr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s, nss, victims = golden_store(tmp_path, monkeypatch)
    try:
        s.set_policy(Policy())
        results: list[Any] = []
        totals = kr.replay_ops(s, _golden_ops(nss, results), results=results)
    finally:
        s.close()
    assert totals["ram_tokens"] + totals["nvme_tokens"] > 0
    golden = json.loads(GOLDEN.read_text())
    assert len(victims) == len(golden)
    for n, (got, want) in enumerate(zip(victims, golden, strict=True)):
        assert got == want, f"victim {n} differs"


# ---- the flood (G10) -------------------------------------------------------------------


def test_flood_adds_long_distinct_prompts(kr: ModuleType) -> None:
    trace = _three_keys()
    flooded = kr.flood(trace, "z", 10)
    mine = [e for e in flooded if e["key"] == "z"]
    others = [e for e in flooded if e["key"] != "z"]
    assert others == trace  # the trace itself is unchanged, in order
    assert [e["t"] for e in flooded] == sorted(e["t"] for e in flooded)
    share = sum(len(e["tokens"]) for e in trace) / 4  # four keys with the flooder
    sent = sum(len(e["tokens"]) for e in mine)
    assert 10 * share <= sent < 10 * share + max(len(e["tokens"]) for e in trace)
    longest = max(len(e["tokens"]) for e in trace)
    assert all(len(e["tokens"]) == longest for e in mine)
    firsts = {tuple(e["tokens"][:B]) for e in mine}
    assert len(firsts) == len(mine)  # no two flood prompts share a block
    assert kr.flood(trace, "z", 10) == flooded  # deterministic


def _g10_trace(victim_blocks: int) -> list[dict[str, Any]]:
    """Key a resumes a session every 400 s (a resume: over ``GAP_S``); b and c are busy
    small sessions, a turn every 50 s."""
    return sorted(
        _session("a", start=victim_blocks * B, grow=B, turns=6, t0=0.0, gap=400.0)
        + _session("b", start=8 * B, grow=4, turns=40, t0=3.0, gap=50.0)
        + _session("c", start=7 * B + 5, grow=4, turns=40, t0=6.0, gap=50.0, isolated=True),
        key=lambda e: (e["t"], e["key"]),
    )


def _g10(kr: ModuleType, trace: list[dict[str, Any]], sim: dict[str, Any]) -> list[Any]:
    tier = max(sim["ram_blocks"], sim["nvme_segments"] * sim["segment_blocks"])
    share = kr.tier_share_tokens(trace, "z", tier, B)
    flooded = kr.flood(trace, "z", 10, share_tokens=share, victim="a")
    return [kr.sim(flooded, policy=p, **sim)["by_key"]["a"] for p in ("fair", "lru")]


def test_tier_share_and_flood_between_the_victims_turns(kr: ModuleType) -> None:
    trace = _g10_trace(12)
    share = kr.tier_share_tokens(trace, "z", 96, B)
    assert share == 96 * B / 4  # four keys at the default priority
    flooded = kr.flood(trace, "z", 10, share_tokens=share, victim="a")
    mine = [e for e in flooded if e["key"] == "z"]
    longest = max(len(e["tokens"]) for e in trace)
    per_gap = -(-10 * share // longest)
    assert len(mine) == 5 * per_gap  # every one of a's five resume pauses
    a_turns = [e["t"] for e in trace if e["key"] == "a"]
    for lo, hi in zip(a_turns, a_turns[1:], strict=False):
        inside = [e for e in mine if lo < e["t"] < hi]
        assert len(inside) == per_gap
    assert [e for e in flooded if e["key"] != "z"] == trace
    # A pause under gap_s gets no flood.
    assert kr.flood(trace, "z", 10, share_tokens=share, victim="a", gap_s=500.0) == trace


def test_flood_fair_keeps_the_resumed_session_and_lru_does_not(kr: ModuleType) -> None:
    """The victim's session (12-17 blocks) fits its RAM share (18 of 72)."""
    fair, lru = _g10(kr, _g10_trace(12), {**SIM, "ram_blocks": 72, "nvme_segments": 96})
    assert fair["resumed_turns"] == lru["resumed_turns"] == 5
    assert fair["served_turns"] / fair["resumed_turns"] >= 0.9
    assert lru["served_turns"] < fair["served_turns"]


def test_flood_fair_resumes_a_session_above_its_ram_share(kr: ModuleType) -> None:
    """The victim's session (12-17 blocks) is above its RAM share (6 of 24) but within
    NVMe's (24 of 96). Before the load borrow (ruling R14 revised) every resume stopped
    at about its RAM share: 0 of 5 served."""
    fair, lru = _g10(kr, _g10_trace(12), {**SIM, "ram_blocks": 24, "nvme_segments": 96})
    assert fair["resumed_turns"] == lru["resumed_turns"] == 5
    assert fair["served_turns"] / fair["resumed_turns"] >= 0.9
    assert lru["served_turns"] < fair["served_turns"]


# ---- build (recordings -> trace) -------------------------------------------------------


class _TinyTokenizer:
    """Whitespace words and the two ChatML markers, each its own id."""

    def __init__(self) -> None:
        self.vocab = {"<|im_start|>": 1, "<|im_end|>": 2, "system": 3, "user": 4, "assistant": 5}

    def encode(self, text: str) -> list[int]:
        for m in ("<|im_start|>", "<|im_end|>"):
            text = text.replace(m, f" {m} ")
        out = []
        for w in text.split():
            out.append(self.vocab.setdefault(w, len(self.vocab) + 1))
        return out


def _recording(
    path: Path, ts: float, body: dict[str, Any], *, session: str, status: int = 200
) -> None:
    text = json.dumps(body)
    path.write_text(
        json.dumps(
            {
                "ts": ts,
                "method": "POST",
                "path": "/v1/chat/completions",
                "request_headers": [["Authorization", "Bearer vwk_not-a-real-secret"]]
                + ([["X-Session-Id", session]] if session else []),
                "request_body": {"b64": base64.b64encode(text.encode()).decode(), "text": text},
                "status": status,
                "response_headers": [],
                "response_body": {"b64": "", "text": "the reply text"},
            }
        )
    )


TWO_MESSAGES = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "hello world"}]},
    ],
}


def test_build_two_message_recording(kr: ModuleType, tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    rec.mkdir()
    _recording(rec / "1.json", 1000.5, TWO_MESSAGES, session="s-one")
    second = {
        "model": "m",
        "messages": [*TWO_MESSAGES["messages"], {"role": "user", "content": "again"}],
    }
    _recording(rec / "2.json", 1003.0, second, session="s-two")
    _recording(rec / "3.json", 1004.0, second, session="s-two", status=502)
    tok = _TinyTokenizer()
    trace = kr.build_trace(kr.read_recordings(rec), tok.encode)
    # <|im_start|> system be brief <|im_end|> <|im_start|> user hello world <|im_end|>
    # <|im_start|> assistant
    want = [1, 3, 6, 7, 2, 1, 4, 8, 9, 2, 1, 5]
    assert trace[0] == {"t": 0.0, "key": "k01", "isolated": False, "priority": 5, "tokens": want}
    assert trace[1]["t"] == 2.5 and trace[1]["key"] == "k02"
    assert trace[1]["tokens"][: len(want) - 2] == want[:-2]  # the shared prefix
    assert len(trace) == 2  # the failed request is left out


def test_build_skips_what_is_not_a_recording(kr: ModuleType, tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    rec.mkdir()
    _recording(rec / "ok.json", 1.0, TWO_MESSAGES, session="s")
    good = json.loads((rec / "ok.json").read_text())
    (rec / "no-ts.json").write_text(json.dumps({k: v for k, v in good.items() if k != "ts"}))
    (rec / "bool-ts.json").write_text(json.dumps({**good, "ts": True}))
    (rec / "bad-headers.json").write_text(json.dumps({**good, "request_headers": [["a"]]}))
    bad_b64 = {**good, "request_body": {"b64": "%%%"}}
    (rec / "bad-b64.json").write_text(json.dumps(bad_b64))
    (rec / "list.json").write_text("[1, 2]")
    (rec / "broken.json").write_text("{")
    as_object = {**good, "ts": 2.0, "request_headers": {"X-Session-Id": "s"}}
    (rec / "object-headers.json").write_text(json.dumps(as_object))
    got = kr.read_recordings(rec)
    assert [r["ts"] for r in got] == [1.0, 2.0]
    assert got[1]["headers"] == {"x-session-id": "s"}


def test_build_gives_each_unnamed_request_its_own_key(kr: ModuleType, tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    rec.mkdir()
    for i, session in enumerate(["", "s", "", "s"]):
        _recording(rec / f"{i}.json", float(i), TWO_MESSAGES, session=session)
    trace = kr.build_trace(kr.read_recordings(rec), _TinyTokenizer().encode)
    assert [e["key"] for e in trace] == ["k01", "k02", "k03", "k02"]


def test_build_reads_every_api_shape(kr: ModuleType) -> None:
    anthropic = {
        "system": [{"type": "text", "text": "sys"}],
        "tools": [{"name": "read"}],
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "tool_use", "id": "1", "name": "read", "input": {"p": "x"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "content": "data"}]},
        ],
    }
    text = kr.render(anthropic)
    assert text.startswith("<|im_start|>system\nsys")
    assert "hidden" not in text and '"read"' in text and "data" in text
    responses = {
        "instructions": "ins",
        "input": [
            {"type": "message", "role": "developer", "content": "dev"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
            {"type": "reasoning", "summary": [{"text": "hidden"}]},
            {"type": "function_call", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "output": "out"},
        ],
    }
    text = kr.render(responses)
    assert text.startswith("<|im_start|>system\nins")
    assert "<|im_start|>system\ndev" in text and "hidden" not in text and "out" in text
    assert kr.render({"input": "plain"}).startswith("<|im_start|>user\nplain")


def test_build_cli_with_a_tokenizer_file(kr: ModuleType, tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    words = ["[UNK]", "system", "user", "assistant", "be", "brief", "hello", "world"]
    tk = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tk.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tk.add_special_tokens(["<|im_start|>", "<|im_end|>"])  # ids 8 and 9
    tk.save(str(tmp_path / "tokenizer.json"))
    rec = tmp_path / "rec"
    rec.mkdir()
    _recording(rec / "1.json", 5.0, TWO_MESSAGES, session="s")
    out = tmp_path / "trace.jsonl"
    rc = kr.main(
        ["build", "--recordings", str(rec), "--tokenizer", str(tmp_path / "tokenizer.json"),
         "--out", str(out)]
    )  # fmt: skip
    assert rc == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["tokens"] == [8, 1, 4, 5, 9, 8, 2, 6, 7, 9, 8, 3]
    raw = out.read_text()
    assert "hello" not in raw and "not-a-real-secret" not in raw
    assert entry["key"] == "k01"


def test_sim_cli_prints_json(
    kr: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in _three_keys()))
    rc = kr.main(
        ["sim", "--trace", str(path), "--policy", "lru", "--ram-blocks", "24",
         "--nvme-segments", "32", "--segment-blocks", "1", "--block-size", str(B),
         *(["--workdir", str(FAST)] if FAST else [])]
    )  # fmt: skip
    assert rc == 0
    got = json.loads(capsys.readouterr().out)
    assert got == kr.sim(_three_keys(), policy="lru", **SIM)  # workdir does not change it


# ---- live ------------------------------------------------------------------------------


def test_live_scores_each_turn_from_its_history_row(kr: ModuleType) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []
    tiers = {  # request id -> its history row's tiers; r4 is never recorded
        "r1": {"gpu": 0, "ram": 0, "nvme": 0},
        "r2": {"gpu": 4, "ram": 0, "nvme": 0},
        "r3": {"gpu": 0, "ram": 2, "nvme": 2},
    }
    waits: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        auth = req.headers.get("authorization", "")
        if req.url.path == "/v1/completions":
            sent.append((auth, json.loads(req.content)))
            usage = {"prompt_tokens_details": {"cached_tokens": 4}}
            rid = {"X-LMW-Request-Id": f"r{len(sent)}"}
            return httpx.Response(200, json={"usage": usage}, headers=rid)
        assert auth == "Bearer admin-secret"
        if req.url.path == "/api/kv/usage":
            return httpx.Response(200, json={"available": True, "models": []})
        if req.url.path.startswith("/api/requests/"):
            waits.append(req.url.params["wait_ms"])
            got = tiers.get(req.url.path.rsplit("/", 1)[1])
            if got is None:
                return httpx.Response(404, json={"detail": "not recorded yet"})
            return httpx.Response(200, json={"tiers": {**got, "remote": None, "load_s": 0.1}})
        w = {"gpu": 0, "ram": 4 * len(sent), "nvme": 0, "recomputed": 5, "hit_rate": None}
        return httpx.Response(200, json={"windows": {"1h": w, "24h": w, "7d": w}})

    toks = [1, 2, 3, 4, 5]  # two whole blocks of 2 are loadable
    trace = [
        {"t": 0.0, "key": "a", "isolated": False, "priority": 5, "tokens": toks},
        {"t": 1.0, "key": "a", "isolated": False, "priority": 5, "tokens": toks},
        {"t": 1000.0, "key": "a", "isolated": False, "priority": 5, "tokens": toks},
        {"t": 1001.0, "key": "b", "isolated": False, "priority": 5, "tokens": [9, 9, 9]},
    ]
    keys = {"a": {"token": "key-x", "token_id": "x"}, "b": {"token": "key-y", "token_id": "y"}}
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    got = kr.live(trace, keys, model="m", admin_token="admin-secret", client=client, block_size=2)
    assert sent[0] == ("Bearer key-x", {"model": "m", "prompt": toks, "max_tokens": 1})
    assert sent[3][0] == "Bearer key-y"
    assert got["requests"] == 4 and got["errors"] == 0 and got["unscored"] == 1
    assert waits == ["5000"] * 4
    a = got["by_key"]["a"]
    assert (a["turns"], a["repeat_turns"], a["repeat_served"]) == (3, 2, 2)
    assert (a["resumed_turns"], a["served_turns"]) == (1, 1)  # only t=1000 is a resume
    assert (a["ram_tokens"], a["nvme_tokens"], a["gpu_tokens"]) == (2, 2, 4)
    assert a["recomputed_tokens"] == 5 + 1 + 1
    assert a["cached_tokens"] == 12 and a["window_24h"]["ram"] == 16
    assert got["usage"] == {"available": True, "models": []}
    assert "admin-secret" not in json.dumps(got) and "key-x" not in json.dumps(got)


def test_live_cli_floods_only_with_a_share(
    kr: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(_three_keys()[0]) + "\n")
    monkeypatch.setenv("LMW_TEST_ADMIN", "x")
    args = ["live", "--trace", str(path), "--url", "http://127.0.0.1:9", "--keys",
            str(tmp_path / "k.json"), "--model", "m", "--block-size", "16",
            "--admin-token-env", "LMW_TEST_ADMIN", "--flood", "z"]  # fmt: skip
    assert kr.main(args) == 2  # no --flood-share-tokens: refused before any request


def _live_one_key(kr: ModuleType, tiers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``live`` over three turns of key a (t=0, then two resumes); ``tiers`` by request id."""
    sent: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/completions":
            sent.append(len(sent) + 1)
            rid = {"X-LMW-Request-Id": f"r{sent[-1]}"}
            return httpx.Response(200, json={"usage": {}}, headers=rid)
        if req.url.path == "/api/kv/usage":
            return httpx.Response(200, json={"available": True})
        if req.url.path.startswith("/api/requests/"):
            return httpx.Response(200, json={"tiers": tiers[req.url.path.rsplit("/", 1)[1]]})
        w = {"gpu": 0, "ram": 0, "nvme": 0, "recomputed": 0}
        return httpx.Response(200, json={"windows": {"24h": w}})

    toks = [1, 2, 3, 4, 5]
    trace = [
        {"t": t, "key": "a", "isolated": False, "priority": 5, "tokens": toks}
        for t in (0.0, 1000.0, 2000.0)
    ]
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    keys = {"a": {"token": "k", "token_id": "x"}}
    return dict(kr.live(trace, keys, model="m", admin_token="t", client=client, block_size=2))


def test_live_counts_an_all_null_tiers_row_as_unscored(kr: ModuleType) -> None:
    """N2: a row whose gpu, ram and nvme are all null says nothing about the turn."""
    nothing = {"gpu": None, "ram": None, "nvme": None, "remote": None, "load_s": None}
    got = _live_one_key(
        kr, {"r1": {"gpu": 0, "ram": 0, "nvme": 0}, "r2": nothing, "r3": {"gpu": 0, "ram": 4}}
    )
    assert got["unscored"] == 1
    a = got["by_key"]["a"]
    assert (a["turns"], a["resumed_turns"], a["served_turns"]) == (2, 1, 1)


def test_live_reports_store_only_served_turns_beside_the_gpu_inclusive_count(
    kr: ModuleType,
) -> None:
    """N3: G10 counts turns served from RAM or NVMe; a GPU hit is served, not by the store."""
    got = _live_one_key(
        kr,
        {
            "r1": {"gpu": 0, "ram": 0, "nvme": 0},
            "r2": {"gpu": 4, "ram": 0, "nvme": 0},
            "r3": {"gpu": 2, "ram": 0, "nvme": 2},  # 2 from the store: not the whole 4
        },
    )
    a = got["by_key"]["a"]
    assert (a["resumed_turns"], a["served_turns"], a["served_store_turns"]) == (2, 2, 0)
    got = _live_one_key(
        kr,
        {
            "r1": {"gpu": 0, "ram": 0, "nvme": 0},
            "r2": {"gpu": 0, "ram": 4, "nvme": 0},
            "r3": {"gpu": 0, "ram": 2, "nvme": 2},
        },
    )
    a = got["by_key"]["a"]
    assert (a["resumed_turns"], a["served_turns"], a["served_store_turns"]) == (2, 2, 2)


def test_sim_store_only_served_turns_equal_served_turns(kr: ModuleType) -> None:
    """``sim`` has no GPU cache model: every reused token came from the store."""
    toks = list(range(10, 10 + 3 * B))
    trace = [
        {"t": t, "key": "x", "isolated": False, "priority": 5, "tokens": toks} for t in (0.0, 400.0)
    ]
    got = kr.sim(trace, policy="lru", **SIM)["by_key"]["x"]
    assert got["served_store_turns"] == got["served_turns"] == 1
