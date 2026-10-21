"""The KV cache in the unified-memory admission estimate (BACKLOG-macos P6).

Weights alone under-count a Mac load: the KV cache for the context the engine
will hold lives in the same unified memory. The rules under test:

* An EXPLICIT context (the live override -- e.g. the load preflight's cap --
  else the row's ``max_model_len``) is FIRM: weights + KV must fit or the load
  is refused, because the engine is told that context.
* No explicit context: the model's own ``max_position_embeddings`` is charged
  SOFTLY -- reserved in the ledger up to the remaining headroom, never the
  reason for a refusal (llama.cpp's ``--fit`` shrinks an unflagged context to
  what is free; mlx_lm grows its cache lazily).
* Unreadable KV shape: a soft fallback of a quarter of the weights -- never 0,
  never a refusal the weights-only rule would not have made.
"""

import json
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.runtime.backends import LaunchPlan
from app.runtime.gpu_ownership import GpuConflict, GpuOwnership
from app.runtime.supervisor import Supervisor
from app.runtime.unified_admission import (
    KvReserve,
    UnifiedMemoryAdmission,
    admission_for_profile,
    kv_cache_reserve,
)

GIB = 1024**3


def _memsize_only(mib: int):
    """A capacity reading with only hw.memsize (no ceiling, no sysctls): the
    pre-P6 stand-in, so the budgets below are unchanged by the real probe."""
    from app.models.capacity import UnifiedCapacity

    return UnifiedCapacity(memsize=mib * 2**20)


# Llama-3-8B shape: 32 layers x 8 KV heads x 128 head_dim, bf16 ->
# 2 * 8 * 128 * 2 bytes * 32 layers = 128 KiB per token; 8192 tokens = 1 GiB.
LLAMA8B = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}
PER_TOKEN = 128 * 1024


def _row(**kw):
    base = {
        "id": "m",
        "served_model_name": "m",
        "max_model_len": None,
        "max_batch_size": 1,
        "extra_args": [],
    }
    base.update(kw)
    return SimpleNamespace(**base)


# --- the pure sizing ---------------------------------------------------------


def test_explicit_row_context_is_firm():
    r = kv_cache_reserve(LLAMA8B, _row(max_model_len=8192), None)
    assert r == KvReserve(nbytes=GIB, context=8192, firm=True)


def test_override_beats_the_row():
    """The load preflight's cap and reload-with-config arrive as overrides:
    that is the context the engine is actually launched with."""
    r = kv_cache_reserve(LLAMA8B, _row(max_model_len=32768), {"max_model_len": 8192})
    assert r.nbytes == GIB and r.context == 8192 and r.firm


def test_override_none_drops_to_the_model_max_softly():
    r = kv_cache_reserve(LLAMA8B, _row(max_model_len=8192), {"max_model_len": None})
    assert r.context == 131072 and not r.firm


def test_null_context_charges_the_model_max_softly():
    r = kv_cache_reserve(LLAMA8B, _row(), None)
    assert r == KvReserve(nbytes=PER_TOKEN * 131072, context=131072, firm=False)


def test_max_batch_size_multiplies_like_the_preflight():
    r = kv_cache_reserve(LLAMA8B, _row(max_model_len=8192, max_batch_size=2), None)
    assert r.nbytes == 2 * GIB


def test_declared_head_dim_and_sliding_layers_are_honoured():
    cfg = dict(
        LLAMA8B,
        head_dim=256,
        layer_types=["sliding_attention", "full_attention"] * 16,
        sliding_window=1024,
    )
    r = kv_cache_reserve(cfg, _row(max_model_len=8192), None)
    per_layer_token = 2 * 8 * 256 * 2
    assert r.nbytes == per_layer_token * (16 * 8192 + 16 * 1024)


def test_text_config_of_a_multimodal_model_is_read():
    cfg = {"text_config": LLAMA8B, "torch_dtype": "bfloat16"}
    assert kv_cache_reserve(cfg, _row(max_model_len=8192), None).nbytes == GIB


@pytest.mark.parametrize(
    ("args", "k", "v"),
    [
        (["-ctk", "q8_0", "-ctv", "q4_0"], Fraction(34, 32), Fraction(18, 32)),
        (["--cache-type-k=q4_0", "--cache-type-v", "f16"], Fraction(18, 32), 2),
        (["--cache-type-k", "f32"], 4, 2),
        # An unknown type is charged the widest ggml width, never under-counted.
        (["-ctk", "weird", "-ctv", "weird"], 4, 4),
    ],
)
def test_llamacpp_kv_cache_types_set_the_element_width(args, k, v):
    r = kv_cache_reserve(LLAMA8B, _row(max_model_len=8192, extra_args=args), None)
    elements_per_side = 8 * 128 * 32 * 8192
    assert r.nbytes == int(elements_per_side * (Fraction(k) + Fraction(v)))


@pytest.mark.parametrize(
    ("args", "context"), [(["-c", "4096"], 4096), (["--ctx-size=16384"], 16384)]
)
def test_llamacpp_ctx_size_in_extra_args_is_a_firm_context(args, context):
    """extra_args are appended last, so llama.cpp's -c/--ctx-size wins over the row."""
    r = kv_cache_reserve(LLAMA8B, _row(max_model_len=8192, extra_args=args), None)
    assert r == KvReserve(nbytes=PER_TOKEN * context, context=context, firm=True)


def test_ctx_size_zero_in_extra_args_means_the_model_max():
    r = kv_cache_reserve(LLAMA8B, _row(extra_args=["-c", "0"]), None)
    assert r.context == 131072 and not r.firm


def test_ctx_size_zero_beats_the_row_and_the_override():
    """-c 0 is llama.cpp's "model max": it overrides the row, so the row's
    8192 must not be charged as a firm context."""
    r = kv_cache_reserve(
        LLAMA8B, _row(max_model_len=8192, extra_args=["-c", "0"]), {"max_model_len": 4096}
    )
    assert r.context == 131072 and not r.firm


def test_the_last_ctx_size_flag_wins():
    r = kv_cache_reserve(
        LLAMA8B, _row(max_model_len=8192, extra_args=["-c", "1024", "--ctx-size", "4096"]), None
    )
    assert r.context == 4096 and r.firm


@pytest.mark.parametrize("value", ["abc", "", "-1", "4k", "1e4", "\u00b2", "\u0663\u0660\u0660"])
def test_garbage_ctx_size_is_soft_never_an_exception(value):
    """llama.cpp would refuse such a value at start; the sizing must not raise."""
    r = kv_cache_reserve(LLAMA8B, _row(max_model_len=8192, extra_args=["-c", value]), None)
    assert r.context == 131072 and not r.firm


@pytest.mark.parametrize(
    "cfg",
    [
        None,
        {},
        {"hidden_size": 4096, "num_attention_heads": 32},  # no layer count
        dict(LLAMA8B, max_position_embeddings=None),  # NULL ctx and no model max
    ],
)
def test_unknown_shape_or_context_is_none(cfg):
    assert kv_cache_reserve(cfg, _row(), None) is None


# --- the admission -----------------------------------------------------------


def _adm(budget, weights, kv):
    return UnifiedMemoryAdmission(
        budget_bytes=budget,
        estimate=lambda model: weights[model.id],
        kv_estimate=lambda model, overrides: kv(model, overrides),
    )


def _claim(adm, model_id, overrides=None):
    adm.claim(model_id, [0], label=model_id, model=_row(id=model_id), overrides=overrides)


def test_firm_kv_that_does_not_fit_is_refused_and_says_kv():
    adm = _adm(9 * GIB, {"a": 6 * GIB}, lambda m, o: KvReserve(4 * GIB, 32768, True))
    with pytest.raises(GpuConflict) as ei:
        _claim(adm, "a")
    msg = str(ei.value)
    assert "10.0 GiB (6.0 GiB weights + 4.0 GiB KV cache for a 32768-token context)" in msg
    assert "lower max_model_len" in msg  # the way out a firm context has
    assert adm.resident() == {}


def test_release_frees_the_kv_part_and_unblocks_the_next_load():
    """The KV share lives in the same ledger entry as the weights: a release
    (unload, crash, failed spawn -- all call it) must hand back BOTH, or the
    held headroom would refuse every later load until the warden restarts."""
    adm = _adm(
        10 * GIB,
        {"a": 6 * GIB, "b": 6 * GIB, "c": 6 * GIB},
        lambda m, o: (
            KvReserve(16 * GIB, 131072, False) if m.id == "a" else KvReserve(4 * GIB, 8192, True)
        ),
    )
    _claim(adm, "a")  # soft: holds the whole headroom
    with pytest.raises(GpuConflict):
        _claim(adm, "b")
    adm.release("a")
    assert adm.resident() == {}
    _claim(adm, "b")  # firm 6 + 4 fits the freed budget exactly
    assert adm.resident() == {"b": 10 * GIB}
    adm.release("b")
    _claim(adm, "c")
    assert adm.resident() == {"c": 10 * GIB}


def test_firm_kv_is_charged_to_the_ledger():
    adm = _adm(10 * GIB, {"a": 6 * GIB}, lambda m, o: KvReserve(4 * GIB, 32768, True))
    _claim(adm, "a")
    assert adm.resident() == {"a": 10 * GIB}


def test_soft_kv_never_refuses_and_holds_only_free_room():
    adm = _adm(10 * GIB, {"a": 6 * GIB}, lambda m, o: KvReserve(16 * GIB, 131072, False))
    _claim(adm, "a")  # weights fit; the 16 GiB model-max KV does not -- admitted
    assert adm.resident() == {"a": 10 * GIB}  # held up to the budget


def test_a_soft_hold_never_blocks_a_later_load_that_weights_admit():
    """A loaded with no context holds the free room softly; B, whose firm
    bytes fit beside A's weights, is still admitted -- A's hold shrinks."""
    adm = _adm(
        10 * GIB,
        {"a": 6 * GIB, "b": 2 * GIB, "c": 1 * GIB},
        lambda m, o: (
            KvReserve(16 * GIB, 131072, False) if m.id == "a" else KvReserve(GIB, 8192, True)
        ),
    )
    _claim(adm, "a")
    _claim(adm, "b")  # firm 2 + 1 beside a's 6 weights: fits
    assert adm.resident() == {"a": 7 * GIB, "b": 3 * GIB}
    with pytest.raises(GpuConflict):
        _claim(adm, "c")  # firm 6 + 3 + 2 > 10: a real refusal
    adm.release("b")
    assert adm.resident() == {"a": 10 * GIB}  # a's hold grows back


def test_release_hands_a_soft_hold_to_the_next_soft_holder():
    """A released soft holder's want must leave the ledger too, or it keeps
    eating the free room no resident model can use."""
    adm = _adm(
        10 * GIB, {"a": 2 * GIB, "b": 2 * GIB}, lambda m, o: KvReserve(16 * GIB, 131072, False)
    )
    _claim(adm, "a")
    _claim(adm, "b")
    assert adm.resident() == {"a": 8 * GIB, "b": 2 * GIB}
    adm.release("a")
    assert adm.resident() == {"b": 10 * GIB}


def test_refusal_names_occupants_by_their_firm_bytes_not_the_soft_hold():
    """A soft hold yields to any load, so it is not what the refusal blames."""
    adm = _adm(
        10 * GIB,
        {"a": 6 * GIB, "b": 5 * GIB},
        lambda m, o: (
            KvReserve(16 * GIB, 131072, False) if m.id == "a" else KvReserve(GIB, 8192, True)
        ),
    )
    _claim(adm, "a")
    assert adm.resident() == {"a": 10 * GIB}
    with pytest.raises(GpuConflict) as ei:
        _claim(adm, "b")
    assert "6.0 GiB is already used by 'a' (6.0 GiB)" in str(ei.value)


def test_soft_holds_are_shared_first_come_and_never_exceed_the_budget():
    adm = _adm(
        10 * GIB, {"a": 4 * GIB, "b": 4 * GIB}, lambda m, o: KvReserve(4 * GIB, 131072, False)
    )
    _claim(adm, "a")
    _claim(adm, "b")
    assert adm.resident() == {"a": 6 * GIB, "b": 4 * GIB}
    assert sum(adm.resident().values()) <= 10 * GIB


def test_unknown_kv_falls_back_to_a_quarter_of_the_weights_softly():
    adm = _adm(20 * GIB, {"a": 8 * GIB}, lambda m, o: None)
    _claim(adm, "a")
    assert adm.resident() == {"a": 10 * GIB}


def test_unknown_kv_never_refuses_what_weights_only_admits():
    adm = _adm(9 * GIB, {"a": 8 * GIB, "b": 1 * GIB}, lambda m, o: None)
    _claim(adm, "a")
    assert adm.resident() == {"a": 9 * GIB}
    _claim(adm, "b")  # weights-only admits 8 + 1; the fallback hold yields
    assert adm.resident() == {"a": 8 * GIB, "b": 1 * GIB}


def test_kv_estimate_that_raises_takes_the_fallback():
    def boom(model, overrides):
        raise ValueError("bad config")

    adm = _adm(20 * GIB, {"a": 8 * GIB}, boom)
    _claim(adm, "a")
    assert adm.resident() == {"a": 10 * GIB}


def test_overrides_reach_the_kv_estimate():
    seen = []

    def kv(model, overrides):
        seen.append(overrides)
        return KvReserve(GIB, 8192, True)

    adm = _adm(20 * GIB, {"a": GIB}, kv)
    _claim(adm, "a", overrides={"max_model_len": 8192})
    assert seen == [{"max_model_len": 8192}]


# --- wiring: the apple profile reads config.json from the snapshot -----------


def _snapshot(tmp_path, files, slug="models--org--tiny"):
    snap = tmp_path / slug / "snapshots" / "s1"
    snap.mkdir(parents=True)
    for name, content in files.items():
        (snap / name).write_bytes(content)
    return snap


@pytest.mark.asyncio
async def test_apple_profile_charges_kv_from_the_snapshot_config(tmp_path):
    _snapshot(tmp_path, {"w.gguf": b"\0" * 1000, "config.json": json.dumps(LLAMA8B).encode()})
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(64 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    cfg_size = len(json.dumps(LLAMA8B))
    model = _row(id="t", hf_repo="org/tiny", filename="w.gguf", max_model_len=8192)
    adm.claim("t", [0], label="t", model=model, overrides=None)
    assert adm.resident() == {"t": 1000 + cfg_size + GIB}


@pytest.mark.asyncio
async def test_apple_profile_reads_config_from_hf_config_repo(tmp_path):
    _snapshot(tmp_path, {"w.gguf": b"\0" * 1000})
    _snapshot(tmp_path, {"config.json": json.dumps(LLAMA8B).encode()}, slug="models--org--base")
    with patch(
        "app.system.gpu._apple_capacity", new=AsyncMock(return_value=_memsize_only(64 * 1024))
    ):
        adm = await admission_for_profile("apple", hf_cache_dir=tmp_path)
    model = _row(
        id="t",
        hf_repo="org/tiny",
        hf_config_repo="org/base",
        filename="w.gguf",
        max_model_len=8192,
    )
    adm.claim("t", [0], label="t", model=model, overrides={"max_model_len": 16384})
    assert adm.resident() == {"t": 1000 + 2 * GIB}


# --- the supervisor hands its overrides to the admission ---------------------


def test_gpu_ownership_accepts_and_ignores_overrides():
    g = GpuOwnership()
    g.claim("m1", [0], label="x", model=None, overrides={"max_model_len": 1})
    with pytest.raises(GpuConflict):
        g.claim("m2", [0], overrides=None)


@pytest.mark.asyncio
async def test_supervisor_passes_load_overrides_to_the_admission(tmp_path):
    seen = []

    def kv(model, overrides):
        seen.append(overrides)
        return KvReserve(GIB, 8192, True)

    adm = UnifiedMemoryAdmission(budget_bytes=20 * GIB, estimate=lambda m: GIB, kv_estimate=kv)
    s = SimpleNamespace(
        load_max_s=7200.0,
        data_dir=str(tmp_path),
        hf_token_path=str(tmp_path / "tok"),
        hf_cache_dir=str(tmp_path / "hf"),
    )
    sup = Supervisor(s, admission=adm)
    model = MagicMock()
    model.id = "a"
    model.served_model_name = "a"
    model.gpu_indices = [0]
    model.tensor_parallel_size = 1
    model.engine_image = None
    model.engine_channel = None
    model.engine_vllm_version = None
    model.backend = None
    model.mmproj_filename = None
    model.filename = None
    proc = MagicMock()
    proc.pid = 1
    proc.returncode = None
    plan = LaunchPlan(argv=["vllm", "serve"], env={}, image=None, port=10001, gpu_indices=[0])
    with (
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        patch("app.runtime.backends.vllm.VllmBackend.plan", return_value=plan),
    ):
        await sup.load(model, port=10001, overrides={"max_model_len": 8192})
    assert seen == [{"max_model_len": 8192}]
    assert adm.resident() == {"a": 2 * GIB}
    for w in list(sup._watchers.values()):
        w.cancel()
