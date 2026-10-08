"""Prove the LAUNCH SEQUENCE, not just the two builders.

Supervisor.load reads the HF token off disk, calls build_subprocess_env and
build_vllm_args, resolves the engine-image pin, and assembles an EngineSpec
that it hands to the driver. This test captures that whole assembly with a fake
driver, so a refactor that gets both builders right but wires them together
differently still fails.
"""

import pytest

from app.runtime.supervisor import Supervisor
from tests.unit.runtime.backends.corpus import CASES, load_golden

GOLDEN = load_golden()

# Three cases chosen to span the assembly: the baseline, a multi-GPU row (which
# exercises gpu_indices threading into both the env and the spec), and a GGUF
# row (whose model_arg on the spec is the BARE hf_repo even though argv carries
# the :QUANT tag — an asymmetry worth pinning).
SPEC_CASES = ["minimal", "tp2", "gguf_quant_tag"]


class _CapturingDriver:
    supports_engine_image = False

    def __init__(self):
        self.spec = None

    async def spawn(self, spec):
        self.spec = spec
        return _DeadHandle()

    async def terminate(self, handle, *, grace_s):
        return None

    def engine_host(self, model_id):
        return "127.0.0.1"


class _DeadHandle:
    pid = 4242
    returncode = None

    async def wait(self):
        import asyncio

        await asyncio.sleep(3600)
        return 0


class _Settings:
    engine_driver = "local"

    def __init__(self, tmp_path):
        self.data_dir = str(tmp_path)
        self.hf_cache_dir = "/cache"
        self.hf_token_path = str(tmp_path / "hf-token")


@pytest.mark.asyncio
@pytest.mark.parametrize("case_name", SPEC_CASES)
async def test_engine_spec_is_unchanged(tmp_path, case_name):
    case = next(c for c in CASES if c.name == case_name)
    (tmp_path / "hf-token").write_text("hf_tok\n")

    driver = _CapturingDriver()
    sup = Supervisor(_Settings(tmp_path), driver=driver)
    await sup.load(case.model, port=case.port, overrides=case.overrides)

    spec = driver.spec
    assert spec.model_id == case.model.id
    assert spec.model_arg == case.model.hf_repo
    assert list(spec.argv) == ["vllm", "serve", *GOLDEN[case_name]["argv"]]
    assert dict(spec.env) == GOLDEN[case_name]["env"]
    assert spec.port == case.port
    assert spec.image is None
    assert list(spec.gpu_indices) == list(case.model.gpu_indices)

    await sup.unload(case.model.id, force=True)


@pytest.mark.asyncio
async def test_unknown_backend_on_a_row_refuses_the_load(tmp_path):
    """A row naming a backend this build does not have must fail loudly at
    load, before any GPU is claimed -- not silently launch vLLM."""
    from app.runtime.backends.registry import UnknownBackendError
    from tests.unit.runtime.backends.corpus import StubModel

    (tmp_path / "hf-token").write_text("hf_tok\n")
    model = StubModel()
    # NOT "llamacpp" any more -- sub-project C registers it, so it is a KNOWN
    # backend and would load. The test needs a name this build genuinely lacks.
    model.backend = "sglang"

    sup = Supervisor(_Settings(tmp_path), driver=_CapturingDriver())
    with pytest.raises(UnknownBackendError):
        await sup.load(model, port=10000)
    # the GPU claim was released on the failure path
    assert sup.gpus.owner_of(0) is None


@pytest.mark.asyncio
async def test_spec_backend_comes_from_the_row(tmp_path):
    from tests.unit.runtime.backends.corpus import StubModel

    (tmp_path / "hf-token").write_text("hf_tok\n")
    driver = _CapturingDriver()
    sup = Supervisor(_Settings(tmp_path), driver=driver)
    await sup.load(StubModel(), port=10000)
    assert driver.spec.backend == "vllm"
    await sup.unload("m1", force=True)


# ---------------------------------------------------------------------------
# Sub-project C: the llama.cpp half of the same assembly
# ---------------------------------------------------------------------------


def _make_gguf_snapshot(cache_root, repo="org/model", sha="abc123", files=("model-IQ3_XXS.gguf",)):
    snap = cache_root / f"models--{repo.replace('/', '--')}" / "snapshots" / sha
    snap.mkdir(parents=True)
    for name in files:
        (snap / name).write_bytes(b"gguf")
    return snap


class _CacheSettings(_Settings):
    """_Settings with a REAL hf_cache_dir, so resolve_model_paths finds files."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.hf_cache_dir = str(tmp_path / "cache")


@pytest.mark.asyncio
async def test_llamacpp_row_produces_a_llama_server_spec(tmp_path):
    """The whole point of the sub-project, at the seam. The supervisor resolves
    the paths once, hands them to plan(), and the driver receives argv whose
    argv[0] is llama-server."""
    from tests.unit.runtime.backends.corpus import StubModel

    (tmp_path / "hf-token").write_text("hf_tok\n")
    snap = _make_gguf_snapshot(tmp_path / "cache")

    model = StubModel()
    model.backend = "llamacpp"
    model.filename = "model-IQ3_XXS.gguf"

    driver = _CapturingDriver()
    sup = Supervisor(_CacheSettings(tmp_path), driver=driver)
    await sup.load(model, port=10000)

    assert driver.spec.backend == "llamacpp"
    assert driver.spec.argv[0] == "llama-server"
    assert "--metrics" in driver.spec.argv
    assert str(snap / "model-IQ3_XXS.gguf") in driver.spec.argv
    # model_arg stays the repo id on the spec even though argv carries a path --
    # the same asymmetry the gguf_quant_tag vLLM case pins.
    assert driver.spec.model_arg == model.hf_repo

    await sup.unload(model.id, force=True)


@pytest.mark.asyncio
async def test_missing_gguf_fails_before_the_gpu_is_claimed(tmp_path):
    """A mistyped filename must fail inside the try/except that releases the
    claim (supervisor.py), or it permanently reserves a GPU.

    The raise comes from build_llamacpp_args, not from the resolver: the
    resolver runs for EVERY backend and ``filename`` is not llama.cpp-only (a
    vLLM GGUF row sets it too and vLLM resolves the file itself), so an
    unresolvable weights file is reported as None and the refusal is left to
    the backend that cannot proceed without one. Either way it is before any
    process is spawned, which is the property that matters here.
    """
    from tests.unit.runtime.backends.corpus import StubModel

    (tmp_path / "hf-token").write_text("hf_tok\n")
    _make_gguf_snapshot(tmp_path / "cache", files=("something-else.gguf",))

    model = StubModel()
    model.backend = "llamacpp"
    model.filename = "not-pulled.gguf"

    driver = _CapturingDriver()
    sup = Supervisor(_CacheSettings(tmp_path), driver=driver)
    with pytest.raises(ValueError, match="model_path"):
        await sup.load(model, port=10000)
    assert sup.gpus.owner_of(0) is None
    assert driver.spec is None  # nothing was spawned


@pytest.mark.asyncio
async def test_missing_mmproj_fails_before_the_gpu_is_claimed(tmp_path):
    """The projector IS llama.cpp-only, so the RESOLVER refuses it -- earlier
    than plan() and with a message naming the directory it searched. Same
    property: inside the try/except, before any spawn."""
    from app.runtime.backends.paths import ModelFileNotFound
    from tests.unit.runtime.backends.corpus import StubModel

    (tmp_path / "hf-token").write_text("hf_tok\n")
    _make_gguf_snapshot(tmp_path / "cache")

    model = StubModel()
    model.backend = "llamacpp"
    model.filename = "model-IQ3_XXS.gguf"
    model.mmproj_filename = "mmproj-never-pulled.gguf"

    driver = _CapturingDriver()
    sup = Supervisor(_CacheSettings(tmp_path), driver=driver)
    with pytest.raises(ModelFileNotFound, match="mmproj"):
        await sup.load(model, port=10000)
    assert sup.gpus.owner_of(0) is None
    assert driver.spec is None


@pytest.mark.asyncio
async def test_the_vllm_spec_is_untouched_by_the_resolver(tmp_path):
    """resolve_model_paths runs for EVERY backend so there is one code path.
    A vLLM row has filename=None, so it resolves to nothing and plan() ignores
    the argument entirely -- the golden proves the argv did not move."""
    case = next(c for c in CASES if c.name == "minimal")
    (tmp_path / "hf-token").write_text("hf_tok\n")
    driver = _CapturingDriver()
    sup = Supervisor(_CacheSettings(tmp_path), driver=driver)
    await sup.load(case.model, port=case.port, overrides=case.overrides)
    assert list(driver.spec.argv) == ["vllm", "serve", *GOLDEN["minimal"]["argv"]]
    await sup.unload(case.model.id, force=True)


# ---------------------------------------------------------------------------
# Sub-project C, CI regression: a vLLM load must not touch the model cache
#
# Task 8 resolved HF-cache paths UNCONDITIONALLY, for every backend. vLLM never
# needs them -- it takes `--model <repo-id>` and does its own cache lookup -- so
# that added a filesystem scan of a directory vLLM will never open, and with it
# a new failure mode on a path that used to have none.
#
# It is not a test-only quirk. `Path.is_dir()` swallows ENOENT/ENOTDIR/EBADF/
# ELOOP but NOT EACCES, so an unreadable cache root PROPAGATES. CI's runner is
# non-root with HOME=/tmp and cannot stat under /root/.cache/huggingface, and
# six pre-existing vLLM tests started failing there while passing locally as
# root. Any deployment with a momentarily unreadable hf_cache_dir -- wrong
# ownership after a volume remount, a stale NFS mount -- would have failed a
# vLLM load that previously succeeded, and reported it as a GGUF-resolution
# error on a model with no GGUF.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_vllm_load_never_touches_the_model_cache(tmp_path, monkeypatch):
    """The contract, stated directly and independently of any filesystem state:
    resolving is work vLLM does not need, so for a vLLM row it must not happen
    AT ALL. A resolver that raises on sight proves the call is absent rather
    than merely tolerant."""

    def _boom(*a, **k):
        raise AssertionError(
            "resolve_model_paths was called for a vLLM row; vLLM takes a repo "
            "id and must not touch the model cache"
        )

    monkeypatch.setattr("app.runtime.supervisor.resolve_model_paths", _boom)

    case = next(c for c in CASES if c.name == "minimal")
    (tmp_path / "hf-token").write_text("hf_tok\n")
    driver = _CapturingDriver()
    sup = Supervisor(_Settings(tmp_path), driver=driver)
    await sup.load(case.model, port=case.port, overrides=case.overrides)

    assert list(driver.spec.argv) == ["vllm", "serve", *GOLDEN["minimal"]["argv"]]
    await sup.unload(case.model.id, force=True)


@pytest.mark.asyncio
async def test_an_unreadable_cache_root_does_not_break_a_vllm_load(tmp_path, monkeypatch):
    """The CI failure in miniature, reproduced deterministically for any user.

    Monkeypatching Path.is_dir to raise EACCES is exactly what CPython does to
    a non-root process under a mode-700 /root, which is what the runner hit.
    Running the assertion for real would require dropping privileges, so the
    errno is injected instead -- the code path under test is identical.
    """
    import pathlib

    real_is_dir = pathlib.Path.is_dir

    def _eacces(self, *a, **k):
        if "huggingface" in str(self) or "cache" in str(self):
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_dir(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "is_dir", _eacces)

    case = next(c for c in CASES if c.name == "minimal")
    (tmp_path / "hf-token").write_text("hf_tok\n")
    driver = _CapturingDriver()
    sup = Supervisor(_CacheSettings(tmp_path), driver=driver)
    await sup.load(case.model, port=case.port, overrides=case.overrides)

    assert driver.spec.argv[0] == "vllm"
    await sup.unload(case.model.id, force=True)


@pytest.mark.asyncio
async def test_a_llamacpp_load_still_resolves(tmp_path, monkeypatch):
    """And the other half: the fix must be 'resolve when the engine needs a
    path', not 'stop resolving'. llama.cpp cannot launch without one."""
    from tests.unit.runtime.backends.corpus import StubModel

    called = {}
    real = __import__(
        "app.runtime.backends.paths", fromlist=["resolve_model_paths"]
    ).resolve_model_paths

    def _spy(model, **kw):
        called["yes"] = True
        return real(model, **kw)

    monkeypatch.setattr("app.runtime.supervisor.resolve_model_paths", _spy)

    (tmp_path / "hf-token").write_text("hf_tok\n")
    snap = _make_gguf_snapshot(tmp_path / "cache")
    model = StubModel()
    model.backend = "llamacpp"
    model.filename = "model-IQ3_XXS.gguf"

    driver = _CapturingDriver()
    sup = Supervisor(_CacheSettings(tmp_path), driver=driver)
    await sup.load(model, port=10000)

    assert called.get("yes") is True
    assert str(snap / "model-IQ3_XXS.gguf") in driver.spec.argv
    await sup.unload(model.id, force=True)


def test_the_capability_says_which_engines_need_a_path():
    """The gate is a CAPABILITY, not a name check. `if name == "vllm"` in the
    supervisor would reintroduce exactly the per-backend branching in the
    control plane that the Backend axis exists to remove (D1, §6.1) -- every
    future backend would need a new case there. Declaring the fact keeps the
    supervisor at one call site with no engine name in it."""
    from app.runtime.backends import registry

    assert registry.get("vllm").capabilities.needs_local_model_path is False
    assert registry.get("llamacpp").capabilities.needs_local_model_path is True
