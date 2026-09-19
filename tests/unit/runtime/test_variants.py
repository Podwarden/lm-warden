"""Model variants (0036, app/runtime/variants.py): which exact configuration of
a model served a request.

Covers:
  * the id is stable across processes and independent of field/key order
  * any identity field changes it; placement-only knobs do not
  * extra_env -- secrets such as HF_TOKEN -- never reaches the descriptor
  * load overrides win over the row for the fields the argv builders take them
  * secret-looking extra_args flags are redacted to a digest, in both forms
  * runtime facts: the image the driver actually ran (pin or default), the
    in-container engine's baked version, and the commit the revision resolves
    to -- so a warden upgrade is a new variant and the same launch is not
  * attribution follows the RUNNING engine: editing the row does not move it,
    a restart does; a model this supervisor did not launch is computed from the
    row (not cached)
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.db.repos.models import ModelRow
from app.runtime.supervisor import Supervisor
from app.runtime.variants import (
    VARIANT_FIELDS,
    canonical_json,
    descriptor_of,
    redact_args,
    resolve_hf_commit,
    running_variant,
    variant_id,
    variant_of,
)
from tests.fakes.fake_engine import FakeDriver


def _row(**kw) -> ModelRow:
    base = dict(
        id="m1", served_model_name="qwen", hf_repo="Qwen/Qwen3-8B-AWQ", hf_revision="main",
        gpu_indices=[0], tensor_parallel_size=1, dtype="auto", max_model_len=32768,
        gpu_memory_utilization=0.9, trust_remote_code=False, extra_args=["--enforce-eager"],
        status="loaded", pulled_bytes=0, pulled_total=None, last_error=None,
        extra_env={"OPENAI_API_KEY": "sk-supersecret"}, backend="vllm",
        engine_channel="stable", engine_vllm_version="0.11.0",
        engine_image="vllm/vllm-openai:v0.11.0",
    )
    base.update(kw)
    return ModelRow(**base)


# ---- identity ----------------------------------------------------------------


def test_the_id_is_pinned():
    # A literal, so a change to the hashing (or to VARIANT_FIELDS) that would
    # re-attribute every running model shows up as a failing test.
    d = descriptor_of(_row())
    assert canonical_json(d) == (
        '{"backend":"vllm","dtype":"auto","engine_channel":"stable",'
        '"engine_image":"vllm/vllm-openai:v0.11.0","engine_image_used":null,'
        '"engine_version":null,"engine_vllm_version":"0.11.0",'
        '"extra_args":["--enforce-eager"],"filename":null,"hf_commit":null,'
        '"hf_config_repo":null,"hf_repo":"Qwen/Qwen3-8B-AWQ","hf_revision":"main",'
        '"max_model_len":32768,"max_num_seqs":null,"mmproj_filename":null,'
        '"quantization":null,"tokenizer_repo":null,"trust_remote_code":false}'
    )
    assert variant_id("m1", d) == variant_of(_row()).id
    assert len(variant_of(_row()).id) == 16
    assert variant_of(_row()).id == "415d5b292f780aa3"


def test_key_order_does_not_matter():
    d = descriptor_of(_row())
    reversed_d = dict(reversed(list(d.items())))
    assert list(reversed_d) != list(d)
    assert variant_id("m1", reversed_d) == variant_id("m1", d)


@pytest.mark.parametrize(("field", "value"), [
    ("engine_vllm_version", "0.9.0"), ("engine_image", "vllm/vllm-openai:v0.9.0"),
    ("engine_channel", "nightly"), ("backend", "llamacpp"), ("hf_revision", "a1b2c3d"),
    ("hf_repo", "Qwen/Qwen3-8B-FP8"), ("filename", "q4.gguf"), ("dtype", "bfloat16"),
    ("max_model_len", 8192), ("trust_remote_code", True), ("extra_args", []),
    ("tokenizer_repo", "Qwen/Qwen3-8B"), ("hf_config_repo", "Qwen/Qwen3-8B"),
    ("mmproj_filename", "mmproj.gguf"),
])
def test_an_identity_field_makes_a_new_variant(field, value):
    assert variant_of(_row(**{field: value})).id != variant_of(_row()).id


@pytest.mark.parametrize(("field", "value"), [
    ("gpu_indices", [1, 2]), ("tensor_parallel_size", 2), ("gpu_memory_utilization", 0.5),
    ("parallelism_strategy", "pipeline"), ("max_batch_size", 8), ("n_gpu_layers", 20),
    ("status", "loading"), ("served_model_name", "renamed"),
    ("extra_env", {"HF_TOKEN": "hf_other"}),
])
def test_placement_and_bookkeeping_do_not(field, value):
    assert variant_of(_row(**{field: value})).id == variant_of(_row()).id


def test_extra_env_never_reaches_the_descriptor():
    assert "extra_env" not in VARIANT_FIELDS
    v = variant_of(_row(extra_env={"HF_TOKEN": "hf_supersecret"}))
    assert "extra_env" not in v.descriptor
    assert "hf_supersecret" not in canonical_json(v.descriptor)
    assert "sk-supersecret" not in canonical_json(variant_of(_row()).descriptor)


def test_the_model_id_is_part_of_the_identity():
    # Two rows serving the same repo the same way are still two models.
    assert variant_of(_row(id="m2")).id != variant_of(_row()).id


def test_load_overrides_win_for_the_fields_the_engine_takes_them_for():
    row = _row()
    v = variant_of(row, {"quantization": "awq", "max_num_seqs": 16, "max_model_len": 4096,
                         "gpu_memory_utilization": 0.5})
    assert (v.descriptor["quantization"], v.descriptor["max_num_seqs"],
            v.descriptor["max_model_len"]) == ("awq", 16, 4096)
    # A placement override is not identity.
    assert variant_of(row, {"gpu_memory_utilization": 0.5}).id == variant_of(row).id


def test_a_null_backend_is_vllm():
    assert variant_of(_row(backend=None)).id == variant_of(_row(backend="vllm")).id


# ---- attribution follows the running engine ------------------------------------


class _Settings:
    def __init__(self, tmp_path):
        self.data_dir = str(tmp_path)
        self.hf_token_path = str(tmp_path / "tok")
        self.hf_cache_dir = str(tmp_path / "hf-cache")


class _ImageDriver(FakeDriver):
    """The docker driver's shape: runs ``spec.image or default_image``."""

    def __init__(self, default_image: str) -> None:
        super().__init__()
        self.default_image = default_image


@pytest.mark.asyncio
async def test_editing_the_row_does_not_move_attribution_a_restart_does(tmp_path):
    sup = Supervisor(_Settings(tmp_path), driver=FakeDriver())
    state = SimpleNamespace(supervisor=sup)
    old = _row()
    await sup.load(old, port=8001)
    launched = sup.get_variant("m1")
    assert launched is not None
    assert running_variant(state, old) == launched

    # The try-stack / settings form rewrites the row while the engine runs.
    edited = replace(old, engine_vllm_version="0.9.0", engine_image="vllm/vllm-openai:v0.9.0",
                     dtype="float16")
    assert running_variant(state, edited) == launched

    await sup.unload("m1", force=True)
    assert sup.get_variant("m1") is None
    await sup.load(edited, port=8001)
    restarted = running_variant(state, edited)
    assert restarted == sup.get_variant("m1") != launched
    assert restarted.descriptor["dtype"] == "float16"


@pytest.mark.asyncio
async def test_the_same_launch_is_the_same_variant(tmp_path):
    sup = Supervisor(_Settings(tmp_path), driver=_ImageDriver("vllm/vllm-openai:v0.10.1"))
    await sup.load(_row(), port=8001)
    first = sup.get_variant("m1")
    await sup.unload("m1", force=True)
    await sup.load(_row(), port=8001)
    assert sup.get_variant("m1") == first


@pytest.mark.asyncio
async def test_a_warden_upgrade_moving_the_default_image_is_a_new_variant(tmp_path):
    # The row pins NO engine: which one runs is the driver's default image.
    row = _row(engine_channel=None, engine_vllm_version=None, engine_image=None)
    ids = []
    for image in ("vllm/vllm-openai:v0.10.1", "vllm/vllm-openai:v0.11.0"):
        sup = Supervisor(_Settings(tmp_path), driver=_ImageDriver(image))
        await sup.load(row, port=8001)
        v = sup.get_variant("m1")
        assert v.descriptor["engine_image_used"] == image
        assert v.descriptor["engine_version"] is None  # the image says it
        ids.append(v.id)
        await sup.unload("m1", force=True)
    assert ids[0] != ids[1]


@pytest.mark.asyncio
async def test_a_pinned_image_is_the_image_used(tmp_path):
    sup = Supervisor(_Settings(tmp_path), driver=_ImageDriver("vllm/vllm-openai:v0.10.1"))
    await sup.load(_row(), port=8001)  # pins vllm/vllm-openai:v0.11.0
    assert sup.get_variant("m1").descriptor["engine_image_used"] == "vllm/vllm-openai:v0.11.0"


@pytest.mark.asyncio
async def test_a_warden_upgrade_moving_the_in_container_engine_is_a_new_variant(
    tmp_path, monkeypatch
):
    from app.runtime import variants

    row = _row(engine_channel=None, engine_vllm_version=None, engine_image=None)
    ids = []
    for baked in ("0.10.1", "0.11.0"):
        monkeypatch.setattr(variants, "baked_engine_version", lambda _b, v=baked: v)
        sup = Supervisor(_Settings(tmp_path), driver=FakeDriver())  # no image
        await sup.load(row, port=8001)
        v = sup.get_variant("m1")
        assert (v.descriptor["engine_image_used"], v.descriptor["engine_version"]) == (None, baked)
        ids.append(v.id)
        await sup.unload("m1", force=True)
    assert ids[0] != ids[1]


@pytest.mark.asyncio
async def test_the_launch_overrides_are_part_of_the_running_variant(tmp_path):
    sup = Supervisor(_Settings(tmp_path), driver=FakeDriver())
    await sup.load(_row(), port=8001, overrides={"quantization": "fp8"})
    assert sup.get_variant("m1").descriptor["quantization"] == "fp8"


def test_a_model_the_supervisor_did_not_launch_is_computed_from_the_row(tmp_path):
    sup = Supervisor(_Settings(tmp_path), driver=FakeDriver())
    state = SimpleNamespace(supervisor=sup)
    assert running_variant(state, _row()) == variant_of(_row())
    assert sup.get_variant("m1") is None  # not cached: the supervisor owns that


def test_without_a_supervisor_it_is_computed_from_the_row():
    assert running_variant(SimpleNamespace(), _row()) == variant_of(_row())


def test_descriptor_is_json_even_for_odd_values():
    odd = SimpleNamespace(id="m9", served_model_name="x", dtype=object(), extra_args=None)
    json.loads(canonical_json(variant_of(odd).descriptor))


# ---- the HF revision resolves to a commit ----------------------------------------

SHA_A = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
SHA_B = "ffffffffffffffffffffffffffffffffffffffff"


def _ref(root, repo, rev, sha, hub=False):
    d = (root / "hub" if hub else root) / f"models--{repo.replace('/', '--')}" / "refs"
    d.mkdir(parents=True, exist_ok=True)
    (d / rev).write_text(sha + "\n")


@pytest.mark.parametrize("hub", [False, True])
def test_a_ref_resolves_through_the_cache(tmp_path, hub):
    _ref(tmp_path, "Qwen/Q", "main", SHA_A, hub=hub)
    assert resolve_hf_commit(tmp_path, "Qwen/Q", "main") == SHA_A


def test_without_a_cache_entry_the_ref_is_kept(tmp_path):
    assert resolve_hf_commit(tmp_path, "Qwen/Q", "main") == "main"
    assert resolve_hf_commit(None, "Qwen/Q", "main") == "main"
    assert resolve_hf_commit(tmp_path, "Qwen/Q", SHA_B) == SHA_B  # already a commit
    assert resolve_hf_commit(tmp_path, "Qwen/Q", None) is None


@pytest.mark.asyncio
async def test_main_moving_upstream_is_a_new_variant(tmp_path):
    settings = _Settings(tmp_path)
    cache = tmp_path / "hf-cache"
    ids = []
    for sha in (SHA_A, SHA_B):
        _ref(cache, "Qwen/Qwen3-8B-AWQ", "main", sha)
        sup = Supervisor(settings, driver=FakeDriver())
        await sup.load(_row(), port=8001)
        v = sup.get_variant("m1")
        assert v.descriptor["hf_commit"] == sha
        assert v.descriptor["hf_revision"] == "main"
        ids.append(v.id)
        await sup.unload("m1", force=True)
    assert ids[0] != ids[1]


# ---- secrets in extra_args ----------------------------------------------------------


def _digest(v):
    import hashlib
    return "<redacted:" + hashlib.sha256(v.encode()).hexdigest()[:8] + ">"


def test_secret_flag_values_are_redacted_in_both_forms():
    args = ["--api-key", "sk-live-1", "--hf-token=hf_abc", "--enforce-eager",
            "--ssl-keyfile-password", "p4ss", "--port", "8000", "--Auth-Header=Bearer x"]
    assert redact_args(args) == [
        "--api-key", _digest("sk-live-1"), f"--hf-token={_digest('hf_abc')}", "--enforce-eager",
        "--ssl-keyfile-password", _digest("p4ss"), "--port", "8000",
        f"--Auth-Header={_digest('Bearer x')}",
    ]


def test_throughput_and_tokenizer_flags_stay_readable():
    args = ["--max-num-batched-tokens", "8192", "--tokenizer", "org/tok",
            "--tokenizer-mode=slow", "--ssl-keyfile", "/certs/key.pem"]
    assert redact_args(args) == args


def test_more_credential_flag_shapes_are_redacted():
    for flag in ("--token", "--api-keys", "--client-secret", "--password",
                 "--auth-token", "--api-credentials"):
        assert redact_args([flag, "s3cr3t"]) == [flag, _digest("s3cr3t")], flag


def test_a_secret_flag_followed_by_another_flag_redacts_nothing():
    assert redact_args(["--use-auth", "--enforce-eager"]) == ["--use-auth", "--enforce-eager"]


def test_secrets_never_reach_the_descriptor_but_still_split_variants():
    a = variant_of(_row(extra_args=["--api-key", "sk-one", "--hf-token=hf_one"]))
    b = variant_of(_row(extra_args=["--api-key", "sk-two", "--hf-token=hf_one"]))
    c = variant_of(_row(extra_args=["--api-key=sk-one", "--hf-token=hf_two"]))
    for v, secrets in ((a, ("sk-one", "hf_one")), (b, ("sk-two",)), (c, ("sk-one", "hf_two"))):
        stored = canonical_json(v.descriptor)
        for secret in secrets:
            assert secret not in stored
    assert len({a.id, b.id, c.id}) == 3
