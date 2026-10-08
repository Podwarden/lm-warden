"""The measurement fingerprint (design §7.3).

A published limit is only true of the exact thing it was measured on. The
fingerprint decides when it stops being true. Getting it too narrow serves
clients a number from different hardware; too wide and every measurement
expires on contact.

Three of these cases come from measurements taken on pw-bonus 2026-09-03:
the same model crashing on one card and running clean on another (so hardware
identity is load-bearing), enabling ECC changing usable VRAM by 1,024 MiB, and
a co-resident model holding several GB.
"""

from __future__ import annotations

from app.stress.fingerprint import Gpu, Neighbour, Subject, fingerprint


def _subject(**over) -> Subject:
    base = dict(
        hf_repo="Qwen/Qwen3.8-27B",
        hf_revision="main",
        filename=None,
        quantization=None,
        backend="vllm",
        mmproj_filename=None,
        max_model_len=32768,
        max_num_seqs=None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype=None,
        n_gpu_layers=None,
        parallelism_strategy="auto",
        extra_args=(),
        extra_env=(),
        engine_version="0.26.0",
        engine_image=None,
        max_num_batched_tokens=8192,
        proxy_max_inflight=16,
        request_max_wall_s=0.0,
        gpus=(
            Gpu(
                uuid="GPU-aaa",
                name="RTX A4000",
                memory_total_mib=16376,
                compute_cap=8.6,
                ecc_enabled=False,
            ),
        ),
        driver_version="580.65.06",
        neighbours=(),
        probe_suite_hash="suite-v1",
        algorithm_version=1,
    )
    base.update(over)
    return Subject(**base)


# ---- stability -----------------------------------------------------------


def test_identical_subjects_fingerprint_identically():
    assert fingerprint(_subject()) == fingerprint(_subject())


def test_the_fingerprint_is_a_sha256_hex_digest():
    fp = fingerprint(_subject())
    assert fp.startswith("sha256:")
    assert len(fp) == len("sha256:") + 64


# ---- hardware identity ---------------------------------------------------


def test_a_different_gpu_invalidates_the_measurement():
    """The headline evidence: same model, same args, one card crashed at 2.2k
    tokens and the other ran clean to 24,376."""
    other = _subject(
        gpus=(
            Gpu(
                uuid="GPU-bbb",
                name="Quadro RTX 5000",
                memory_total_mib=16384,
                compute_cap=7.5,
                ecc_enabled=True,
            ),
        )
    )
    assert fingerprint(_subject()) != fingerprint(other)


def test_gpus_are_identified_by_uuid_not_by_index():
    """gpu_indices are positional and not stable identity.

    Two cards can swap indices across a reboot; a measurement must not follow
    the index onto different silicon.
    """
    a = Gpu(uuid="GPU-aaa", name="X", memory_total_mib=16000, compute_cap=8.6, ecc_enabled=False)
    b = Gpu(uuid="GPU-bbb", name="X", memory_total_mib=16000, compute_cap=8.6, ecc_enabled=False)
    assert fingerprint(_subject(gpus=(a, b))) == fingerprint(_subject(gpus=(b, a)))


def test_enabling_ecc_invalidates_the_measurement():
    """Measured 2026-09-03: ECC cut usable VRAM 16,384 -> 15,360 MiB."""
    on = _subject(
        gpus=(
            Gpu(
                uuid="GPU-aaa", name="X", memory_total_mib=15360, compute_cap=7.5, ecc_enabled=True
            ),
        )
    )
    off = _subject(
        gpus=(
            Gpu(
                uuid="GPU-aaa", name="X", memory_total_mib=16384, compute_cap=7.5, ecc_enabled=False
            ),
        )
    )
    assert fingerprint(on) != fingerprint(off)


def test_a_driver_upgrade_invalidates_the_measurement():
    assert fingerprint(_subject()) != fingerprint(_subject(driver_version="590.00"))


# ---- neighbours ----------------------------------------------------------


def test_a_co_resident_model_is_part_of_the_measurement():
    """A limit measured while a neighbour held 8 GB is not true once it unloads.

    v1's fingerprint matched perfectly across exactly that change.
    """
    alone = _subject()
    crowded = _subject(neighbours=(Neighbour(model_id="m2", vram_mib=8000),))
    assert fingerprint(alone) != fingerprint(crowded)


def test_neighbour_order_does_not_matter():
    n1, n2 = Neighbour("m2", 8000), Neighbour("m3", 2000)
    assert fingerprint(_subject(neighbours=(n1, n2))) == fingerprint(_subject(neighbours=(n2, n1)))


# ---- engine and warden ---------------------------------------------------


def test_an_engine_version_change_invalidates():
    assert fingerprint(_subject()) != fingerprint(_subject(engine_version="0.27.0"))


def test_max_num_batched_tokens_is_part_of_the_fingerprint():
    """It governs workspace memory, which is what actually OOMs at runtime.

    The warden sets it nowhere, so it is read from the engine — which means it
    can change under us without any config edit we made.
    """
    assert fingerprint(_subject()) != fingerprint(_subject(max_num_batched_tokens=16384))


def test_the_proxy_inflight_cap_is_part_of_the_fingerprint():
    """recommended_concurrency is clamped to it, so changing it changes the
    published number — with no other signal that anything moved."""
    assert fingerprint(_subject()) != fingerprint(_subject(proxy_max_inflight=32))


def test_request_max_wall_s_is_part_of_the_fingerprint():
    """Defaults to 0.0 but docs/operating.md recommends 600 in production.

    Where set, it truncates long generations, so max_tokens_headroom becomes
    fiction above it.
    """
    assert fingerprint(_subject()) != fingerprint(_subject(request_max_wall_s=600.0))


# ---- load config ---------------------------------------------------------


def test_changing_max_model_len_invalidates():
    """Each reload-sweep candidate is therefore its own fingerprint."""
    assert fingerprint(_subject()) != fingerprint(_subject(max_model_len=65536))


def test_extra_args_order_is_significant():
    """argv order can change behaviour, so it is a sequence, not a set."""
    assert fingerprint(_subject(extra_args=("--a", "--b"))) != fingerprint(
        _subject(extra_args=("--b", "--a"))
    )


def test_the_harness_version_invalidates_old_measurements():
    """Changing our own probes makes old numbers incomparable.

    The case most likely to be forgotten, because nothing about the model or
    the machine moved.
    """
    assert fingerprint(_subject()) != fingerprint(_subject(algorithm_version=2))
    assert fingerprint(_subject()) != fingerprint(_subject(probe_suite_hash="suite-v2"))


def test_there_is_no_kv_cache_dtype_field():
    """v1 listed it in the fingerprint; it is not a column or a ModelRow field.

    It exists only as a suggest.py output and is reachable solely through
    extra_args, which is already covered. A phantom field in a list whose whole
    purpose is completeness is a bad sign, so its absence is asserted.
    """
    assert not hasattr(_subject(), "kv_cache_dtype")
