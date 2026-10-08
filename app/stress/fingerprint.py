"""The measurement fingerprint (design §7.3).

A measured limit is only true of the exact thing it was measured on. This is
what decides when it stops being true — and therefore when a client must stop
being told it.

Getting the scope wrong is dangerous in both directions. Too narrow and we
serve a client a number taken from different hardware or a different engine
build. Too wide and every measurement expires on contact and the feature
produces nothing.

Three of the components below are here because of measurements taken on
pw-bonus on 2026-09-03, not because they seemed prudent:

* **GPU identity**, because the same model, file and arguments crashed at
  ~2.2k tokens on one card and ran clean to 24,376 on another in the same host.
* **ECC mode**, because enabling it cut usable VRAM from 16,384 to 15,360 MiB.
* **Neighbours**, because co-residency is normal and a limit measured while a
  neighbour held several GB is not true once that neighbour unloads.

Known gap, stated rather than hidden: a reload with *identical* configuration
produces an identical fingerprint, so a measurement survives it. Whether that
is correct is unknown — the observed non-determinism suggests one load may be
luckier than another. The run row carries ``loads_since_measurement`` so this
can be tested rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Gpu:
    """A GPU by stable identity.

    Keyed on UUID, never on ``gpu_indices``: indices are positional and can be
    reassigned across a reboot, so a measurement keyed on one can silently
    follow the index onto different silicon.
    """

    uuid: str
    name: str
    memory_total_mib: int
    compute_cap: float | None
    ecc_enabled: bool


@dataclass(frozen=True)
class Neighbour:
    """A co-resident model sharing at least one GPU with the subject."""

    model_id: str
    vram_mib: int


@dataclass(frozen=True)
class Subject:
    """Everything that, if changed, makes a measurement no longer apply.

    Note what is NOT here: ``kv_cache_dtype``. v1 of the design listed it, but
    it is not a ``models`` column and not a ``ModelRow`` field — it exists only
    as an output of ``suggest.py`` and reaches the engine solely through
    ``extra_args``, which is already covered. A phantom entry in a list whose
    entire purpose is completeness is worse than a missing one, because it
    reads as coverage.
    """

    # model
    hf_repo: str
    hf_revision: str | None
    filename: str | None
    quantization: str | None
    backend: str
    mmproj_filename: str | None
    # load configuration
    max_model_len: int | None
    max_num_seqs: int | None
    tensor_parallel_size: int | None
    gpu_memory_utilization: float | None
    dtype: str | None
    n_gpu_layers: int | None
    parallelism_strategy: str | None
    extra_args: tuple[str, ...]
    extra_env: tuple[tuple[str, str], ...]
    # engine
    engine_version: str | None
    engine_image: str | None
    #: Governs workspace memory, which is what actually OOMs at runtime. The
    #: warden sets it nowhere, so it is read from the engine — meaning it can
    #: change without any config edit of ours.
    max_num_batched_tokens: int | None
    # warden — both bound published numbers, and neither was in v1
    proxy_max_inflight: int
    request_max_wall_s: float
    # hardware
    gpus: tuple[Gpu, ...]
    driver_version: str | None
    # environment
    neighbours: tuple[Neighbour, ...]
    # harness — changing our own probes makes old numbers incomparable, and
    # this is the case most likely to be forgotten because nothing about the
    # model or the machine moved
    probe_suite_hash: str
    algorithm_version: int


def _canonical(subject: Subject) -> dict:
    """Order-independent where identity is a set, order-preserving where it is not.

    GPUs and neighbours are sorted: which card the driver enumerated first is
    not a property of the measurement. ``extra_args`` is left in order, because
    argv order can change engine behaviour.
    """
    return {
        "model": {
            "hf_repo": subject.hf_repo,
            "hf_revision": subject.hf_revision,
            "filename": subject.filename,
            "quantization": subject.quantization,
            "backend": subject.backend,
            "mmproj_filename": subject.mmproj_filename,
        },
        "load": {
            "max_model_len": subject.max_model_len,
            "max_num_seqs": subject.max_num_seqs,
            "tensor_parallel_size": subject.tensor_parallel_size,
            "gpu_memory_utilization": subject.gpu_memory_utilization,
            "dtype": subject.dtype,
            "n_gpu_layers": subject.n_gpu_layers,
            "parallelism_strategy": subject.parallelism_strategy,
            "extra_args": list(subject.extra_args),
            "extra_env": sorted([list(kv) for kv in subject.extra_env]),
        },
        "engine": {
            "version": subject.engine_version,
            "image": subject.engine_image,
            "max_num_batched_tokens": subject.max_num_batched_tokens,
        },
        "warden": {
            "proxy_max_inflight": subject.proxy_max_inflight,
            "request_max_wall_s": subject.request_max_wall_s,
        },
        "hardware": {
            "gpus": sorted(
                [
                    {
                        "uuid": g.uuid,
                        "name": g.name,
                        "memory_total_mib": g.memory_total_mib,
                        "compute_cap": g.compute_cap,
                        "ecc_enabled": g.ecc_enabled,
                    }
                    for g in subject.gpus
                ],
                key=lambda g: g["uuid"],
            ),
            "driver_version": subject.driver_version,
        },
        "neighbours": sorted(
            [{"model_id": n.model_id, "vram_mib": n.vram_mib} for n in subject.neighbours],
            key=lambda n: n["model_id"],
        ),
        "harness": {
            "probe_suite_hash": subject.probe_suite_hash,
            "algorithm_version": subject.algorithm_version,
        },
    }


def fingerprint(subject: Subject) -> str:
    """``sha256:<hex>`` over the canonical JSON form.

    ``sort_keys`` plus a fixed separator makes the encoding independent of
    dict construction order, so two identical subjects always agree.
    """
    blob = json.dumps(_canonical(subject), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()
