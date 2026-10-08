"""Turning an HF repo id into the absolute files llama.cpp needs.

The layout under test is the one huggingface_hub's snapshot_download writes and
app/chat2/catalog.py:53-69 already walks:

    <root>/models--org--name/snapshots/<revision-sha>/<filename>

with the snapshot entries being symlinks into ../../blobs/. Two wrinkles are
real and are tested here because both were observed on the live host:

* The revision on the model row is a REF ('main'), not the snapshot's SHA
  directory name. We resolve by newest-first directory scan, the same way
  read_hf_config does, rather than by trusting the ref.
* The same repo can exist under BOTH <root>/models--... and
  <root>/hub/models--... . On bonus on 2026-09-01 there were two complete,
  non-hardlinked copies of models--ISTA-DASLab--Qwen3.8-27B-3Bit-GSQ, 11.85 GB
  each, because the pull task passes cache_dir=<root> while anything using
  huggingface_hub's own defaults writes to <root>/hub. Resolving only one of
  them would make a model that is plainly on disk look missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.runtime.backends.paths import (
    ModelFileNotFound,
    ResolvedModelPaths,
    resolve_model_paths,
)


@dataclass
class _Row:
    hf_repo: str = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
    hf_revision: str = "main"
    filename: str | None = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
    mmproj_filename: str | None = None


def _make_snapshot(root, repo, sha, files, under_hub=False):
    base = root / "hub" if under_hub else root
    snap = base / f"models--{repo.replace('/', '--')}" / "snapshots" / sha
    snap.mkdir(parents=True)
    for name in files:
        (snap / name).write_bytes(b"gguf")
    return snap


def test_resolves_the_pinned_filename(tmp_path):
    snap = _make_snapshot(tmp_path, _Row().hf_repo, "abc123", ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"])
    got = resolve_model_paths(_Row(), hf_cache_dir=tmp_path)
    assert got.model_path == str(snap / "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf")
    assert got.mmproj_path is None
    assert got.snapshot_dir == str(snap)


def test_resolves_the_mmproj_beside_the_weights(tmp_path):
    snap = _make_snapshot(
        tmp_path,
        _Row().hf_repo,
        "abc123",
        ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf", "mmproj-Qwen3.8-27B-BF16.gguf"],
    )
    row = _Row(mmproj_filename="mmproj-Qwen3.8-27B-BF16.gguf")
    got = resolve_model_paths(row, hf_cache_dir=tmp_path)
    assert got.mmproj_path == str(snap / "mmproj-Qwen3.8-27B-BF16.gguf")


def test_finds_a_repo_that_landed_under_hub(tmp_path):
    snap = _make_snapshot(
        tmp_path,
        _Row().hf_repo,
        "abc123",
        ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"],
        under_hub=True,
    )
    got = resolve_model_paths(_Row(), hf_cache_dir=tmp_path)
    assert got.model_path == str(snap / "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf")


def test_prefers_the_cache_root_over_hub_when_both_exist(tmp_path):
    """Both copies are complete and identical in content; the root is the one
    the pull task writes (cache_dir=<root>), so it is the one we serve from.
    Deterministic beats arbitrary -- an operator debugging a stale file needs to
    know which copy the engine opened."""
    root_snap = _make_snapshot(
        tmp_path, _Row().hf_repo, "abc123", ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"]
    )
    _make_snapshot(
        tmp_path,
        _Row().hf_repo,
        "abc123",
        ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"],
        under_hub=True,
    )
    got = resolve_model_paths(_Row(), hf_cache_dir=tmp_path)
    assert got.model_path.startswith(str(root_snap))


def test_picks_the_newest_snapshot_when_several_exist(tmp_path):
    _make_snapshot(tmp_path, _Row().hf_repo, "aaa", ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"])
    newer = _make_snapshot(tmp_path, _Row().hf_repo, "zzz", ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"])
    got = resolve_model_paths(_Row(), hf_cache_dir=tmp_path)
    assert got.model_path.startswith(str(newer))


def test_missing_weights_file_resolves_to_none_rather_than_raising(tmp_path):
    """This resolver runs for EVERY backend, and ``filename`` is not
    llama.cpp-only: a vLLM GGUF row sets it too and vLLM resolves the file
    itself from ``--model <repo>:<QUANT>``, downloading it if need be. Raising
    here would refuse to launch vLLM rows that load perfectly well today
    (tests/unit/runtime/backends/corpus.py's `gguf_quant_tag` is one).

    So the unresolvable weights file is reported as None and the decision is
    left to the backend that actually needs a path -- build_llamacpp_args
    raises, still before any process starts and still inside the try/except
    that releases the GPU claim.
    """
    snap = _make_snapshot(tmp_path, _Row().hf_repo, "abc123", ["something-else.gguf"])
    got = resolve_model_paths(_Row(), hf_cache_dir=tmp_path)
    assert got.model_path is None
    assert got.snapshot_dir == str(snap)


def test_missing_mmproj_raises_rather_than_launching_blind(tmp_path):
    """A vision model launched without its projector loads, serves, and silently
    ignores every image. Failing here is the difference between a clear error
    and a model that quietly cannot see."""
    _make_snapshot(tmp_path, _Row().hf_repo, "abc123", ["Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"])
    row = _Row(mmproj_filename="mmproj-Qwen3.8-27B-BF16.gguf")
    with pytest.raises(ModelFileNotFound) as e:
        resolve_model_paths(row, hf_cache_dir=tmp_path)
    assert "mmproj" in str(e.value)


def test_no_pinned_filename_resolves_nothing_and_does_not_raise(tmp_path):
    """A whole-repo safetensors row has filename=None. vLLM does not need a
    path at all, and this resolver is called for every backend."""
    _make_snapshot(tmp_path, _Row().hf_repo, "abc123", ["config.json"])
    got = resolve_model_paths(_Row(filename=None), hf_cache_dir=tmp_path)
    assert got == ResolvedModelPaths(
        model_path=None,
        mmproj_path=None,
        snapshot_dir=str(
            tmp_path / "models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF" / "snapshots" / "abc123"
        ),
    )


def test_absent_repo_resolves_to_all_none(tmp_path):
    assert resolve_model_paths(_Row(filename=None), hf_cache_dir=tmp_path) == ResolvedModelPaths(
        model_path=None, mmproj_path=None, snapshot_dir=None
    )


def test_absent_repo_with_a_pinned_filename_resolves_to_all_none(tmp_path):
    """Same reasoning as the previous test: not-pulled-yet is not this
    resolver's error to raise, because a vLLM row in the same state launches."""
    assert resolve_model_paths(_Row(), hf_cache_dir=tmp_path) == ResolvedModelPaths()


def test_absent_repo_with_an_mmproj_raises(tmp_path):
    """The projector IS llama.cpp-only -- migration 0028, no vLLM analogue --
    so requiring it costs no other backend anything, and it earns its place."""
    row = _Row(mmproj_filename="mmproj-Qwen3.8-27B-BF16.gguf")
    with pytest.raises(ModelFileNotFound) as e:
        resolve_model_paths(row, hf_cache_dir=tmp_path)
    assert "mmproj" in str(e.value)
    assert "models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF" in str(e.value)


def test_an_empty_snapshots_dir_is_treated_as_absent(tmp_path):
    """An interrupted pull can leave the directory skeleton with no snapshot
    inside it. That is 'not on disk', not 'on disk and broken'."""
    slug = "models--ISTA-DASLab--Qwen3.8-27B-GSQ-RCO-GGUF"
    (tmp_path / slug / "snapshots").mkdir(parents=True)
    got = resolve_model_paths(_Row(filename=None), hf_cache_dir=tmp_path)
    assert got == ResolvedModelPaths()
