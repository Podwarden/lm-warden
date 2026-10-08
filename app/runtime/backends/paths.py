"""HF repo id -> the absolute files on disk, for backends that need a path.

vLLM takes ``--model <repo-id>`` and does its own cache lookup. llama.cpp takes
``-m <file>``, so something has to bridge the two. That something is this
module, and design spec §6.6 is explicit about what it must NOT be: a new
download mechanism, a second cache, or a PVC change. It only READS the tree that
app/models/pull_task.py's snapshot_download(cache_dir=settings.hf_cache_dir)
already wrote, in the layout app/chat2/catalog.py:53-69 already walks:

    <root>/models--org--name/snapshots/<sha>/<filename>

TWO THINGS THAT LOOK LIKE BUGS AND ARE NOT
------------------------------------------
1. We do not use ``hf_revision``. The row stores a REF ('main'), the directory
   is named after a SHA, and nothing on the row records the mapping. Newest
   snapshot wins -- the same heuristic read_hf_config already uses.

2. We look in both <root> and <root>/hub. The pull task passes
   cache_dir=<root>, while huggingface_hub's own default (HF_HUB_CACHE) is
   <root>/hub, so a repo can legitimately be under either. Observed live on
   2026-09-01: bonus held two complete, NON-hardlinked 11.85 GB copies of
   models--ISTA-DASLab--Qwen3.8-27B-3Bit-GSQ, one in each. Looking in only one
   place would report a model as missing while it sat on disk. The root wins
   when both exist, because the root is where the pull task writes and an
   operator debugging a stale file needs a deterministic answer.

   That duplication is a real pre-existing waste (~17.6 GB of a 35 GB PVC) and
   is filed as its own issue. This module tolerates it; it does not fix it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ModelFileNotFound(Exception):
    """A file the launch needs is not in the cache.

    Raised BEFORE any process starts and before any GPU is claimed, so the
    operator gets 'pull this model' rather than a subprocess that exits rc=1
    forty seconds later with a path deep in a log tail.
    """


@dataclass(frozen=True)
class ResolvedModelPaths:
    """Absolute on-disk paths for one model row. All three may be None.

    ``model_path``  -- the weights file for ``-m``. None when the row pins no
                       filename (a whole-repo safetensors pull), which is the
                       normal vLLM case.
    ``mmproj_path`` -- the multimodal projector for ``--mmproj``.
    ``snapshot_dir``-- the snapshot directory both live in. Kept because sharded
                       GGUF sets resolve by sibling files in the same directory.
    """

    model_path: str | None = None
    mmproj_path: str | None = None
    snapshot_dir: str | None = None


def _snapshot_dir(hf_cache_dir: Path, hf_repo: str) -> Path | None:
    slug = f"models--{hf_repo.replace('/', '--')}"
    # Root first: it is where the pull task writes. See the module docstring.
    for base in (hf_cache_dir, hf_cache_dir / "hub"):
        snaps = base / slug / "snapshots"
        if not snaps.is_dir():
            continue
        candidates = sorted((d for d in snaps.iterdir() if d.is_dir()), reverse=True)
        if candidates:
            return candidates[0]
    return None


def resolve_model_paths(model, *, hf_cache_dir) -> ResolvedModelPaths:
    """Resolve a model row's pinned files inside the HF cache.

    Called once per load by Supervisor.load and handed to Backend.plan() so that
    plan() stays PURE -- the filesystem read happens here, once, where it can be
    reported as a clean error.
    """
    root = Path(hf_cache_dir)
    slug = f"models--{model.hf_repo.replace('/', '--')}"
    snap = _snapshot_dir(root, model.hf_repo)

    filename = getattr(model, "filename", None)
    mmproj = getattr(model, "mmproj_filename", None)

    # ------------------------------------------------------------------
    # WHY THE WEIGHTS FILE IS *NOT* REQUIRED HERE, AND THE PROJECTOR IS
    # ------------------------------------------------------------------
    # This resolver runs for EVERY backend, on every load, so that the
    # supervisor has one code path instead of a branch on the backend's name.
    # That means it must not impose llama.cpp's needs on vLLM.
    #
    # ``filename`` is NOT llama.cpp-only: a vLLM GGUF row sets it too, and vLLM
    # takes ``--model <repo>:<QUANT>`` and resolves the file itself -- possibly
    # by downloading it. Raising here for an unresolvable ``filename`` would
    # refuse to launch vLLM rows that load perfectly well today (the corpus's
    # ``gguf_quant_tag`` case is exactly one). So a weights file we cannot find
    # is reported as ``model_path=None`` and the decision is left to the backend
    # that needs it: ``build_llamacpp_args`` raises, with a message naming the
    # directory it searched, still inside the try/except that releases the GPU
    # claim and still before any process is spawned.
    #
    # ``mmproj_filename`` IS llama.cpp-only -- migration 0028, no vLLM analogue,
    # because vLLM's vision tower lives inside the checkpoint. Nothing else can
    # set it, so requiring it here costs no other backend anything, and it earns
    # its place: a vision model launched without its projector loads, serves,
    # and silently ignores every image.
    if snap is None:
        if mmproj:
            raise ModelFileNotFound(
                f"mmproj projector {mmproj!r} is not in the model cache: the "
                f"repo {model.hf_repo!r} is not there at all. Looked for "
                f"{slug!r} under {root} and {root / 'hub'}. Pull the model "
                f"first."
            )
        return ResolvedModelPaths()

    def _find(name: str) -> str | None:
        p = snap / name
        return str(p) if p.is_file() else None

    model_path = _find(filename) if filename else None

    mmproj_path = None
    if mmproj:
        mmproj_path = _find(mmproj)
        if mmproj_path is None:
            raise ModelFileNotFound(
                f"mmproj projector {mmproj!r} is not in the model cache. Looked "
                f"in {snap}. Pull the model (or re-pull it if the file was "
                f"added to the repo after the last pull)."
            )

    return ResolvedModelPaths(
        model_path=model_path, mmproj_path=mmproj_path, snapshot_dir=str(snap)
    )
