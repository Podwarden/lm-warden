"""Pure-sync HF cache walker.

Decoupled from FastAPI so unit tests can drop a few directories into a
tmpdir and assert on the returned list without standing up an app.
Route handlers wrap calls in ``asyncio.to_thread`` so a slow walk on a
large cache doesn't block the event loop.

The HF library writes its cache as ``models--<org>--<name>/`` directories
under either ``HF_HOME`` (``$HF_HOME/hub/``) or ``HUGGINGFACE_HUB_CACHE``
(directly under the configured root). vllm-warden's ``VW_HF_CACHE_DIR``
maps to whichever the runtime container ends up exporting, so the
scanner probes BOTH ``<root>/`` and ``<root>/hub/`` and MERGES the two
layouts for a given repo id into a single :class:`CachedRepo` — one row
per repo, never two.

Byte counting is deduplicated at the inode level, not the path level:
every file (including a ``snapshots/<rev>/file`` symlink, followed to
its target) is keyed by ``(st_dev, st_ino)`` in a set shared across
BOTH layouts for that repo, so a blob referenced from ``blobs/<sha>``
directly and again via a symlink under ``snapshots/`` — or shared
between the ``<root>/`` and ``<root>/hub/`` layouts — contributes its
bytes exactly once (vllm-warden#238: the previous path-level dedup let
the symlink's target bytes get summed a second time on top of the real
file, doubling reported size and, since the two layouts were never
merged, also reporting each repo twice).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The HF library encodes ``<org>/<name>`` as ``models--<org>--<name>``.
# Same convention ``pull_task._snapshot_dir_size`` walks in reverse.
_PREFIX = "models--"


@dataclass(frozen=True)
class CachedRepo:
    """A single decoded ``models--<org>--<name>`` directory on disk.

    ``repo`` is the decoded ``<org>/<name>`` HF repo id (the inverse of
    the on-disk encoding). ``path`` is the absolute path so callers
    that need to delete the dir don't have to reconstruct it. ``size_bytes``
    sums ``blobs/ + snapshots/ + refs/`` (all of HF's per-repo state).
    ``last_modified`` is the max mtime seen anywhere under the dir —
    proxies "when did vllm-warden last touch this".
    """

    repo: str
    path: Path
    size_bytes: int
    last_modified: float


def _decode_repo_name(dirname: str) -> str | None:
    """Convert ``models--Qwen--Qwen3.6-27B-GGUF`` to ``Qwen/Qwen3.6-27B-GGUF``.

    Returns None if ``dirname`` doesn't match the HF cache layout — the
    caller skips it. Note "--" inside a repo name (e.g.
    ``models--org--has--double--dashes``) is ambiguous; HF treats the FIRST
    ``--`` after the prefix as the org/name split and leaves the rest
    verbatim. We mirror that with a single ``replace("--", "/", 1)`` —
    consistent with how HF's own ``_get_repo_id_from_cache_dir`` decodes,
    and the same convention vllm-warden's pull task uses on the way in.
    """
    if not dirname.startswith(_PREFIX):
        return None
    body = dirname[len(_PREFIX) :]
    if not body:
        return None
    # First "--" separates org from repo; rest of the name (which may
    # contain dashes or even "--" inside a repo slug) is preserved.
    return body.replace("--", "/", 1)


def _walk_size_and_mtime(roots: list[Path], seen: set[tuple[int, int]]) -> tuple[int, float]:
    """Recursive ``du`` + max mtime across one repo's on-disk directories.

    ``roots`` is every directory that backs this repo (typically just
    ``<cache_root>/models--<org>--<name>``, but a repo split across the
    ``<root>/`` and ``<root>/hub/`` layouts passes both). ``seen`` is an
    ``(st_dev, st_ino)`` set the caller creates per-repo and threads
    through every root here — that's what makes the count correct:

    - The HF cache stores real bytes under ``blobs/<sha>`` and symlinks
      them from ``snapshots/<rev>/file``. ``os.stat`` (follow_symlinks=
      True, the default) on the symlink returns the BLOB's inode, so the
      blob and every symlink pointing at it collapse to one ``seen``
      entry — the bytes are added exactly once no matter how many
      snapshot revisions reference it.
    - The same set is shared across the ``<root>/`` and ``<root>/hub/``
      directories for one repo (see ``scan_hf_cache``), so a file that
      happens to be hardlinked or symlinked between the two layouts is
      likewise counted once, not once per layout.

    Tolerates partially-unreadable subtrees: a single ``PermissionError``
    on a blob is logged and skipped, the rest of the walk continues.
    """
    total = 0
    latest = 0.0
    for root in roots:
        try:
            # ``stat_result.st_mtime`` on the root itself is the floor
            # for an empty repo dir — captures the directory creation
            # timestamp.
            latest = max(latest, root.stat().st_mtime)
        except OSError as exc:
            logger.debug("scan: stat root failed %s: %s", root, exc)

        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            try:
                d_stat = os.stat(dirpath)
                if d_stat.st_mtime > latest:
                    latest = d_stat.st_mtime
            except OSError as exc:
                logger.debug("scan: stat dir %s failed: %s", dirpath, exc)
            for name in filenames:
                full = os.path.join(dirpath, name)
                try:
                    # follow_symlinks=True: resolves snapshots/<rev>/file
                    # to the blobs/<sha> it targets so the dedupe key
                    # below is the BLOB's identity, not the symlink's.
                    st = os.stat(full)
                except OSError as exc:
                    logger.debug("scan: stat %s failed: %s", full, exc)
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += st.st_size
                if st.st_mtime > latest:
                    latest = st.st_mtime
    return total, latest


def _collect_repo_dirs(cache_root: Path) -> list[tuple[str, list[Path]]]:
    """Group on-disk ``models--<org>--<name>`` directories by decoded repo id.

    Probes both ``cache_root/`` and ``cache_root/hub/`` (HF layout drift).
    A repo present under both comes back as ONE ``(repo, dirs)`` pair
    with ``len(dirs) == 2`` — merging the layouts here (by repo id,
    rather than deduplicating by directory path like the old
    implementation) is what stops the same repo from being reported
    twice.

    ``dirs`` preserves probe order (``cache_root`` before
    ``cache_root/hub``), so ``dirs[0]`` is always the ``cache_root``
    entry when both exist — that's the directory that actually holds
    the model weights, so it's the one worth surfacing as the repo's
    display path.
    """
    order: list[str] = []
    by_repo: dict[str, list[Path]] = {}
    for parent in (cache_root, cache_root / "hub"):
        if not parent.exists() or not parent.is_dir():
            continue
        try:
            entries = list(parent.iterdir())
        except OSError as exc:
            logger.warning("scan: iterdir %s failed: %s", parent, exc)
            continue
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            repo = _decode_repo_name(entry.name)
            if repo is None:
                continue
            if repo not in by_repo:
                by_repo[repo] = []
                order.append(repo)
            by_repo[repo].append(entry)
    return [(repo, by_repo[repo]) for repo in order]


def scan_hf_cache(cache_root: Path) -> list[CachedRepo]:
    """Walk ``cache_root`` for ``models--<org>--<name>`` directories.

    Probes both ``cache_root/`` and ``cache_root/hub/`` (HF layout drift)
    and merges them into exactly one :class:`CachedRepo` per repo id,
    with bytes deduplicated at the inode level (see
    ``_walk_size_and_mtime``) so a blob referenced by a symlink is never
    counted twice. Tolerates:

    - missing ``cache_root`` (returns ``[]``);
    - unreadable individual subtrees (logged + skipped);
    - directories that don't match the prefix (skipped);
    - exotic encodings (skipped, logged).

    Never raises. The route handler calls this via
    ``asyncio.to_thread`` so a slow walk on a multi-100-GiB cache does
    not block the event loop.
    """
    if not cache_root.exists():
        return []

    out: list[CachedRepo] = []
    for repo, dirs in _collect_repo_dirs(cache_root):
        seen: set[tuple[int, int]] = set()
        try:
            size, mtime = _walk_size_and_mtime(dirs, seen)
        except Exception as exc:  # belt-and-suspenders: never raise
            logger.warning("scan: walk %s failed: %s", repo, exc)
            continue
        out.append(
            CachedRepo(
                repo=repo,
                path=dirs[0].resolve(),
                size_bytes=size,
                last_modified=mtime,
            )
        )
    return sorted(out, key=lambda r: r.repo)
