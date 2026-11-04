import asyncio
import fnmatch
import json
import logging
import re
import signal
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from app.config import Settings
from app.db.database import open_db
from app.db.repos.models import ModelRepo
from app.models.sharding import shard_glob_for
from app.system.disk import disk_free_bytes

logger = logging.getLogger(__name__)
SAFETY_FACTOR = 1.5
PROGRESS_POLL_INTERVAL_S = 1.0

# Typed sentinel surfaced on the model row's ``last_error`` field when HF
# rejects a pull with 401/403 — gated or private repo and the operator's
# stored token (if any) isn't sufficient. Mirrors ``DiscoveryAuthRequired``
# from ``app/models/discovery.py`` so the wizard's two stages (discover then
# pull) speak the same error vocabulary; the FE can match this prefix on
# ``last_error`` and surface the same actionable hint instead of a raw
# huggingface_hub stack trace (#88).
PULL_AUTH_REQUIRED_PREFIX = "auth_required"
PULL_AUTH_REQUIRED_HINT = (
    "HuggingFace Hub requires authentication for this repo "
    "(gated or private). Set the HF token via /setup/hf-token and retry."
)


class PullWorkerError(Exception):
    """The downloader child process failed; the message says how (#pull-iso)."""


# The line prefix ``app/models/pull_worker.py`` puts before its JSON verdict.
WORKER_RESULT_MARKER = "@@VW_PULL_RESULT@@ "
_APP_ROOT = Path(__file__).resolve().parents[2]


def _worker_argv() -> list[str]:
    """argv for the downloader child. A function so tests can swap it.

    Carries the parent's ``-I`` / ``-B``: the macOS bundle runs the server as
    ``python -I -B -m uvicorn`` so user site-packages cannot shadow the bundled
    huggingface_hub, and no ``__pycache__`` is written inside the signed .app
    (which would break its seal). The child must keep both.
    """
    flags = []
    if sys.flags.isolated:
        flags.append("-I")
    if sys.flags.dont_write_bytecode:
        flags.append("-B")
    return [sys.executable, *flags, "-m", "app.models.pull_worker"]


# Pulls running in THIS process: model ids, and repo -> number of running
# pulls. The DB status cannot answer "is a download live right now?" — a row
# can read 'pulling' after a crash, and two POSTs can both pass a status check
# before either flips it. Only this process runs downloads, so this is exact.
_ACTIVE_MODELS: set[str] = set()
_ACTIVE_REPOS: dict[str, int] = {}


def pull_is_running(model_id: str) -> bool:
    """True while ``run_pull`` for ``model_id`` is in flight in this process."""
    return model_id in _ACTIVE_MODELS


class PullAuthRequired(Exception):
    """HF Hub returned 401/403 mid-pull for a gated or private repo.

    Raised by the pull-task path so ``run_pull`` can persist a typed,
    operator-facing ``last_error`` on the model row instead of a raw
    ``HfHubHTTPError`` stack. Mirrors ``DiscoveryAuthRequired`` from #84 —
    same condition, different stage (discovery vs. download).
    """


def allow_patterns_for(
    filename: str | None, *, mmproj_filename: str | None = None
) -> list[str] | None:
    """Build the ``allow_patterns`` list for a per-file pull (#85).

    When the wizard pinned a specific weights file, we pull only that file
    plus the bits vLLM needs to load it (config + tokenizer + docs). Returning
    ``None`` preserves the legacy "whole repo" behaviour for legacy rows and
    callers that didn't pick a filename.

    For sharded weight sets, expand the single picked shard into a glob
    covering all sibling shards plus the safetensors weight_map index
    (``model.safetensors.index.json``). Without this, vLLM/transformers
    refuses to load because the index references shards that aren't on
    disk. Fixes vllm-warden#111.

    ``mmproj_filename`` (sub-project C) is llama.cpp's multimodal projector: a
    SEPARATE GGUF in the same repo, passed to llama-server as --mmproj. Without
    it here the projector is never downloaded and the load fails at resolve time
    -- correct, but late, and baffling to an operator who explicitly picked a
    vision model. It is appended rather than folded into the shard glob because
    it is not a member of the weights shard family. It also lands in the size
    ESTIMATE, which shares this list, so the progress bar's denominator is not
    short by the projector's ~888 MB.
    """
    if filename is None:
        return None
    patterns = [filename, "config.json", "tokenizer*", "*.txt", "*.md"]
    shard_glob = shard_glob_for(filename)
    if shard_glob is not None:
        patterns.append(shard_glob)
        if filename.endswith(".safetensors"):
            # vLLM/transformers needs the weight_map to know which tensor
            # lives in which shard.
            patterns.append("*.safetensors.index.json")
        # GGUF sharded sets need no index file — llama.cpp resolves by suffix.
    if mmproj_filename:
        patterns.append(mmproj_filename)
    return patterns


class DiskShortage(Exception):
    pass


def _classify_hf_auth_error(exc: BaseException) -> PullAuthRequired | None:
    """Return a typed sentinel if ``exc`` looks like an HF auth refusal (401/403).

    Examines ``huggingface_hub.errors.GatedRepoError`` and ``HfHubHTTPError``
    with a 401/403 status. Returns None for any other exception so callers
    propagate it unchanged. Imports are local to avoid coupling the module
    load path to the huggingface_hub error hierarchy (mirrors the lazy import
    in ``app/models/discovery.py``).
    """
    try:
        from huggingface_hub.errors import (  # noqa: PLC0415
            GatedRepoError,
            HfHubHTTPError,
        )
    except ImportError:  # pragma: no cover — defensive, huggingface_hub is required
        return None

    if isinstance(exc, GatedRepoError):
        return PullAuthRequired(str(exc))
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return PullAuthRequired(str(exc))
    return None


async def estimate_repo_bytes(
    repo: str,
    revision: str,
    hf_token: str | None,
    allow_patterns: list[str] | None = None,
) -> int:
    """Sum HF-reported sibling sizes, optionally filtered by ``allow_patterns``.

    When ``allow_patterns`` is set, only siblings matching at least one
    glob count toward the estimate — that keeps the disk-shortage check
    aligned with what we'll actually download in the per-file case (#85),
    otherwise the 19.8 GB single-file pulls trip a false disk-shortage
    sized at the whole 200+ GB repo.

    HF 401/403 surface here too (gated repo metadata fetch is itself gated);
    we let them bubble — ``run_pull`` reclassifies via ``_classify_hf_auth_error``.
    """

    def _sync():
        api = HfApi(token=hf_token)
        info = api.repo_info(repo, revision=revision, files_metadata=True)
        total = 0
        for f in info.siblings or []:
            if f.size is None:
                continue
            if allow_patterns is not None and not any(
                fnmatch.fnmatch(f.rfilename, pat) for pat in allow_patterns
            ):
                continue
            total += f.size
        return total

    return await asyncio.to_thread(_sync)


async def insufficient_disk(
    repo: str,
    revision: str,
    cache_dir: Path,
    hf_token: str | None,
    estimate: int | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    if estimate is None:
        estimate = await estimate_repo_bytes(repo, revision, hf_token, allow_patterns)
    free = disk_free_bytes(cache_dir)
    # A re-pull (or a shared-blob revision) only downloads what the cache
    # doesn't already have: subtract the bytes already under this repo's
    # blobs/ dir and floor at 0, so the preflight sizes the actual
    # incremental need instead of assuming a fully-fresh download every time
    # (#272).
    cached = await asyncio.to_thread(_snapshot_dir_size, cache_dir, repo)
    incremental = max(0, estimate - cached)
    needed = int(incremental * SAFETY_FACTOR)
    if needed > free:
        raise DiskShortage(
            f"{repo}@{revision} estimated {estimate // (1024**3)} GiB "
            f"({cached // (1024**3)} GiB already cached, "
            f"{incremental // (1024**3)} GiB to fetch); "
            f"with {SAFETY_FACTOR}x safety needs {needed // (1024**3)} GiB; "
            f"free {free // (1024**3)} GiB"
        )


async def _read_hf_token(settings: Settings) -> str | None:
    p = settings.data_dir / "hf-token"
    if p.exists():
        return p.read_text().strip()
    return None


def _blobs_dir(cache_dir: Path, repo: str) -> Path:
    return Path(cache_dir) / ("models--" + repo.replace("/", "--")) / "blobs"


def _snapshot_dir_size(cache_dir: Path, repo: str, *, complete_only: bool = False) -> int:
    """Sum byte sizes under {cache_dir}/models--{org}--{name}/blobs/.

    huggingface_hub writes downloads to that blobs/ directory before linking
    them into snapshots/. Counting blob bytes, in-flight ``*.incomplete``
    files included, is the live progress signal while a pull runs.
    ``complete_only`` drops the ``*.incomplete`` files: after a pull died,
    only finished blobs are bytes the next attempt will not fetch again.
    Returns 0 if the dir doesn't exist yet (download hasn't started).
    """
    blobs = _blobs_dir(cache_dir, repo)
    if not blobs.exists():
        return 0
    total = 0
    try:
        for entry in blobs.iterdir():
            if complete_only and entry.name.endswith(".incomplete"):
                continue
            try:
                total += entry.stat().st_size
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return 0
    return total


# ``<etag>.<uuid4 hex[:8]>.incomplete``: huggingface_hub >= 1.x gives every
# download attempt its own temp file and never resumes one, so after an
# interrupted attempt these are dead weight. ``<etag>.incomplete`` (no uuid)
# is the older, resumable naming and is left alone.
_ORPHAN_INCOMPLETE = re.compile(r"^[0-9a-f]{40,64}\.[0-9a-f]{8}\.incomplete$")


def purge_orphan_incompletes(cache_dir: Path, repo: str) -> int:
    """Delete partial files left by earlier, dead attempts. Returns bytes freed.

    Only safe while no pull of ``repo`` is running: a live attempt's temp file
    has the same shape. ``run_pull`` checks that before calling.
    On lmwarden.com (2026-11-04) an interrupted pull left 1.2 GB of these; they
    also inflated the progress counter of the next attempt.
    """
    blobs = _blobs_dir(cache_dir, repo)
    if not blobs.is_dir():
        return 0
    freed = 0
    for entry in blobs.iterdir():
        if not _ORPHAN_INCOMPLETE.match(entry.name):
            continue
        try:
            size = entry.stat().st_size
            entry.unlink()
            freed += size
        except FileNotFoundError:
            continue
    return freed


async def _poll_progress(
    model_id: str,
    settings: Settings,
    repo: str,
    total: int | None,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            size = await asyncio.to_thread(_snapshot_dir_size, settings.hf_cache_dir, repo)
            async with open_db(settings.db_path) as db:
                await ModelRepo(db).update_pull_progress(model_id, size, total)
        except Exception:
            logger.exception("progress poll failed for %s", model_id)
        try:
            await asyncio.wait_for(stop.wait(), timeout=PROGRESS_POLL_INTERVAL_S)
        except TimeoutError:
            pass


async def _snapshot_download(
    model_id: str,
    settings: Settings,
    repo: str,
    revision: str,
    token: str | None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Run ``snapshot_download`` in a child process; see app/models/pull_worker.py.

    A native crash or OOM kill in the downloader ends the child, not the
    api process. Raises ``PullAuthRequired`` for an HF auth refusal and
    ``PullWorkerError`` for anything else, including a child that died
    without a verdict.
    """
    request = {
        "repo_id": repo,
        "revision": revision,
        "cache_dir": str(settings.hf_cache_dir),
        "token": token,
        "allow_patterns": allow_patterns,
    }
    proc = await asyncio.create_subprocess_exec(
        *_worker_argv(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
        cwd=str(_APP_ROOT),
    )
    try:
        out, _ = await proc.communicate(json.dumps(request).encode())
    except asyncio.CancelledError:
        # Shutdown or a cancelled task: do not leave a downloader behind.
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise

    verdict: dict[str, Any] | None = None
    for line in out.decode(errors="replace").splitlines():
        if line.startswith(WORKER_RESULT_MARKER):
            try:
                verdict = json.loads(line[len(WORKER_RESULT_MARKER) :])
            except json.JSONDecodeError:
                verdict = None
    if verdict is None:
        rc = proc.returncode
        if rc is not None and rc < 0:
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = str(-rc)
            how = f"was killed by signal {name}"
            if -rc == signal.SIGKILL:
                how += " (often the out-of-memory killer)"
        else:
            how = f"exited with code {rc}"
        raise PullWorkerError(f"downloader process {how} before finishing {repo}")
    if verdict.get("ok"):
        return str(verdict.get("path", ""))
    error = str(verdict.get("error") or "unknown error")
    if verdict.get("auth"):
        raise PullAuthRequired(error)
    raise PullWorkerError(error)


async def run_pull(model_id: str, settings: Settings, force: bool = False) -> None:
    # A second run of a pull already in flight (double-click, or an operator
    # POST racing the boot-time resume) leaves it to the first. The check and
    # the add have no await between them, so in one event loop they are atomic.
    if model_id in _ACTIVE_MODELS:
        logger.info("pull %s is already running; not starting a second one", model_id)
        return
    _ACTIVE_MODELS.add(model_id)
    try:
        # Phase 1: read row + flip status to pulling
        async with open_db(settings.db_path) as db:
            repo = ModelRepo(db)
            row = await repo.get(model_id)
            if not row:
                logger.warning("run_pull: model %s missing", model_id)
                return
            await repo.update_status(model_id, "pulling")
        repo_busy = _ACTIVE_REPOS.get(row.hf_repo, 0) > 0
        _ACTIVE_REPOS[row.hf_repo] = _ACTIVE_REPOS.get(row.hf_repo, 0) + 1
        try:
            await _run_pull_claimed(model_id, settings, row, force=force, repo_busy=repo_busy)
        finally:
            left = _ACTIVE_REPOS.get(row.hf_repo, 1) - 1
            if left > 0:
                _ACTIVE_REPOS[row.hf_repo] = left
            else:
                _ACTIVE_REPOS.pop(row.hf_repo, None)
    finally:
        _ACTIVE_MODELS.discard(model_id)


async def _run_pull_claimed(
    model_id: str, settings: Settings, row: Any, *, force: bool, repo_busy: bool
) -> None:
    # Dead partial files from earlier attempts: delete them, unless another
    # pull of the same repo is running in this process right now (its live
    # temp file looks exactly like an orphan).
    if not repo_busy:
        try:
            freed = await asyncio.to_thread(
                purge_orphan_incompletes, settings.hf_cache_dir, row.hf_repo
            )
            if freed:
                logger.info(
                    "pull %s: removed %d bytes of partial files from earlier attempts",
                    model_id,
                    freed,
                )
        except OSError:
            logger.warning("pull %s: could not clear old partial files", model_id, exc_info=True)

    token = await _read_hf_token(settings)

    # Per-file pull (#85): when the wizard pinned ``filename``, ``allow_patterns``
    # narrows snapshot_download to that file + config/tokenizer/docs. Same
    # patterns get plumbed into estimate_repo_bytes so the disk-shortage check
    # sees the filtered size — otherwise a 19.8 GB single-file pull would trip
    # a false shortage sized at the whole 200+ GB repo.
    allow_patterns = allow_patterns_for(
        row.filename, mmproj_filename=getattr(row, "mmproj_filename", None)
    )

    # Phase 2: estimate total + persist (drives the progress bar denominator),
    # disk check (unless forced), then download with a polling coroutine that
    # walks the cache dir and writes pulled_bytes every PROGRESS_POLL_INTERVAL_S.
    final_status: str
    last_error: str | None
    total: int | None = None
    poller: asyncio.Task | None = None
    stop = asyncio.Event()
    try:
        try:
            total = await estimate_repo_bytes(row.hf_repo, row.hf_revision, token, allow_patterns)
        except Exception as e:
            # Pre-download metadata fetch: a 401/403 here is the same gated-repo
            # signal we get from snapshot_download — re-raise as the typed
            # sentinel so the outer handler writes a stable ``last_error`` (#88).
            # Any other failure is non-fatal at this stage (we'll just lose
            # the progress-bar denominator and let snapshot_download do its
            # own retries/errors).
            auth = _classify_hf_auth_error(e)
            if auth is not None:
                raise auth from e
            logger.warning("repo_info failed for %s: %s", row.hf_repo, e)
            total = None
        # Start from the bytes already finished (a resumed pull keeps them),
        # not 0: the bar must not jump back to empty and climb again.
        done = await asyncio.to_thread(
            _snapshot_dir_size, settings.hf_cache_dir, row.hf_repo, complete_only=True
        )
        async with open_db(settings.db_path) as db:
            await ModelRepo(db).update_pull_progress(model_id, done, total)

        if not force:
            await insufficient_disk(
                row.hf_repo,
                row.hf_revision,
                settings.hf_cache_dir,
                token,
                estimate=total,
                allow_patterns=allow_patterns,
            )

        poller = asyncio.create_task(_poll_progress(model_id, settings, row.hf_repo, total, stop))
        try:
            await _snapshot_download(
                model_id,
                settings,
                row.hf_repo,
                row.hf_revision,
                token,
                allow_patterns=allow_patterns,
            )
        except Exception as e:
            # snapshot_download raises GatedRepoError / 401 HfHubHTTPError when
            # the operator's HF token is missing or doesn't cover the repo.
            # Map to PullAuthRequired so ``last_error`` carries the
            # ``auth_required:`` prefix the FE / docs key off — operators see
            # an actionable hint instead of a raw urllib3 stack (#88).
            auth = _classify_hf_auth_error(e)
            if auth is not None:
                raise auth from e
            raise
        final_status, last_error = "pulled", None
    except DiskShortage as e:
        final_status, last_error = "failed", str(e)
    except PullAuthRequired as e:
        # Operator-actionable error envelope. ``last_error`` is the source of
        # truth the FE reads, so prefix-tag + hint go directly into the column.
        logger.warning("pull auth_required for %s@%s: %s", row.hf_repo, row.hf_revision, e)
        final_status = "failed"
        last_error = f"{PULL_AUTH_REQUIRED_PREFIX}: {PULL_AUTH_REQUIRED_HINT} ({e})"
    except Exception as e:
        logger.exception("pull failed")
        final_status, last_error = "failed", f"pull error: {e}"
    finally:
        stop.set()
        if poller is not None:
            try:
                await poller
            except Exception:
                logger.exception("progress poller raised on shutdown")

    # Phase 3: write a final pulled_bytes (post-completion size) + flip status
    async with open_db(settings.db_path) as db:
        m_repo = ModelRepo(db)
        try:
            final_size = await asyncio.to_thread(
                _snapshot_dir_size, settings.hf_cache_dir, row.hf_repo
            )
            await m_repo.update_pull_progress(model_id, final_size, total)
        except Exception:
            logger.exception("final progress write failed")
        await m_repo.update_status(model_id, final_status, last_error=last_error)
