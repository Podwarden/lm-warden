"""Gated-repo typed error path in ``run_pull`` (#88).

Mirrors the discovery-stage ``auth_required`` envelope (#84): when HF Hub
returns 401/403 mid-pull because the repo is gated/private and the operator's
HF token is missing or insufficient, ``run_pull`` must:

1. Classify the exception via ``_classify_hf_auth_error`` (covers both
   ``GatedRepoError`` and ``HfHubHTTPError`` with 401/403 status).
2. Raise ``PullAuthRequired`` so the outer handler catches a typed sentinel
   instead of mis-tagging a "pull error: …" stack in ``last_error``.
3. Persist ``last_error`` with the ``auth_required:`` prefix + an operator
   actionable hint pointing at ``/setup/hf-token``.

Two source code paths can hit the auth refusal — ``estimate_repo_bytes`` (the
pre-download metadata fetch) and ``_snapshot_download`` (the actual file
pull). We exercise both. We also lock the negative case: an unrelated HF
error (e.g. 500) must NOT be misclassified as auth_required.
"""

from __future__ import annotations

import sqlite3

from app.db.database import open_db
from app.db.migrations import apply_migrations
from app.db.repos.models import ModelRepo, ModelRow
from app.models.pull_task import (
    PULL_AUTH_REQUIRED_HINT,
    PULL_AUTH_REQUIRED_PREFIX,
    PullAuthRequired,
    _classify_hf_auth_error,
)

# ---- Classifier unit tests -----------------------------------------------


def _make_fake_response(status_code: int):
    class _Req:
        method = "GET"
        url = "https://huggingface.co/api/models/fake"

    class _R:
        def __init__(self, code: int) -> None:
            self.status_code = code
            self.headers: dict[str, str] = {}
            self.text = ""
            self.request = _Req()

        def json(self):
            return {}

    return _R(status_code)


def test_classifier_maps_gated_repo_error():
    from huggingface_hub.errors import GatedRepoError

    exc = GatedRepoError("gated", response=_make_fake_response(403))
    result = _classify_hf_auth_error(exc)
    assert isinstance(result, PullAuthRequired)


def test_classifier_maps_401_http_error():
    from huggingface_hub.errors import HfHubHTTPError

    exc = HfHubHTTPError("unauthorized", response=_make_fake_response(401))
    result = _classify_hf_auth_error(exc)
    assert isinstance(result, PullAuthRequired)


def test_classifier_maps_403_http_error():
    from huggingface_hub.errors import HfHubHTTPError

    exc = HfHubHTTPError("forbidden", response=_make_fake_response(403))
    result = _classify_hf_auth_error(exc)
    assert isinstance(result, PullAuthRequired)


def test_classifier_passes_through_500_http_error():
    """A 5xx is a server-side problem, NOT an auth refusal — must not be
    misclassified or the operator gets a misleading hint."""
    from huggingface_hub.errors import HfHubHTTPError

    exc = HfHubHTTPError("boom", response=_make_fake_response(500))
    assert _classify_hf_auth_error(exc) is None


def test_classifier_passes_through_unrelated_exception():
    assert _classify_hf_auth_error(RuntimeError("not HF")) is None


# ---- End-to-end run_pull integration --------------------------------------


async def _seed_model(db_path, model_id="m1", repo="meta-llama/Llama-3-70B"):
    async with open_db(db_path) as db:
        await apply_migrations(db)
        await ModelRepo(db).insert(
            ModelRow(
                id=model_id,
                served_model_name=model_id,
                hf_repo=repo,
                hf_revision="main",
                gpu_indices=[0],
                tensor_parallel_size=1,
                dtype=None,
                max_model_len=None,
                gpu_memory_utilization=0.9,
                trust_remote_code=False,
                extra_args=[],
                extra_env={},
                status="registered",
                pulled_bytes=0,
                pulled_total=None,
                last_error=None,
            )
        )


async def test_run_pull_gated_during_snapshot_download_writes_typed_envelope(
    tmp_data_dir,
    monkeypatch,
):
    """The common case: estimate succeeds (HF lets metadata through), but
    snapshot_download trips on a gated weight file. ``last_error`` must
    carry the typed prefix + actionable hint."""
    from huggingface_hub.errors import GatedRepoError

    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path)

    async def fake_estimate(*a, **kw):
        return 1024

    async def fake_disk_check(*a, **kw):
        return None

    async def fake_download(*a, **kw):
        raise GatedRepoError("gated weights", response=_make_fake_response(403))

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute("SELECT status, last_error FROM models WHERE id = 'm1'").fetchone()
    assert status == "failed"
    assert err is not None
    assert err.startswith(PULL_AUTH_REQUIRED_PREFIX + ":")
    assert PULL_AUTH_REQUIRED_HINT in err
    # Operator-facing hint must reference the setup path so they know where
    # to go fix it.
    assert "/setup/hf-token" in err


async def test_run_pull_gated_during_estimate_writes_typed_envelope(
    tmp_data_dir,
    monkeypatch,
):
    """Some gated repos refuse the repo_info() metadata call too — that
    branch must also surface the typed envelope, not get swallowed by the
    pre-#88 ``logger.warning + total = None`` path which would then fail
    with a misleading "pull error" message at snapshot_download time."""
    from huggingface_hub.errors import HfHubHTTPError

    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path)

    async def fake_estimate(*a, **kw):
        raise HfHubHTTPError("metadata-gated", response=_make_fake_response(401))

    download_was_called = False

    async def fake_download(*a, **kw):
        nonlocal download_was_called
        download_was_called = True
        return "/tmp/fake"

    async def fake_disk_check(*a, **kw):
        return None

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute("SELECT status, last_error FROM models WHERE id = 'm1'").fetchone()
    assert status == "failed"
    assert err is not None
    assert err.startswith(PULL_AUTH_REQUIRED_PREFIX + ":")
    # We bailed before reaching snapshot_download — no point hammering HF
    # a second time once we know auth is the blocker.
    assert download_was_called is False


async def test_run_pull_non_auth_hf_error_falls_back_to_generic_pull_error(
    tmp_data_dir,
    monkeypatch,
):
    """Negative-side guard: a 5xx HF outage must NOT be misclassified as
    ``auth_required`` — operators would chase a wrong fix (token rotation)
    for a server-side problem (HF being down)."""
    from huggingface_hub.errors import HfHubHTTPError

    from app.config import load_settings
    from app.models import pull_task as pt

    settings = load_settings()
    await _seed_model(settings.db_path)

    async def fake_estimate(*a, **kw):
        return 1024

    async def fake_download(*a, **kw):
        raise HfHubHTTPError("upstream boom", response=_make_fake_response(503))

    async def fake_disk_check(*a, **kw):
        return None

    monkeypatch.setattr(pt, "estimate_repo_bytes", fake_estimate)
    monkeypatch.setattr(pt, "insufficient_disk", fake_disk_check)
    monkeypatch.setattr(pt, "_snapshot_download", fake_download)

    await pt.run_pull("m1", settings)

    with sqlite3.connect(settings.db_path) as db:
        status, err = db.execute("SELECT status, last_error FROM models WHERE id = 'm1'").fetchone()
    assert status == "failed"
    assert err is not None
    # Generic "pull error" envelope — NOT the auth one.
    assert not err.startswith(PULL_AUTH_REQUIRED_PREFIX + ":")
    assert err.startswith("pull error:")
