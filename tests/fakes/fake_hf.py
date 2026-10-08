"""Fake ``huggingface_hub`` surface for unit tests.

We hand-roll a minimal stand-in rather than mock the real HfApi because the
discovery path (#84) only consumes a small slice of its surface: ``model_info``
returning an object with ``siblings`` / ``private`` / ``gated`` attributes,
plus a config.json fetcher. Both seams are dependency-injected, so the tests
pass these fakes directly without touching imports.

Dev-2 reuses this in #86 (FE-driven discovery wiring); keep the surface here
stable when adding to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeSibling:
    """Stand-in for ``huggingface_hub.RepoSibling``."""

    rfilename: str
    size: int | None = None
    lfs: Any | None = None


@dataclass
class FakeModelInfo:
    """Stand-in for ``huggingface_hub.ModelInfo``."""

    id: str = "fake/repo"
    private: bool = False
    gated: bool = False
    siblings: list[FakeSibling] = field(default_factory=list)


class FakeHfApi:
    """Hand-rolled HfApi stand-in.

    Construct with ``info`` to return on ``model_info`` calls, or an
    ``Exception`` instance to raise. The test passes a factory
    ``lambda token: FakeHfApi(...)`` to ``discover_repo_files``.
    """

    def __init__(
        self,
        info: FakeModelInfo | Exception | None = None,
        *,
        token: str | None = None,
    ) -> None:
        self._info = info
        self.token = token
        self.calls: list[dict[str, Any]] = []

    def model_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        files_metadata: bool = False,
    ) -> FakeModelInfo:
        self.calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "files_metadata": files_metadata,
            }
        )
        if isinstance(self._info, Exception):
            raise self._info
        if self._info is None:
            return FakeModelInfo()
        return self._info


def make_hf_api_factory(api: FakeHfApi):
    """Return a callable matching the ``HfApiFactory`` Protocol shape."""

    def factory(token: str | None) -> FakeHfApi:
        api.token = token
        return api

    return factory


def make_config_fetcher(config: dict | Exception | None):
    """Return a callable matching the ``ConfigFetcher`` Protocol shape.

    ``None`` => returns None (simulates ``EntryNotFoundError`` already
    swallowed inside the fetcher).
    ``Exception`` => raises it (so the caller can map auth/404 errors).
    ``dict`` => returns it verbatim.
    """

    def fetch(repo_id: str, revision: str | None, token: str | None):
        if isinstance(config, Exception):
            raise config
        return config

    return fetch
