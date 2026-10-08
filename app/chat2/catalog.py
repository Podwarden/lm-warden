from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.db.database import open_db
from app.db.repos.models import ModelRepo

IMAGE_TOKENS_ESTIMATE = 1000


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display: str
    context_window: int | None
    supports_tools: bool
    supports_vision: bool
    supports_reasoning: bool
    image_tokens_estimate: int
    pricing: None
    # The `reasoning_effort` values this model's chat template accepts, in the
    # template's own vocabulary (Qwen3.8: `xhigh`, `medium`, `low`), read from
    # the on-disk template. Empty when the template does not take the kwarg or
    # its vocabulary could not be established -- nothing is offered and nothing
    # is sent, because a template that validates the value `raise_exception`s
    # on one it does not know rather than ignoring it (#241).
    reasoning_efforts: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display": self.display,
            "context_window": self.context_window,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "supports_reasoning": self.supports_reasoning,
            "image_tokens_estimate": self.image_tokens_estimate,
            "pricing": self.pricing,
            "reasoning_efforts": list(self.reasoning_efforts),
        }


@lru_cache(maxsize=128)
def _parse_config(path: str, mtime_ns: int, size: int) -> dict[str, Any] | None:
    """Parse one `config.json`, keyed by the file's identity.

    `mtime_ns` and `size` are cache-key material only — an edited config gets a
    new key and is re-read. The models list polls every 60s per model and these
    files never change under a running warden, so re-parsing on every hit is
    pure waste.

    The returned dict is SHARED between callers: treat it as read-only.
    """
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_hf_config(hf_cache_dir: Path, hf_repo: str) -> dict[str, Any] | None:
    """The model's on-disk HF `config.json`, or None if there isn't a readable one.

    The result is cached and shared — callers must not mutate it.
    """
    folder = hf_cache_dir / f"models--{hf_repo.replace('/', '--')}" / "snapshots"
    if not folder.is_dir():
        return None
    for snap in sorted(folder.iterdir(), reverse=True):
        cfg = snap / "config.json"
        if cfg.is_file():
            try:
                st = cfg.stat()
            except OSError:
                return None
            return _parse_config(str(cfg), st.st_mtime_ns, st.st_size)
    return None


# `{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}` -- the
# membership test a validating template runs right before `raise_exception`.
# The tuple IS the vocabulary, so it is read rather than assumed.
_EFFORT_TUPLE = re.compile(r"reasoning_effort\s+not\s+in\s*\(([^)]*)\)")
_QUOTED = re.compile(r"""['"]([A-Za-z0-9_-]+)['"]""")


def reasoning_efforts_from_template(template: str | None) -> tuple[str, ...]:
    """The `reasoning_effort` values a chat template accepts, or `()`.

    Only a template that both reads the kwarg AND states its accepted set
    yields anything. A template that never mentions `reasoning_effort` ignores
    the kwarg (Jinja drops unknown variables), so there is nothing to offer;
    one that mentions it without a membership test has a vocabulary we cannot
    enumerate, and guessing would hand the model a value it may raise on.
    """
    if not template or "reasoning_effort" not in template:
        return ()
    m = _EFFORT_TUPLE.search(template)
    if m is None:
        return ()
    return tuple(_QUOTED.findall(m.group(1)))


def _window_from_config(config: dict[str, Any] | None) -> int | None:
    v = (config or {}).get("max_position_embeddings")
    return int(v) if isinstance(v, int) and v > 0 else None


def read_max_position_embeddings(hf_cache_dir: Path, hf_repo: str) -> int | None:
    return _window_from_config(read_hf_config(hf_cache_dir, hf_repo))


def config_suggests_vision(config: dict[str, Any] | None) -> bool:
    """Whether an HF `config.json` describes a multimodal model.

    Two signals, both cheap and both present on every VLM we have shipped:
    a `vision_config` sub-config, and the `...ForConditionalGeneration`
    architecture suffix HF uses for the image-in/text-out wrappers. An
    unreadable or absent config is not evidence of anything, so it reads False.
    """
    if not config:
        return False
    if "vision_config" in config:
        return True
    archs = config.get("architectures")
    return isinstance(archs, list) and any(
        isinstance(a, str) and a.endswith("ForConditionalGeneration") for a in archs
    )


def read_hf_chat_template(hf_cache_dir: Path, hf_repo: str) -> str | None:
    """The model's chat template source, or None if there isn't a readable one.

    Newer HF repos ship the template as a standalone `chat_template.jinja`
    file in the repo root (checked first); older ones embed it as the
    `chat_template` key inside `tokenizer_config.json`. Mirrors
    `read_hf_config`'s snapshot-folder walk so `hf_config_repo` is honoured
    the same way. Never raises -- this feeds `list_catalog`, which is on the
    chat's hot path, so a missing/unreadable file or a repo not in the cache
    just means "no signal".
    """
    folder = hf_cache_dir / f"models--{hf_repo.replace('/', '--')}" / "snapshots"
    try:
        if not folder.is_dir():
            return None
        snapshots = sorted(folder.iterdir(), reverse=True)
    except OSError:
        return None
    for snap in snapshots:
        jinja = snap / "chat_template.jinja"
        if jinja.is_file():
            try:
                return jinja.read_text()
            except OSError:
                return None
        tok_cfg = snap / "tokenizer_config.json"
        if tok_cfg.is_file():
            try:
                st = tok_cfg.stat()
            except OSError:
                return None
            data = _parse_config(str(tok_cfg), st.st_mtime_ns, st.st_size)
            template = (data or {}).get("chat_template")
            if isinstance(template, str):
                return template
    return None


def chat_template_suggests_reasoning(template: str | None) -> bool:
    """Whether a chat template branches on `enable_thinking`.

    A template that gates its behaviour on this Jinja variable belongs to a
    reasoning-capable model family (Qwen3 and others). An absent or
    unreadable template is not evidence of anything, so it reads False.
    """
    return template is not None and "enable_thinking" in template


def _info(settings: Any, r: Any) -> ModelInfo:  # noqa: ANN401
    # `supports_vision` and `supports_reasoning` are TRI-state in the DB: 1/0
    # is an operator's explicit answer and always wins; NULL means "nobody has
    # said", and is where an on-disk signal gets a vote. Collapsing NULL into
    # 0 is what silently turned every pasted image into "[image omitted]" on a
    # model that was perfectly capable of reading it (#106), and it is the
    # same failure shape that hid the "Enable thinking" toggle on every
    # reasoning model until an operator found and set the flag by hand
    # (#239). `supports_tools` stays purely manual -- there is no equally
    # cheap config signal for it, and guessing wrong there breaks a turn
    # rather than degrading it.
    vision: int | None = getattr(r, "supports_vision", None)
    reasoning: int | None = getattr(r, "supports_reasoning", None)
    hf_repo = getattr(r, "hf_config_repo", None) or r.hf_repo
    config: dict[str, Any] | None = None
    if not r.max_model_len or vision is None:
        # #106: some models ship weights and config under different repos, and
        # `hf_config_repo` is where the config actually is when it is set.
        config = read_hf_config(settings.hf_cache_dir, hf_repo)
    template: str | None = None
    if reasoning is None or reasoning:
        # Read when the model may reason: it decides `supports_reasoning` for a
        # NULL row, and it is where the `reasoning_effort` vocabulary lives
        # (#241). An operator's explicit "no" skips the read -- there is
        # nothing to offer such a model.
        template = read_hf_chat_template(settings.hf_cache_dir, hf_repo)
    window = r.max_model_len or _window_from_config(config)
    return ModelInfo(
        id=r.served_model_name,
        display=r.served_model_name,
        context_window=window,
        supports_tools=bool(getattr(r, "supports_tools", 0)),
        supports_vision=bool(vision) if vision is not None else config_suggests_vision(config),
        supports_reasoning=(
            bool(reasoning) if reasoning is not None else chat_template_suggests_reasoning(template)
        ),
        image_tokens_estimate=IMAGE_TOKENS_ESTIMATE,
        pricing=None,
        reasoning_efforts=reasoning_efforts_from_template(template),
    )


async def list_catalog(settings: Any) -> list[ModelInfo]:  # noqa: ANN401
    async with open_db(settings.db_path) as db:
        rows = await ModelRepo(db).list_all()
    return [_info(settings, r) for r in rows if r.status == "loaded"]


async def get_model_info(settings: Any, served_name: str) -> ModelInfo | None:  # noqa: ANN401
    async with open_db(settings.db_path) as db:
        row = await ModelRepo(db).get_by_served_name(served_name)
    if row is None or row.status != "loaded":
        return None
    return _info(settings, row)
