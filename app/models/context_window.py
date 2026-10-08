"""The context window a client may actually use, for one loaded model.

Kept out of ``app/chat2/catalog.py`` because two very different surfaces need
the same answer -- the chat catalog and the OpenAI-compatible ``/v1/models`` --
and a client that guesses this number wrong gets a refusal it cannot interpret.

The resolution order is the engine's own, not the row's:

1. **A live override.** ``supervisor.get_overrides()`` tracks the configuration
   each running engine was actually launched with, which is not necessarily
   what the row says: a stress sweep or a reload-with-config leaves the row
   untouched. ``"max_model_len" in ov`` rather than ``ov.get(...)`` because an
   override may set it to ``None`` deliberately, meaning "drop the flag and let
   the engine derive it" -- which is a different instruction from "no override".
2. **The row.** What the operator typed at load time.
3. **The model's own ceiling.** Both engines derive the window from
   ``max_position_embeddings`` when no flag is passed, so for a NULL row that
   config value *is* the effective window, not a guess about it.

Returns ``None`` only when every source is silent, so callers can omit the
field rather than publish a zero or a default that would be believed.
"""

from typing import Any

# `read_max_position_embeddings` reads the on-disk HF config and caches on
# (path, mtime, size), so step 3 costs a stat() rather than a parse -- this is
# what lets a data-plane endpoint afford it per request.
from app.chat2.catalog import read_max_position_embeddings


def effective_context_window(
    settings: Any,  # noqa: ANN401 -- the app settings object, as everywhere else
    supervisor: Any,  # noqa: ANN401
    row: Any,  # noqa: ANN401
) -> int | None:
    """``row``'s usable context window in tokens, or None if nothing states one."""
    ov: dict[str, Any] = {}
    if supervisor is not None:
        try:
            ov = supervisor.get_overrides(row.id) or {}
        except Exception:  # noqa: BLE001 -- advisory metadata never breaks /v1
            ov = {}

    if "max_model_len" in ov:
        window = ov["max_model_len"]
    else:
        window = getattr(row, "max_model_len", None)
    if window:
        return int(window)

    # #106: some models ship weights and config under different repos, and
    # `hf_config_repo` is where the config actually is when it is set.
    hf_repo = getattr(row, "hf_config_repo", None) or row.hf_repo
    try:
        return read_max_position_embeddings(settings.hf_cache_dir, hf_repo)
    except Exception:  # noqa: BLE001 -- an unreadable config is not an error here
        return None
