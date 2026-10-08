"""Resolving a model's own context ceiling (design §6).

The sweep needs to know how far up it is even allowed to look. Without a
ceiling it can only ask *"is the current setting too ambitious?"* — never
*"can we go higher?"*, which is half of the question this feature exists to
answer.

**A running engine is usually not a valid source, and this is the trap.** What
it reports is what it was *configured* with. Measured on pw-bonus, 2026-09-03,
llama-server b10731 serving qwen3.8-27b started with ``--ctx-size 131072``::

    GET /props  ->  default_generation_settings.n_ctx = 131072

That is our own flag echoed back. Treating it as a ceiling makes the sweep
circular: it would cap the ladder at the setting we are trying to improve on,
confirm it, and report ``limited_by: model_ceiling`` as though the model could
go no further — a confident answer derived entirely from the input.

There is exactly one case where the engine *is* informative: when we imposed
no limit at all, both engines fall back to the model's own maximum, so the
reported value is the trained ceiling rather than an echo. A value strictly
greater than what we configured is also real evidence, since it cannot have
come from our flag.

The portable source is the model's own config — ``max_position_embeddings``,
which ``app/models/discovery.py`` already keeps. GGUF repos frequently ship no
``config.json`` (``unsloth/gpt-oss-20b-GGUF`` lists ``notebook.ipynb``,
``params`` and ``template``, and no config), so ``None`` is a normal outcome
and not an error. The sweep degrades to the narrower question rather than
inventing a number.
"""

from __future__ import annotations


def _positive_int(value: object) -> int | None:
    """Reject anything that would make the candidate ladder degenerate."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def ceiling_from_engine(payload: object, *, configured: int | None) -> int | None:
    """The model's ceiling per a live engine, when that is not circular.

    ``configured`` is the row's ``max_model_len``. When it is set and the
    engine merely repeats it, the engine has told us nothing we did not already
    know, so this returns ``None`` rather than laundering our own input into
    evidence.

    Only vLLM's OpenAI-shaped ``/v1/models`` carries the field at all;
    llama.cpp's is Ollama-shaped (``{"models": [{"name": ...}]}``) and has no
    context anywhere in it.
    """
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        reported = _positive_int(entry.get("max_model_len"))
        if reported is None:
            continue
        if configured is None:
            # Nothing was imposed, so the engine fell back to the model's own
            # maximum. That fallback is the trained ceiling.
            return reported
        if reported > configured:
            # Cannot be an echo of our flag, so it is real evidence.
            return reported
        return None
    return None


def ceiling_from_config(config: object) -> int | None:
    """The model's ceiling per its own ``config.json``.

    The portable source, and the only one that works for a model the engine
    has capped. Absent for most GGUF-only repos, which is expected.
    """
    if not isinstance(config, dict):
        return None
    return _positive_int(config.get("max_position_embeddings"))


def resolve_ceiling(
    *,
    config: object = None,
    engine_models_payload: object = None,
    configured: int | None = None,
) -> int | None:
    """Best available ceiling, config first.

    Config wins because it is a property of the model rather than of how we
    happen to have launched it, so it stays true across a sweep that changes
    ``max_model_len`` at every step — which is precisely when the engine's own
    report is least trustworthy.
    """
    return ceiling_from_config(config) or ceiling_from_engine(
        engine_models_payload, configured=configured
    )


async def resolve_for_model(app_state, model) -> int | None:
    """Best-effort ceiling for a loaded model, for the sweep to climb toward.

    Deliberately best-effort. A stress run costs 15 minutes to 4 hours, so one
    config lookup at the start is free by comparison — but it can fail (no
    network, a gated repo, a GGUF-only repo with no ``config.json`` at all),
    and a failure here must never stop the run. Returning ``None`` costs the
    headroom question and keeps the "is the current setting too ambitious?"
    one, which is the more urgent question for a model that is crashing.

    The in-process discovery cache is tried first because it is free; only a
    cold cache pays for a round trip.
    """
    repo = getattr(model, "hf_repo", None)
    if not repo:
        return None
    revision = getattr(model, "hf_revision", None) or "main"

    cache = getattr(app_state, "discovery_cache", None)
    if cache is not None:
        cached = getattr(cache, "peek", lambda _k: None)((repo, revision))
        if isinstance(cached, dict):
            found = ceiling_from_config(cached.get("config"))
            if found:
                return found

    try:
        from app.models.discovery import discover_repo_files, load_hf_token

        settings = getattr(app_state, "settings", None)
        token = await load_hf_token(settings) if settings is not None else None
        result = await discover_repo_files(repo, revision, token)
        return ceiling_from_config(
            getattr(result, "config", None)
            or (result.to_dict().get("config") if hasattr(result, "to_dict") else None)
        )
    except Exception:
        # Network, auth, a repo with no config.json -- all normal, none fatal.
        return None
