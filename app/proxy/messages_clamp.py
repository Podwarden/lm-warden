"""Clamp a translated request's ``max_tokens`` to the local model's context.

Shared by the router's local leg and the plain ``/v1/messages`` path.
"""

import logging
from typing import Any

from fastapi import Request

from app.proxy.messages_translate import count_text

logger = logging.getLogger(__name__)


def clamp_margin(prompt_tokens: int, *, estimated: bool) -> int:
    """Tokens kept free below the context on top of the counted prompt.

    The warden's prompt count leaves out the chat template's framing and the
    template's rendering of tools, so it undercounts what the engine sees: a
    flat 256 plus 5% of the prompt covers that. When no real tokenizer is
    available the count is itself a characters/4 guess, so the prompt share
    grows to 25%.
    """
    return 256 + prompt_tokens // (4 if estimated else 20)


async def clamp_max_tokens(
    request: Request, resolved: tuple[Any, ...], oai: dict[str, Any]
) -> None:
    """Lower ``oai['max_tokens']`` so prompt + output fits the target's context.

    Claude Code sends max_tokens of 32000+ on top of a ~20k-token prompt; a
    local model with a smaller window rejects that outright (400) although the
    answer would fit. Unknown window, no max_tokens, or a prompt that alone
    does not fit: untouched (the engine's 400 takes the usual fallback road).
    """
    model = resolved[0]
    variant = resolved[3] if len(resolved) > 3 else None
    descriptor = getattr(variant, "descriptor", None)
    window = (
        descriptor["max_model_len"]
        if isinstance(descriptor, dict) and "max_model_len" in descriptor
        else getattr(model, "max_model_len", None)
    )
    requested = oai.get("max_tokens")
    if not isinstance(window, int) or window <= 0 or not isinstance(requested, int):
        return
    tokenizers = request.app.state.tokenizers
    prompt_tokens = await tokenizers.count(
        model.hf_repo, count_text(oai), fallback_repo=getattr(model, "tokenizer_repo", None)
    )
    try:
        estimated = model.hf_repo in tokenizers.estimating()
    except Exception:  # noqa: BLE001 - a missing flag must not fail the leg
        estimated = False
    room = window - prompt_tokens - clamp_margin(prompt_tokens, estimated=estimated)
    if room < 1 or room >= requested:
        return
    oai["max_tokens"] = room
    logger.debug("router: max_tokens %d -> %d (window %d)", requested, room, window)
