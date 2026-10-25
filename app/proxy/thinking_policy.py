"""Thinking policy values (spec 2026-10-25 §17.3).

``client`` leaves the request's own thinking setting alone; ``off`` makes the
proxy force ``chat_template_kwargs.enable_thinking=false``. Effective policy is
``off`` if the token OR the model says ``off``.
"""

from typing import Any, Literal

ThinkingPolicy = Literal["client", "off"]
THINKING_POLICIES = ("client", "off")


def effective_policy(token_policy: str | None, model_policy: str | None) -> ThinkingPolicy:
    """``off`` if either the token or the model says ``off``; anything else is ``client``."""
    return "off" if token_policy == "off" or model_policy == "off" else "client"


def apply_thinking_off(body_json: dict[str, Any]) -> bool:
    """Force ``chat_template_kwargs.enable_thinking=false`` on a chat body.

    Merges into any existing ``chat_template_kwargs`` (overwriting a client
    ``enable_thinking``). Only bodies with a ``messages`` list are touched.
    Returns True when the body was changed (False also for a non-chat body or
    one already off; the caller decides the forced-off marking).
    """
    if not isinstance(body_json, dict) or not isinstance(body_json.get("messages"), list):
        return False
    kwargs = body_json.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    if kwargs.get("enable_thinking") is False and body_json.get("chat_template_kwargs") is kwargs:
        return False
    kwargs["enable_thinking"] = False
    body_json["chat_template_kwargs"] = kwargs
    return True
