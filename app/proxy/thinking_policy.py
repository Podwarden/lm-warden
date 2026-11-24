"""Thinking policy values (spec 2026-10-25 §17.3).

``client`` leaves the request's own thinking setting alone; ``off`` makes the
proxy force ``chat_template_kwargs.enable_thinking=false``; ``on`` forces it
``true``. Effective policy is ``off`` if the token OR the model says ``off``.

An alias (D4c) can also set the policy: an alias policy other than ``inherit``
takes the model's place, and the key still wins.
"""

from typing import Any, Literal

# What a key or a model may store: "on" exists only on aliases (D4c), so keys and models
# keep their two values and never accept a setting the policy would ignore.
ThinkingPolicy = Literal["client", "off"]
THINKING_POLICIES = ("client", "off")
# The effective policy for one request: an alias can force thinking on.
EffectivePolicy = Literal["client", "on", "off"]


def effective_policy(
    token_policy: str | None, model_policy: str | None, alias_policy: str | None = None
) -> EffectivePolicy:
    """The effective thinking policy.

    ``off`` if the token says ``off``; otherwise an alias policy other than
    ``inherit`` takes the model's place; anything else is ``off`` when the model
    says ``off``, else ``client``.
    """
    if token_policy == "off":
        return "off"
    if alias_policy not in (None, "inherit"):
        if alias_policy == "off":
            return "off"
        if alias_policy == "on":
            return "on"
        return "client"
    return "off" if model_policy == "off" else "client"


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


def apply_thinking_on(body_json: dict[str, Any]) -> bool:
    """Force ``chat_template_kwargs.enable_thinking=true`` on a chat body.

    Merges into any existing ``chat_template_kwargs`` (overwriting a client
    ``enable_thinking``). Only bodies with a ``messages`` list are touched.
    Returns True when the body was changed.
    """
    if not isinstance(body_json, dict) or not isinstance(body_json.get("messages"), list):
        return False
    kwargs = body_json.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    if kwargs.get("enable_thinking") is True and body_json.get("chat_template_kwargs") is kwargs:
        return False
    kwargs["enable_thinking"] = True
    body_json["chat_template_kwargs"] = kwargs
    return True
