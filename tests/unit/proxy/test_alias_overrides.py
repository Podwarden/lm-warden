"""Alias settings applied to a request (D4c).

Pure: no app, no engine, no DB. The parameter overrides, the thinking-policy
precedence (key > alias > model) and the guard's stricter mode are tested at
the boundary the proxy calls.
"""

import json

import pytest

from app.guard.proxy import guard_for
from app.proxy.alias_overrides import apply_alias_params
from app.proxy.thinking_policy import apply_thinking_off, effective_policy


class _Tok:
    def __init__(self, guard=None):
        self.id = "tok1"
        self.guard = guard


class _Alias:
    def __init__(self, guard="inherit", thinking_policy="inherit", params=None):
        self.name = "alias"
        self.guard = guard
        self.thinking_policy = thinking_policy
        self.params = params or {}


# --- parameter overrides --------------------------------------------------------


def test_default_only_when_absent():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    assert apply_alias_params(body, {"temperature": {"value": 0.5, "mode": "default"}}) is True
    assert body["temperature"] == 0.5

    body = {
        "model": "m",
        "temperature": 0.9,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert apply_alias_params(body, {"temperature": {"value": 0.5, "mode": "default"}}) is False
    assert body["temperature"] == 0.9


def test_force_replaces():
    body = {
        "model": "m",
        "temperature": 0.9,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert (
        apply_alias_params(
            body,
            {
                "temperature": {"value": 0.1, "mode": "force"},
                "max_tokens": {"value": 50, "mode": "force"},
            },
        )
        is True
    )
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 50


def test_cap_takes_min_and_fills_absent():
    body = {"max_tokens": 40000, "messages": [{"role": "user", "content": "hi"}]}
    assert apply_alias_params(body, {"max_tokens": {"value": 16384, "mode": "cap"}}) is True
    assert body["max_tokens"] == 16384

    body = {"max_tokens": 1000, "messages": [{"role": "user", "content": "hi"}]}
    assert apply_alias_params(body, {"max_tokens": {"value": 16384, "mode": "cap"}}) is False
    assert body["max_tokens"] == 1000

    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert apply_alias_params(body, {"max_tokens": {"value": 16384, "mode": "cap"}}) is True
    assert body["max_tokens"] == 16384


def test_cap_applies_to_max_completion_tokens():
    body = {
        "max_completion_tokens": 40000,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert apply_alias_params(body, {"max_tokens": {"value": 16384, "mode": "cap"}}) is True
    assert body["max_completion_tokens"] == 16384
    assert "max_tokens" not in body


def test_chat_template_kwargs_force():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert (
        apply_alias_params(
            body,
            {"chat_template_kwargs.enable_thinking": {"value": False, "mode": "force"}},
        )
        is True
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False}

    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"foo": 1, "enable_thinking": True},
    }
    assert (
        apply_alias_params(
            body,
            {"chat_template_kwargs.enable_thinking": {"value": False, "mode": "force"}},
        )
        is True
    )
    assert body["chat_template_kwargs"] == {"foo": 1, "enable_thinking": False}


def test_overrides_tolerate_bad_client_values():
    body = {"max_tokens": "lots", "messages": [{"role": "user", "content": "hi"}]}
    assert apply_alias_params(body, {"max_tokens": {"value": 16384, "mode": "cap"}}) is True
    assert body["max_tokens"] == 16384

    body = {"temperature": None, "messages": [{"role": "user", "content": "hi"}]}
    assert apply_alias_params(body, {"temperature": {"value": 0.5, "mode": "default"}}) is True
    assert body["temperature"] == 0.5


# --- thinking policy -----------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "model", "alias", "want"),
    [
        ("client", "client", None, "client"),
        ("off", "client", None, "off"),
        ("client", "off", None, "off"),
        (None, None, "inherit", "client"),
        (None, "off", "inherit", "off"),
        (None, "client", "inherit", "client"),
        (None, "client", "off", "off"),
        ("off", "client", "off", "off"),
        (None, "client", "on", "on"),
        # an alias other than "inherit" takes the model's place
        (None, "off", "on", "on"),
        (None, "off", "off", "off"),
        # the key still wins
        ("off", "off", "on", "off"),
        ("off", "client", "on", "off"),
        ("garbage", "client", "on", "on"),
    ],
)
def test_effective_policy_with_alias(token, model, alias, want):
    assert effective_policy(token, model, alias) == want


def test_alias_thinking_off_forces_off():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert effective_policy("client", "client", "off") == "off"
    assert apply_thinking_off(body) is True
    assert body["chat_template_kwargs"]["enable_thinking"] is False


def test_key_thinking_beats_alias():
    # a token set to "off" forces off even when the alias says "on"
    assert effective_policy("off", "client", "on") == "off"
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert effective_policy("off", "client", "on") == "off"
    assert apply_thinking_off(body) is True
    assert body["chat_template_kwargs"]["enable_thinking"] is False


# --- guard ----------------------------------------------------------------------


def test_alias_guard_never_weakens_key():
    # key block + alias sanitize -> block
    g = guard_for(
        None,
        _Tok(json.dumps({"override": {"mode": "block"}})),
        _Alias(guard="sanitize"),
    )
    assert g is not None and g.mode == "block"

    # key none + alias sanitize -> sanitize
    g = guard_for(None, _Tok(), _Alias(guard="sanitize"))
    assert g is not None and g.mode == "sanitize"


def test_alias_guard_inherit_changes_nothing():
    g = guard_for(None, _Tok(json.dumps({"override": {"mode": "block"}})), _Alias(guard="inherit"))
    assert g is not None and g.mode == "block"

    assert guard_for(None, _Tok(), _Alias(guard="inherit")) is None


def test_alias_guard_block_upgrades_weak_key():
    g = guard_for(None, _Tok(), _Alias(guard="block"))
    assert g is not None and g.mode == "block"

    g = guard_for(
        None,
        _Tok(json.dumps({"override": {"mode": "sanitize"}})),
        _Alias(guard="block"),
    )
    assert g is not None and g.mode == "block"
