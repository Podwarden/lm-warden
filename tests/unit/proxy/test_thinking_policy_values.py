"""Keys and models store only client/off; "on" is an alias-only setting (D4c)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.proxy.thinking_policy import THINKING_POLICIES, ThinkingPolicy, effective_policy


def test_keys_and_models_cannot_store_on():
    assert THINKING_POLICIES == ("client", "off")
    with pytest.raises(ValidationError):
        TypeAdapter(ThinkingPolicy).validate_python("on")


def test_alias_on_still_effective():
    assert effective_policy("client", "client", "on") == "on"
    assert effective_policy("off", "client", "on") == "off"
