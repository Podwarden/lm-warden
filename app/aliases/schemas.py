"""Wire shapes for the model-alias admin API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.schemas import SLUG_RE

# -- the D4c request-parameter table -----------------------------------------
#
# mode "cap" is numeric-only (min(client, value)); reasoning_effort and the
# chat_template_kwargs.<key> entries take default/force; anything not listed
# here is refused at write time. The 422 message names the offending key.

GUARD_VALUES = ("inherit", "sanitize", "block")
THINKING_POLICY_VALUES = ("inherit", "client", "on", "off")

_NUMERIC_PARAMS: dict[str, tuple[str, float | None, float | None]] = {
    # name: ("int" | "float", minimum, maximum)
    "max_tokens": ("int", 1, None),
    "temperature": ("float", 0.0, 2.0),
    "top_p": ("float", 0.0, 1.0),
    "top_k": ("int", -1, None),
    "min_p": ("float", 0.0, 1.0),
    "presence_penalty": ("float", -2.0, 2.0),
    "frequency_penalty": ("float", -2.0, 2.0),
    "repetition_penalty": ("float", 0.0, 2.0),
    "seed": ("int", None, None),
}
_REASONING_EFFORTS = ("low", "medium", "high")


def _is_json_scalar(v: Any) -> bool:
    return v is None or isinstance(v, str | bool | int | float)


def _check_param(key: str, value: Any, mode: str) -> None:
    if key.startswith("chat_template_kwargs."):
        suffix = key[len("chat_template_kwargs.") :]
        if not suffix:
            raise ValueError(f"'{key}': chat_template_kwargs.<key> needs a non-empty key")
        if not _is_json_scalar(value):
            raise ValueError(
                f"'{key}': value must be a JSON scalar (string, number, boolean or null)"
            )
        if mode == "cap":
            raise ValueError(f"'{key}': cap is only for numeric parameters")
        return
    if key == "reasoning_effort":
        if value not in _REASONING_EFFORTS:
            raise ValueError(f"'reasoning_effort': must be one of {'|'.join(_REASONING_EFFORTS)}")
        if mode == "cap":
            raise ValueError("'reasoning_effort': cap is only for numeric parameters")
        return
    if key not in _NUMERIC_PARAMS:
        raise ValueError(f"'{key}': unknown parameter")
    if key == "seed" and mode == "cap":
        raise ValueError("'seed': cap makes no sense for a seed; use default or force")
    kind, lo, hi = _NUMERIC_PARAMS[key]
    if isinstance(value, bool):
        raise ValueError(
            f"'{key}': must be {'an int' if kind == 'int' else 'a number'} (not a boolean)"
        )
    if kind == "int" and not isinstance(value, int):
        raise ValueError(f"'{key}': must be an int")
    if kind == "float" and not isinstance(value, int | float):
        raise ValueError(f"'{key}': must be a number")
    if lo is not None and value < lo:
        raise ValueError(f"'{key}': must be >= {lo:g}")
    if hi is not None and value > hi:
        raise ValueError(f"'{key}': must be <= {hi:g}")


class AliasParam(BaseModel):
    """One entry in the params map: a request parameter's value and mode."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    mode: Literal["default", "force", "cap"] = "default"


class AliasIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100, pattern=SLUG_RE)
    targets: list[str] = Field(default_factory=list, max_length=8)
    guard: Literal["inherit", "sanitize", "block"] = "inherit"
    thinking_policy: Literal["inherit", "client", "on", "off"] = "inherit"
    params: dict[str, AliasParam] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("targets")
    @classmethod
    def _distinct_targets(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("targets must be distinct")
        return v

    @field_validator("params")
    @classmethod
    def _params_known(cls, v: dict[str, AliasParam]) -> dict[str, AliasParam]:
        for key, p in v.items():
            _check_param(key, p.value, p.mode)
        return v


class AliasPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100, pattern=SLUG_RE)
    targets: list[str] | None = Field(default=None, max_length=8)
    guard: Literal["inherit", "sanitize", "block"] | None = None
    thinking_policy: Literal["inherit", "client", "on", "off"] | None = None
    params: dict[str, AliasParam] | None = None
    enabled: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_nulls(cls, data: object) -> object:
        if isinstance(data, dict):
            for k, v in data.items():
                if v is None:
                    raise ValueError(f"{k}: null is not allowed")
        return data

    @field_validator("targets")
    @classmethod
    def _distinct_targets(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) != len(set(v)):
            raise ValueError("targets must be distinct")
        return v

    @field_validator("params")
    @classmethod
    def _params_known(cls, v: dict[str, AliasParam] | None) -> dict[str, AliasParam] | None:
        if v is not None:
            for key, p in v.items():
                _check_param(key, p.value, p.mode)
        return v


class AliasTargetOut(BaseModel):
    id: str
    served_model_name: str | None
    status: str | None


class AliasOut(BaseModel):
    id: str
    name: str
    targets: list[AliasTargetOut]
    guard: str
    thinking_policy: str
    params: dict[str, Any]
    enabled: bool
    created_at: str
    updated_at: str


class AliasesOut(BaseModel):
    aliases: list[AliasOut]
