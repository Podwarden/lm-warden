"""Pure detector for the engine's model-id flag in ``extra_args``, per backend.

The engine's model id must come from the ``served_model_name`` column: the proxy
routes and forwards by that column, so an ``extra_args`` flag that overrides the
engine's served name (vLLM ``--served-model-name``, llama.cpp ``--alias`` /
``-a``, MLX ``--model``) would serve an id the proxy never asks for. This module
only detects such a token in an argv; it is pure (no ``app.runtime`` import, no
I/O) and is applied at load time and write time, like the parallel-size flags
in :mod:`app.models.parallelism`.
"""

#: The engine model-id flag per backend, as written (the canonical long name).
MODEL_ID_FLAGS: dict[str, frozenset[str]] = {
    "vllm": frozenset({"--served-model-name"}),
    "llamacpp": frozenset({"--alias"}),
    "mlx": frozenset({"--model"}),
}

# Short aliases that each backend registers for its model-id flag. vLLM and MLX
# have none; llama.cpp registers ``-a`` for ``--alias``.
_SHORT_ALIASES: dict[str, frozenset[str]] = {
    "vllm": frozenset(),
    "llamacpp": frozenset({"-a"}),
    "mlx": frozenset(),
}

# Backends where an underscore spelling of the flag reaches it. vLLM's
# ``FlexibleArgumentParser`` rewrites ``_`` to ``-`` in a long option name (up
# to its first ``.``). llama.cpp also rewrites ``_`` in ``--`` tokens, but
# ``--alias`` has no ``-`` after the prefix, so no underscore spelling of it
# exists (``--al_ias`` becomes the unknown ``--al-ias``); MLX's plain
# ``argparse`` does not rewrite at all.
_NORMALISE_UNDERSCORES: frozenset[str] = frozenset({"vllm"})

# Backends whose parser turns a dotted long option into a JSON value for the
# option before the first ``.``: vLLM's ``FlexibleArgumentParser`` rewrites
# ``--served-model-name.x=y`` to ``--served-model-name '{"x": "y"}'``, which
# sets the served name. The name is matched without its ``.`` suffix.
_DOTTED_KEYS: frozenset[str] = frozenset({"vllm"})

# Backends that keep ``argparse``'s ``allow_abbrev=True`` and therefore accept
# unambiguous prefixes. vLLM (``FlexibleArgumentParser``) and MLX do; llama.cpp
# does not (``--alia`` is an unknown flag there).
_ALLOW_ABBREV: frozenset[str] = frozenset({"vllm", "mlx"})

# Shortest ``--`` prefix of each flag that argparse still resolves to it, per
# abbreviation-enabled backend: the shortest prefix no other option of the
# engine shares. vLLM (v0.20.0) has ``--seed`` beside ``--served-model-name``
# and no other ``--ser*`` option, so ``--ser`` / ``--serv`` / ``--serve`` all
# resolve to ``--served-model-name`` and ``--se`` is ambiguous (argparse
# errors). mlx_lm.server has ``--min-p`` / ``--max-tokens`` beside ``--model``,
# so ``--mo`` is the floor and ``--m`` is ambiguous. ``--draft-model`` /
# ``--model-extra`` are not prefixes, so they never match. A floor set lower
# than the engine's real one only over-refuses (the engine would reject the
# ambiguous prefix anyway); one set higher lets a prefix through.
_ABBREV_FLOORS: dict[str, dict[str, str]] = {
    "vllm": {"--served-model-name": "--ser"},
    "mlx": {"--model": "--mo"},
}


def _is_model_id_name(backend: str, name: str) -> bool:
    """True if a normalised, abbrev-resolved long name is this backend's flag."""
    if backend in _ALLOW_ABBREV:
        for full, floor in _ABBREV_FLOORS[backend].items():
            if len(name) >= len(floor) and full.startswith(name):
                return True
    return name in MODEL_ID_FLAGS[backend]


def _is_model_id_token(backend: str, tok: str) -> bool:
    """True if one argv token is the backend's model-id flag, in any accepted
    spelling (canonical long name, ``=value`` form, underscore spelling, vLLM
    dotted key, short alias, or argparse abbreviation down to the floor)."""
    name = tok.partition("=")[0]
    if name in _SHORT_ALIASES[backend]:
        return True
    if name.startswith("--"):
        if backend in _DOTTED_KEYS:
            name = name.partition(".")[0]
        if backend in _NORMALISE_UNDERSCORES:
            name = name.replace("_", "-")
        return _is_model_id_name(backend, name)
    return False


def model_id_flag_in(backend: str, extra_args: list[str]) -> str | None:
    """The first ``extra_args`` token that sets the backend's model id, as written.

    Returns ``None`` when no token does. An unknown ``backend`` raises
    ``ValueError`` naming the accepted backend names.
    """
    if backend not in MODEL_ID_FLAGS:
        accepted = ", ".join(sorted(MODEL_ID_FLAGS))
        raise ValueError(f"unknown backend {backend!r}: expected one of {accepted}")
    for tok in extra_args:
        if _is_model_id_token(backend, tok):
            return tok
    return None


def model_id_refusal(backend: str, token: str) -> str:
    """A refusal message naming the offending token and the ``served_model_name``
    column (for MLX, the local snapshot path the engine is pointed at)."""
    if backend not in MODEL_ID_FLAGS:
        accepted = ", ".join(sorted(MODEL_ID_FLAGS))
        raise ValueError(f"unknown backend {backend!r}: expected one of {accepted}")
    if backend == "mlx":
        source = "the local snapshot path (the served_model_name column)"
    else:
        source = "the served_model_name column"
    return (
        f"extra_args must not set {token} (the engine's model id): the engine's "
        f"model id comes from {source}, which the proxy routes and forwards by"
    )
