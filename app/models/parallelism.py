"""Parallel-layout helpers shared by the spec, the argv builder and the proxy (#286).

The layout (replicas x GPUs per replica) lives in the ``tensor_parallel_size``
and ``data_parallel_size`` columns. ``extra_args`` may not carry it, and a legacy
row that did is promoted once at boot (``app.runtime.boot_reconcile``).
"""

PARALLEL_FLAGS: frozenset[str] = frozenset(
    {"--tensor-parallel-size", "--data-parallel-size", "--pipeline-parallel-size"}
)

PARALLEL_FLAGS_REFUSAL = (
    "extra_args must not set --tensor-parallel-size / --data-parallel-size / "
    "--pipeline-parallel-size (or their -tp / -dp / -pp aliases): use the "
    "tensor_parallel_size and data_parallel_size fields"
)

#: vLLM's default ``--max-num-seqs`` when the flag is absent.
DEFAULT_MAX_NUM_SEQS = 256


def _parse_positive(raw: str) -> int | None:
    try:
        v = int(raw)
    except ValueError:
        return None
    return v if v > 0 else None


#: Short aliases vLLM registers for the parallel flags (``-tp`` / ``-dp`` / ``-pp``).
_SHORT_ALIASES: dict[str, str] = {
    "-tp": "--tensor-parallel-size",
    "-dp": "--data-parallel-size",
    "-pp": "--pipeline-parallel-size",
}


#: Shortest ``--`` prefix of each parallel flag that is treated as that flag.
#: vLLM's ``FlexibleArgumentParser`` keeps argparse's ``allow_abbrev=True``, so
#: ``--tensor-parallel 2`` is accepted as ``--tensor-parallel-size 2``. The
#: floor is where the prefix stops matching any other vLLM option family
#: (``--tensor-parallel`` / ``--pipeline-parallel`` have only the ``-size``
#: member; ``--data-parallel-s`` is the first prefix that cannot be
#: ``--data-parallel-rank`` / ``-backend`` / ``-address`` / ``-rpc-port``).
#: A prefix that is ambiguous among the ``--data-parallel-s*`` family
#: (``-size-local``, ``-start-rank``) is refused by argparse at boot anyway;
#: treating it as the size flag only makes our refusal stricter, never looser.
#: Shorter prefixes (``--tensor``, ``--data``) are left alone: they could be
#: other options, and flagging them would reject legitimate arguments.
_ABBREV_FLOORS: dict[str, str] = {
    "--tensor-parallel-size": "--tensor-parallel",
    "--data-parallel-size": "--data-parallel-s",
    "--pipeline-parallel-size": "--pipeline-parallel",
}


def _resolve_abbrev(name: str) -> str:
    for full, floor in _ABBREV_FLOORS.items():
        if len(name) >= len(floor) and full.startswith(name):
            return full
    return name


def _split(tok: str) -> tuple[str, bool, str]:
    """``(canonical_name, had_equals, value)`` for one argv token.

    vLLM's ``FlexibleArgumentParser`` rewrites ``_`` to ``-`` in long option
    names and accepts the short aliases above, so ``--tensor_parallel_size``
    and ``-tp`` are the same flag as ``--tensor-parallel-size``, and so is an
    unambiguous argparse abbreviation such as ``--tensor-parallel``.
    """
    name, eq, val = tok.partition("=")
    if name.startswith("--"):
        name = _resolve_abbrev(name.replace("_", "-"))
    else:
        name = _SHORT_ALIASES.get(name, name)
    return name, bool(eq), val


def flag_values(extra_args: list[str], flags: frozenset[str]) -> dict[str, int | None]:
    """flag -> parsed positive int (None when present but unparsable).

    ``flags`` holds canonical long names. Handles ``--flag v``, ``--flag=v``,
    ``--flag_name`` and the ``-tp`` / ``-dp`` / ``-pp`` aliases; the last
    occurrence wins, like argparse. Absent flags are absent from the result.
    """
    out: dict[str, int | None] = {}
    i = 0
    n = len(extra_args)
    while i < n:
        name, eq, val = _split(extra_args[i])
        if name in flags:
            if eq:
                out[name] = _parse_positive(val)
                i += 1
            else:
                out[name] = _parse_positive(extra_args[i + 1]) if i + 1 < n else None
                i += 2
            continue
        i += 1
    return out


def parallel_flag_values(extra_args: list[str]) -> dict[str, int | None]:
    """The parallel-size flags present in ``extra_args`` and their values."""
    return flag_values(extra_args, PARALLEL_FLAGS)


def strip_parallel_flags(extra_args: list[str]) -> list[str]:
    """``extra_args`` without any parallel-size flag or its value, order kept."""
    out: list[str] = []
    i = 0
    n = len(extra_args)
    while i < n:
        name, eq, _ = _split(extra_args[i])
        if name in PARALLEL_FLAGS:
            i += 1 if eq else 2
            continue
        out.append(extra_args[i])
        i += 1
    return out


def spill_threshold_for(extra_args: list[str], dp_spill_threshold: int | None) -> tuple[int, str]:
    """``(threshold, "setting" | "auto")`` per #286 decision 8."""
    if dp_spill_threshold is not None:
        return dp_spill_threshold, "setting"
    seqs = flag_values(extra_args, frozenset({"--max-num-seqs"})).get("--max-num-seqs")
    return max(1, (seqs or DEFAULT_MAX_NUM_SEQS) // 4), "auto"
