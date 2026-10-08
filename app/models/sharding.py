"""Shared helpers for HuggingFace sharded-weight filenames.

Sharded weights on the Hub use the ``<prefix>-NNNNN-of-NNNNN.<ext>``
convention — both ``transformers`` (safetensors) and ``llama.cpp`` (GGUF)
follow it. Picking any single member should usually be lifted to the
*whole* shard set: fit-preview classifies against aggregate VRAM, and
``snapshot_download`` must fetch every sibling (#111) or vLLM refuses to
load because ``model.safetensors.index.json``'s weight_map references
shards that aren't on disk.

Single-file weights (single ``.safetensors``, single ``.gguf``,
``pytorch_model.bin``) do NOT match and fall through to the regular
single-file path.

Used by both:
- ``app/models/routes_api.py`` — fit-preview aggregate VRAM math
- ``app/models/pull_task.py`` — ``allow_patterns`` expansion for HF pull
"""

import re

# Sharded weights filenames look like ``model-00001-of-00004.safetensors`` or
# ``llama-3.3-70B-Q5_K_M-00001-of-00003.gguf`` — both common conventions on
# the Hub (llama.cpp uses the same suffix pattern as transformers for GGUF
# splits). Originally lived in ``routes_api.py`` for the fit-preview math
# (#85, extended to GGUF in #88); lifted to this shared module in #111 so
# the pull path can reuse the same shard-set definition.
SHARD_NAME_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.(?P<ext>safetensors|gguf)$"
)


def shard_glob_for(filename: str) -> str | None:
    """If ``filename`` is a sharded safetensors or GGUF file, return the glob
    matching the whole shard set; otherwise return None.

    For ``"model-00002-of-00008.safetensors"`` returns
    ``"model-*-of-00008.safetensors"``; for
    ``"llama-Q5_K_M-00001-of-00003.gguf"`` returns
    ``"llama-Q5_K_M-*-of-00003.gguf"``. We group every shard in the same
    "of-N + extension" family — there can legitimately be multiple shard
    sets in one repo (e.g. Q4_K_M and Q5_K_M GGUF families side-by-side, or
    bf16 + fp8 safetensors in the same repo), so we do NOT glob across
    "of-N" boundaries OR across extensions.
    """
    m = SHARD_NAME_RE.match(filename)
    if not m:
        return None
    return f"{m.group('prefix')}-*-of-{m.group('total')}.{m.group('ext')}"
