"""HF tokenizer cache used by the proxy for prompt/completion accounting.

The cache grows by one entry per distinct (hf_repo, trust_remote_code) tuple
the proxy has been asked to count tokens for. Each entry holds a fully
loaded ``AutoTokenizer`` (vocab + merges + special-tokens config), which
runs anywhere from ~1 MiB (small word-piece tokenizers) to ~20+ MiB
(sentencepiece + BPE merges for some Qwen/Llama variants) of process RSS.

S7 (#124) — code-review finding #5: the cache was previously **never**
flushed on model unload. A long-lived warden that loads-and-unloads a
rotating set of models (operator iterating on tuning, or a CI-style
"smoke every catalog entry" workflow) would accumulate one entry per
distinct repo ever seen — eventually OOMing the process. The fix is
``evict(hf_repo)``, called from ``unload`` in the supervisor's unload
hook (see app/models/routes_api.py::unload_model). Same hf_repo loaded
again later transparently re-fetches.

The cache is per-(hf_repo, trust_remote_code) on purpose: the same repo
with and without ``trust_remote_code`` produces different tokenizer
classes (custom code is loaded only with the flag set), so they must
not share an entry. Evict by repo only — both variants for the same
hf_repo go at the same time.
"""

import asyncio
import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Average characters per token across the tokenizers this product serves. A
# crude constant, and deliberately so: this number is only ever reached when the
# exact count is unavailable, and pretending to more precision than
# "approximately four" would disguise an estimate as a measurement.
_CHARS_PER_TOKEN = 4


class TokenizerCache:
    """Lazy-loaded HF tokenizer cache keyed by (hf_repo, trust_remote_code). Used for accounting only."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, bool], object] = {}
        # Model repos whose token counts are character ESTIMATES because no
        # tokenizer could be loaded. Keyed by the model's own hf_repo, which is
        # what an operator looking for the row to fix would search for -- not by
        # whichever repo we happened to try.
        self._estimating: set[str] = set()
        self._lock = asyncio.Lock()

    def estimating(self) -> frozenset[str]:
        """Repos currently being character-estimated rather than tokenized.

        Exposed so the health/status surface can say so. Approximate accounting
        feeds the per-token usage rollups, and an operator is entitled to know
        their billing is an estimate.
        """
        return frozenset(self._estimating)

    async def get(self, hf_repo: str, *, trust_remote_code: bool):
        key = (hf_repo, trust_remote_code)
        async with self._lock:
            if key not in self._cache:
                loop = asyncio.get_running_loop()
                self._cache[key] = await loop.run_in_executor(
                    None, lambda: AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=trust_remote_code),
                )
            return self._cache[key]

    async def count(
        self,
        hf_repo: str,
        text: str,
        *,
        trust_remote_code: bool,
        fallback_repo: str | None = None,
    ) -> int:
        """Token count for accounting. NEVER raises.

        ``fallback_repo`` is the model row's ``tokenizer_repo`` and, when set, is
        used INSTEAD OF ``hf_repo`` -- not after it. A GGUF-only repo has no
        tokenizer files at all, so trying it first is a guaranteed miss and, on a
        cold cache, a pointless network round trip. (The name is
        ``fallback_repo`` because that is what the column is for from the row's
        point of view: a fallback source for files the weights repo lacks.)

        When no tokenizer can be loaded we fall back to a character estimate
        rather than raising. This call sits on the proxy's hot path
        (app/proxy/routes.py:418) with no try/except above it, so an exception
        here is a 500 on a request the engine could have served perfectly well --
        which is exactly what a GGUF-only repo produced before sub-project C.
        The degradation is logged ONCE per repo (a per-request log on the hot
        path is a second incident) and reported through ``estimating()``.
        """
        if not text:
            return 0
        repo = fallback_repo or hf_repo
        try:
            tok = await self.get(repo, trust_remote_code=trust_remote_code)
            return len(tok.encode(text))
        except Exception:  # noqa: BLE001 -- accounting must not fail a request
            if hf_repo not in self._estimating:
                self._estimating.add(hf_repo)
                logger.warning(
                    "TokenizerCache: no usable tokenizer for %r (tried %r); token "
                    "accounting for this model is a CHARACTER ESTIMATE, which "
                    "also feeds the per-token usage figures. Set the model's "
                    "tokenizer_repo to a repo that ships tokenizer.json -- for a "
                    "GGUF quant that is normally the upstream safetensors repo.",
                    hf_repo,
                    repo,
                )
            return max(1, len(text) // _CHARS_PER_TOKEN)

    async def evict(self, hf_repo: str) -> int:
        """Drop every cached tokenizer for ``hf_repo`` (both trust_remote_code
        variants). Returns the number of entries that were evicted. Idempotent
        — calling on an hf_repo never seen by the cache is a no-op that
        returns 0.

        Called from the model-unload path so the cache doesn't accumulate
        an entry per ever-loaded model over a long-lived warden's lifetime.
        See code-review finding #5 (S7, #124).
        """
        async with self._lock:
            to_drop = [k for k in self._cache if k[0] == hf_repo]
            for k in to_drop:
                self._cache.pop(k, None)
            # Clear the estimate marker too, or a model whose tokenizer_repo the
            # operator has just fixed keeps estimating until the process
            # restarts -- and the once-per-repo log would never fire again to
            # say so.
            self._estimating.discard(hf_repo)
        if to_drop:
            logger.debug(
                "TokenizerCache.evict(%r) dropped %d entr%s",
                hf_repo, len(to_drop), "y" if len(to_drop) == 1 else "ies",
            )
        return len(to_drop)

    def size(self) -> int:
        """Number of cached tokenizer entries. For tests + observability."""
        return len(self._cache)
