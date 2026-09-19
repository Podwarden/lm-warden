"""Binning for ``GET /api/tokens/{id}/series`` (spec 2026-09-18 §3.4).

Pure: no database, no clock. The route validates a window with
``validate_window``, picks a bin width with ``pick_bin_minutes``, runs two
queries, and hands their rows to ``build_series``.

WHY SERVER-SIDE, AND WHY SUM ÷ WIDTH. The stats page bins in the browser and
averages only the minutes that have rows. That is right for ``model_samples``,
which has a row every minute. ``token_usage_minute`` has rows only for minutes
with traffic, so the same averaging would make a quiet token look busy: one
request in a 30-minute bin would read as "1 request per minute". Here every
rate is the bin's SUM divided by its WIDTH, so minutes without rows count as
zero and the y-axis stays "per minute" at every zoom level.

Bins are keyed ``floor(minute / width) * width``: UTC-aligned, so they do not
move between polls. Day bins therefore start at UTC midnight (spec §7).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

from app.stats.request_history import quantile

#: Bin widths in minutes, smallest first. The presets land exactly on the
#: stats page's client-side table: 1h -> 1, 6h -> 1, 24h -> 5, 7d -> 30.
LADDER: tuple[int, ...] = (1, 2, 5, 10, 15, 30, 60, 120, 360, 720, 1440, 10080)
#: Default and ceiling for ``max_bins``.
MAX_BINS = 360
#: Longest window the endpoint serves.
MAX_SPAN_S = 366 * 86400

#: (bin, requests, prompt_tokens, completion_tokens, peak_prompt, peak_completion)
#: -- one row of ``TokenUsageRepo.chain_bins``.
CountRow = tuple[int, int, int, int, int, int]
#: (finished_at, queued_s, ttft_s, duration_s) -- one row of
#: ``request_history.query_token_timings``.
TimingRow = tuple[float, float | None, float | None, float | None]
#: (model_id, model, variant_id, descriptor_json, first_seen, requests,
#: prompt_tokens, completion_tokens) -- one row of
#: ``TokenModelUsageRepo.by_variant``.
VariantRow = tuple[str, str, str, str | None, float | None, int, int, int]
#: (bin, model_id, prompt_tokens, completion_tokens) -- one row of
#: ``TokenModelUsageRepo.model_bins``.
ModelBinRow = tuple[int, str, int, int]

_TIMING_FIELDS = ("queue", "ttft", "duration")


class SeriesWindowError(ValueError):
    """A ``from``/``to`` pair the endpoint refuses; the route answers 422."""


def validate_window(from_s: float, to_s: float, *, now_s: float) -> float:
    """Validates ``[from_s, to_s)`` and returns the ``to`` to actually use.

    A client clock that runs fast must not turn a "last hour" poll into an
    empty chart: rather than 422 a future ``to``, it is clamped to the
    server's own ``now_s`` (ruled in 2026-09-18, clock-skew review). 422 is
    still raised, against the CLAMPED ``to``, when that leaves ``from >= to``
    (a ``from`` that is also in the future) or the span exceeds 366 days.
    The two ``from >= to`` cases get different messages (#251): a client that
    did send ``from < to`` is told its ``from`` is past the server's clock,
    not that it got the order wrong.
    """
    if not from_s < to_s:
        raise SeriesWindowError("'from' must be before 'to'")
    to_s = min(to_s, now_s)
    if not from_s < to_s:
        raise SeriesWindowError("'from' is after the server's current time")
    if to_s - from_s > MAX_SPAN_S:
        raise SeriesWindowError("the period is longer than 366 days")
    return to_s


def pick_bin_minutes(span_s: float, max_bins: int = MAX_BINS) -> int:
    """The smallest ladder width that splits ``span_s`` into <= ``max_bins``.

    Falls back to the widest rung when even that is too many (a tiny
    ``max_bins`` over a long span); the response then carries more bins than
    asked for rather than a width the rest of the product does not use.
    """
    span_min = span_s / 60.0
    for width in LADDER:
        if math.ceil(span_min / width) <= max_bins:
            return width
    return LADDER[-1]


def minute_window(from_s: float, to_s: float) -> tuple[int, int]:
    """``[from_minute, to_minute)``: every minute ``[from_s, to_s)`` touches,
    partial minutes at both ends included."""
    return int(from_s // 60), int(math.ceil(to_s / 60.0))


def bin_of_minute(minute: int, width: int) -> int:
    return (minute // width) * width


def chain_token_ids(
    lineage_ids: Sequence[str], token_id: str, *, include_earlier: bool
) -> list[str]:
    """The keys a chart covers, oldest first: ``token_id`` alone, or it and
    every EARLIER key of its rotation chain. Later keys are never included --
    a successor's traffic is not this key's history."""
    if not include_earlier or token_id not in lineage_ids:
        return [token_id]
    return list(lineage_ids[: list(lineage_ids).index(token_id) + 1])


def timing_bins(rows: Iterable[TimingRow], width: int) -> dict[int, dict[str, Any]]:
    """Per-bin request count and p50/p95 of queue wait, TTFT and duration.

    A request lands in the bin of the minute it FINISHED in. ``n`` counts
    every request; each percentile uses only the requests that measured that
    quantity (a non-streaming request has no TTFT), and is None when none did.
    """
    groups: dict[int, dict[str, Any]] = {}
    for finished_at, queued, ttft, duration in rows:
        key = bin_of_minute(int(finished_at // 60), width)
        g = groups.setdefault(key, {"n": 0, "queue": [], "ttft": [], "duration": []})
        g["n"] += 1
        for name, value in (("queue", queued), ("ttft", ttft), ("duration", duration)):
            if value is not None:
                g[name].append(float(value))
    out: dict[int, dict[str, Any]] = {}
    for key, g in groups.items():
        entry: dict[str, Any] = {"n": g["n"]}
        for name in _TIMING_FIELDS:
            values = sorted(g[name])
            entry[f"{name}_p50"] = quantile(values, 0.5)
            entry[f"{name}_p95"] = quantile(values, 0.95)
        out[key] = entry
    return out


_NO_TIMINGS: dict[str, Any] = {
    "n": 0,
    **{f"{name}_{q}": None for name in _TIMING_FIELDS for q in ("p50", "p95")},
}


def image_tag(image: str | None) -> str | None:
    """The tag of an image reference (``repo:tag`` -> ``tag``; a digest pin
    -> ``sha256:`` plus 12 hex), ``latest`` for an untagged one, None for
    none."""
    if not image:
        return None
    if "@" in image:
        return image.split("@", 1)[1][:19]
    last = image.rsplit("/", 1)[-1]
    return last.split(":", 1)[1] if ":" in last else "latest"


#: Descriptor fields the card's variant line shows; the rest (repo, files,
#: extra args...) only split variants apart. ``engine_version`` is the
#: in-container engine's baked version and ``hf_commit`` the commit the
#: revision resolved to at launch (app/runtime/variants.py, RUNTIME_FIELDS).
_SUMMARY_FIELDS = (
    "backend", "engine_channel", "engine_vllm_version", "engine_version",
    "quantization", "dtype", "hf_revision", "hf_commit", "max_model_len",
)


def _variant_summary(descriptor_json: str | None) -> dict[str, Any]:
    try:
        d = json.loads(descriptor_json) if descriptor_json else {}
    except ValueError:
        d = {}
    if not isinstance(d, dict):
        d = {}
    out: dict[str, Any] = {k: d.get(k) for k in _SUMMARY_FIELDS}
    # The image the engine actually ran (the pin, or the driver's default),
    # else the pin a row without runtime facts carries.
    out["engine_image_tag"] = image_tag(d.get("engine_image_used") or d.get("engine_image"))
    return out


def _totals(requests: int, prompt: int, completion: int) -> dict[str, int]:
    return {
        "requests": requests,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def by_model_rows(rows: Iterable[VariantRow]) -> list[dict[str, Any]]:
    """The "Usage by model" table: one entry per MODEL, busiest (prompt +
    completion) first, ties by model id, each with its ``variants`` (same
    order rule, ties by variant id).

    ``share`` -- on models and variants alike -- is the tokens over the sum
    across every model: the key's total WITHIN the per-model data, so model
    shares add up to 1 even when the window starts before 0036 did. 0.0 when
    that sum is 0.
    """
    models: dict[str, dict[str, Any]] = {}
    variants: dict[str, list[dict[str, Any]]] = {}
    for model_id, model, variant_id, desc, first_seen, requests, prompt, completion in rows:
        m = models.setdefault(model_id, {
            "model_id": model_id, "model": model, **_totals(0, 0, 0),
        })
        for key, n in (("requests", requests), ("prompt_tokens", prompt),
                       ("completion_tokens", completion), ("total_tokens", prompt + completion)):
            m[key] += n
        variants.setdefault(model_id, []).append({
            "variant_id": variant_id,
            **_variant_summary(desc),
            "first_seen": first_seen,
            **_totals(requests, prompt, completion),
        })
    grand = sum(m["total_tokens"] for m in models.values())

    def share(n: int) -> float:
        return n / grand if grand else 0.0

    out = sorted(models.values(), key=lambda m: (-m["total_tokens"], m["model_id"]))
    for m in out:
        vs = sorted(variants[m["model_id"]], key=lambda v: (-v["total_tokens"], v["variant_id"]))
        for v in vs:
            v["share"] = share(v["total_tokens"])
        m["share"] = share(m["total_tokens"])
        m["variants"] = vs
    return out


def model_bins(rows: Iterable[ModelBinRow], width: int) -> list[dict[str, Any]]:
    """Per-bin, per-model token rates for the tokens chart. Sparse at both
    levels -- only bins with per-model rows, and in each only the models that
    had traffic -- ascending by bin. Same keying and sum ÷ width rule as
    ``bins``."""
    grouped: dict[int, dict[str, dict[str, float]]] = {}
    for key, model_id, prompt, completion in rows:
        grouped.setdefault(int(key), {})[model_id] = {
            "prompt_per_min": prompt / width,
            "completion_per_min": completion / width,
        }
    return [{"minute": key, "models": grouped[key]} for key in sorted(grouped)]


def build_series(
    *,
    token_ids: Sequence[str],
    from_minute: int,
    to_minute: int,
    bin_minutes: int,
    count_rows: Iterable[CountRow],
    timing_rows: Sequence[TimingRow],
    timing_total: int,
    timing_stride: int,
    latency_since: float | None,
    by_model: Iterable[VariantRow] = (),
    model_bin_rows: Iterable[ModelBinRow] = (),
    by_model_since: float | None = None,
) -> dict[str, Any]:
    """The endpoint's response body. Bins are SPARSE -- only bins with usage
    or timings -- and ascending; the client fills the gaps with zeros for the
    counts and nulls for the timings.

    ``by_model`` / ``model_bins`` split the same window by model, from
    ``token_model_usage_minute`` (0036, no backfill): they cover only minutes
    from ``by_model_since`` (the first row store-wide, epoch seconds) on, so
    before that the chart falls back to the total line of ``bins``. All three
    are empty/None when the caller skipped them (``by_model=0``).

    ``timing_rows`` may be a stride sample (``timing_stride`` > 1) of
    ``timing_total`` rows; ``timing_sample`` says so, and each bin's ``n``
    counts the SAMPLED rows. The counts and tokens come from
    ``token_usage_minute`` and are always exact.
    """
    width = bin_minutes
    counts = {int(r[0]): r for r in count_rows}
    timings = timing_bins(timing_rows, width)
    totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
    bins: list[dict[str, Any]] = []
    for key in sorted(set(counts) | set(timings)):
        _bin, requests, prompt, completion, peak_prompt, peak_completion = counts.get(
            key, (key, 0, 0, 0, 0, 0)
        )
        totals["requests"] += requests
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        bins.append({
            "minute": key,
            "requests_per_min": requests / width,
            "prompt_per_min": prompt / width,
            "completion_per_min": completion / width,
            "peak_prompt": peak_prompt,
            "peak_completion": peak_completion,
            "requests": requests,
            **timings.get(key, _NO_TIMINGS),
        })
    return {
        "token_ids": list(token_ids),
        "from_minute": from_minute,
        "to_minute": to_minute,
        "bin_minutes": width,
        "latency_since": latency_since,
        "timing_sample": {
            "total": timing_total,
            "used": len(timing_rows),
            "stride": timing_stride,
        },
        "totals": totals,
        "bins": bins,
        "by_model_since": by_model_since,
        "by_model": by_model_rows(by_model),
        "model_bins": model_bins(model_bin_rows, width),
    }
