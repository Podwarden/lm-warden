"""Prometheus exposition-text parsing. Knows nothing about any engine.

Lifted verbatim out of app/stats/live_engine.py by sub-project C. It lived there
because there was only one dialect; with two, the PARSER is the half that is
shared and the metric NAMES are the half that is not. Backends import this to
implement Backend.parse_metrics (app/runtime/backends/*/metrics.py); the stats
layer imports it for the same scrape. It deliberately depends on nothing under
``app`` so neither direction can cycle.

Not a general-purpose Prometheus client. Two behaviours are load-bearing and
must survive any tidying: a malformed sample line is DROPPED rather than raised,
because a partial scrape must degrade the dashboard rather than kill the stream;
and Metrics.value returns None -- never 0 -- for a name it cannot find, because
a metric an engine does not publish must render as absent rather than as idle.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Prometheus text parser (inline, no new dependency)
# --------------------------------------------------------------------------- #

# A metric line is ``name{labels} value [timestamp]`` or ``name value``.
_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+(.+)$")
# One label pair: ``key="value"`` where value may contain escaped quotes.
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def _unescape(v: str) -> str:
    return v.replace("\\\\", "\\").replace('\\"', '"').replace("\\n", "\n")


def _parse_value(raw: str) -> float | None:
    tok = raw.split()[0] if raw else ""
    try:
        return float(tok)
    except ValueError:
        return None


@dataclass(frozen=True)
class PromSample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus(text: str) -> list[PromSample]:
    """Parse Prometheus exposition text into a flat list of samples.

    ``# HELP`` / ``# TYPE`` comment lines and blanks are skipped. A malformed
    sample line is dropped rather than raising — a partial scrape must never
    crash the stream.
    """
    out: list[PromSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, label_blob, raw_val = m.group(1), m.group(2), m.group(3)
        val = _parse_value(raw_val)
        if val is None:
            continue
        labels: dict[str, str] = {}
        if label_blob:
            for k, v in _LABEL_RE.findall(label_blob):
                labels[k] = _unescape(v)
        out.append(PromSample(name=name, labels=labels, value=val))
    return out


class Metrics:
    """Accessor over parsed Prometheus samples.

    All lookups aggregate (sum) across label series with the same metric name.
    That is right for ADDITIVE series — request counts, cumulative counters —
    where a metric that an engine splits by an extra label (e.g. a
    data-parallel engine index) must collapse to the engine-wide total. It is
    NOT right for per-rank FRACTIONS, whose sum runs to the rank count instead
    of 1: the dialect that knows a metric is a fraction
    (app/runtime/backends/vllm/metrics.py) normalises by counting the label
    values with ``label_values``. A name with no matching series returns
    ``None`` — never a crash — which is how a renamed/absent metric turns into
    ``null`` in the frame.
    """

    def __init__(self, samples: list[PromSample]) -> None:
        self._samples = samples
        self._by_name: dict[str, list[PromSample]] = defaultdict(list)
        for s in samples:
            self._by_name[s.name].append(s)

    def _matching(self, name: str, label_filter: dict[str, str]):
        for s in self._by_name.get(name, ()):
            if all(s.labels.get(k) == v for k, v in label_filter.items()):
                yield s

    def value(self, name: str, **label_filter: str) -> float | None:
        """Sum of matching series' values, or ``None`` if none match."""
        vals = [s.value for s in self._matching(name, label_filter)]
        return sum(vals) if vals else None

    def value_any(self, *names: str, **label_filter: str) -> float | None:
        """First non-None ``value`` across alternative metric names.

        vLLM 0.25.1 renamed several counters and sometimes appends ``_total``;
        callers pass the plausible spellings and take whichever exists.
        """
        for name in names:
            v = self.value(name, **label_filter)
            if v is not None:
                return v
        return None

    def info(self, name: str) -> dict[str, str] | None:
        """Labels of the first series with ``name`` (Prometheus ``*_info``)."""
        for s in self._by_name.get(name, ()):
            return s.labels
        return None

    def label_values(self, name: str, label: str) -> set[str]:
        """Distinct values ``label`` takes across ``name``'s series.

        How a dialect counts the per-rank series a data-parallel engine splits
        a metric into (vLLM labels each ``engine="0".."N-1"``). Empty when the
        metric is absent or never carries the label — the caller decides what
        that means (for vLLM: a single engine, no normalisation).
        """
        return {s.labels[label] for s in self._by_name.get(name, ()) if label in s.labels}

    def histogram(self, base: str):
        """Return ``(buckets, sum, count)`` for a histogram, or ``None``.

        ``buckets`` is a list of ``(le, cumulative_count)`` sorted ascending,
        counts summed across any non-``le`` label series. Returns ``None`` when
        the histogram is entirely absent.
        """
        bmap: dict[float, float] = defaultdict(float)
        saw_bucket = False
        for s in self._by_name.get(base + "_bucket", ()):
            le = s.labels.get("le")
            if le is None:
                continue
            try:
                le_f = float(le)
            except ValueError:
                continue
            bmap[le_f] += s.value
            saw_bucket = True
        total_sum = self.value(base + "_sum")
        total_count = self.value(base + "_count")
        if not saw_bucket and total_count is None:
            return None
        buckets = sorted(bmap.items())
        return buckets, total_sum, total_count


def hist_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Interpolate a quantile from cumulative histogram buckets.

    ``buckets`` is ascending ``(le, cumulative_count)`` including the ``+Inf``
    bucket. Linear interpolation within the bucket that straddles the rank —
    the standard Prometheus ``histogram_quantile`` shape. Returns ``None`` for
    an empty or all-zero histogram.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if not total or total <= 0:
        return None
    rank = q * total
    prev_le = 0.0
    prev_c = 0.0
    for le, c in buckets:
        if rank <= c:
            if math.isinf(le):
                # Rank falls in the open-ended top bucket; the best finite
                # answer is the last finite boundary we passed.
                return prev_le if prev_c > 0 else None
            if c <= prev_c:
                return le
            frac = (rank - prev_c) / (c - prev_c)
            return prev_le + frac * (le - prev_le)
        if not math.isinf(le):
            prev_le = le
        prev_c = c
    return prev_le


def hist_mean(sum_v: float | None, count_v: float | None) -> float | None:
    if sum_v is None or count_v is None or count_v <= 0:
        return None
    return sum_v / count_v
