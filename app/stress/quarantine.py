"""Excluding probes that cannot discriminate (design §4.3).

A probe that trips a gate at the BASELINE condition tells us nothing about
where the model breaks: it reads the same at the reference length as at the
axis maximum, so every rung fails and no value is ever confirmed. It does not
find a limit — it hides one.

Measured on the second production run, 2026-09-04, `gpt-oss-20b-gguf` with the
budget fix in place and a clean baseline otherwise::

    needle   -> pass at 1024, 960, 896, 8192
    summary  -> repeating at EVERY length, baseline included
    result: raw_confirmed 0, first_observed_failure 1024, nothing published

Asked the same question directly the model answered in two clean sentences, so
that `repeating` was a false positive — but the argument for exclusion does not
rest on that. A signal that is constant across the whole axis cannot locate a
threshold on it, whether or not the signal is "correct".

``gates.py`` already refuses to judge against a degenerate baseline for the
SHORT gate. This applies the same rule to the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from app.stress.gates import Gate


def quarantined(baseline_gates: Mapping[str, Iterable[str]]) -> tuple[str, ...]:
    """Probe ids that tripped any gate at the baseline, in stable order.

    Sorted so the published record does not churn between runs that found the
    same thing.
    """
    return tuple(sorted(pid for pid, gates in baseline_gates.items() if tuple(gates)))


def ladder_probes(ladder: Iterable[str], excluded: Iterable[str]) -> tuple[str, ...]:
    """The ladder probes still able to discriminate.

    An empty result is a real answer and is deliberately not special-cased into
    "everything passed": with no oracle left, the caller must abort rather than
    publish the axis maximum on no evidence.
    """
    drop = set(excluded)
    return tuple(p for p in ladder if p not in drop)


def confirm_exclusion(first: Iterable[Gate], second: Iterable[Gate]) -> bool:
    """True when a baseline failure repeated, and the probe should be retired.

    Quarantine used to rest on a single sample, which contradicts the argument
    the rest of the search is built on: ``search.py`` replicates every ladder
    verdict because one sample near a threshold is a coin flip. The baseline
    shapes the whole run and was deciding on one.

    The cost of being wrong is asymmetric. Keeping a genuinely broken probe
    wastes some probes and it fails again at the first rung. Retiring a good one
    removes an oracle permanently -- and with only two ladder probes, two such
    mistakes abort the run outright, which is exactly what happened twice on
    2026-09-04.

    The two failures need not be the SAME gate: what disqualifies a probe is
    that it cannot produce a usable reference, not the particular way it fails.
    """
    return bool(tuple(first)) and bool(tuple(second))
