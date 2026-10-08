"""Did a run actually measure anything? (design §7.2)

Separate from the repo and from the routes because both need the same answer
and neither should own it — and because keeping it dependency-free lets it be
tested without a database.

The distinction this draws is not pedantic. A run that COMPLETED and a run that
MEASURED something are different states, and conflating them cost an operator
their retest: after the reasoning-budget bug (2026-09-04) a run completed
having confirmed nothing at all, and because it was `completed`, matched the
fingerprint, and was neither non-monotone nor traffic-tainted, it satisfied
every condition `current_for` checks and held the six-hour cooldown. Pressing
"Stress test" returned `200 reused` naming that dead run — no new run, no
error, and no way out, because the UI reveals `force` only on a 409.

A run that learned nothing is the strongest possible argument for running
again, not against it.
"""

from __future__ import annotations

#: Keys the runner writes into `limits` that are its own bookkeeping rather
#: than measurements. Counting them as evidence would make every run that
#: reached the baseline look like it had found something.
_NOT_AXES = frozenset({"baseline", "neighbours", "aborted_reason"})


def _positive(v: object) -> bool:
    """A real, positive number. `True` is not 1 here."""
    return not isinstance(v, bool) and isinstance(v, int | float) and v > 0


def is_axis(name: str) -> bool:
    """True when a `limits` key is a measured axis rather than bookkeeping.

    The runner files its own notes in the same dict as its measurements, so
    without this the publication layer renders them as limits: the shipped UI
    showed `neighbours` and `baseline` as MEASURED LIMITS cards, badge and all,
    each reporting a value of "—". A badge that appears on the runner's
    scratch notes is worth nothing on a real number.
    """
    return name not in _NOT_AXES


def has_measurement(limits: object, recommended_config: object) -> bool:
    """True when this run established at least one usable number.

    A recommendation is sufficient on its own. Failing that, an axis must carry
    a positive ``raw_confirmed`` — the value the search actually confirmed, as
    opposed to merely observed. Zero is not a small measurement: it means the
    search confirmed nothing, which is precisely the state this exists to
    catch.

    Total, and never raises. It gates a user-facing button, so an exception on
    a malformed blob would turn a bad record into a 500.
    """
    if isinstance(recommended_config, dict) and recommended_config:
        return True
    if not isinstance(limits, dict):
        return False
    for name, axis in limits.items():
        if name in _NOT_AXES:
            continue
        # An axis may be a bare number: the runner has recorded scalars, and
        # `_normalise_limit` widens them into objects on the way out. Requiring
        # the object form here would read a real measurement as no measurement.
        if _positive(axis):
            return True
        if isinstance(axis, dict):
            # The runner marks an axis it refuses to stand behind -- a
            # concurrency number taken against an undefeated prefix cache is
            # recorded with `invalid_reason: prefix_cache_not_defeated` and a
            # positive raw_confirmed beside it. Counting that would let a run
            # that published nothing usable hold the cooldown: the same
            # reader/writer disagreement this module exists to end.
            if axis.get("invalid_reason"):
                continue
            if _positive(axis.get("raw_confirmed")) or _positive(axis.get("value")):
                return True
    return False
