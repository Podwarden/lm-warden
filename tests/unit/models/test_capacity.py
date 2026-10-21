"""Pure-function tests for ``app/models/capacity.py`` (Metal design §7, part a).

Hand-computed expected values for a known M1 16 GiB box plus boundary legs for
every arm of the formula: wired-limit fallback, the wired-vs-ceiling min, the
os-reserve max(3 GiB, 15%) crossover, the two subtractions, and the negative
result that feeds ``fit.classify_fit`` as-is. If any of these numbers drift,
the weights budget the supervisor reports has drifted — update the test
intentionally.

The §7 formula::

    device_ceiling = MTLDevice.recommendedMaxWorkingSetSize
    wired_ceiling  = iogpu.wired_limit_mb * 2^20  if nonzero else device_ceiling
    budget         = min(device_ceiling, wired_ceiling)
    os_reserve     = max(3 GiB, 0.15 * hw.memsize_usable)   # hw.memsize fallback only
    weights_budget = budget - os_reserve - other_engines - kv_reserve

With a readable Metal ceiling the OS reserve is 0 (``os_reserve=False``):
the ceiling already leaves the OS its share. The hand-computed legs below
exercise the fallback arithmetic (the default ``os_reserve=True``).

The pure part is ``unified_weights_budget``: it takes every one of those
inputs as a number and does the arithmetic. ``app/models/fit.py`` is untouched
and still takes a number.
"""

from __future__ import annotations

import math

from app.models.capacity import unified_weights_budget
from app.models.fit import classify_fit, kv_reserve_bytes

# ---- Constants used by hand-computed cases -------------------------------

GIB = 1024**3
MIB = 1024**2

# A real M1 16 GB box (Metal design §7): measured device ceiling, 16 GiB
# machine reporting 15.4 GiB usable (the spec's "verified 597 MB apart").
M1_DEVICE_CEILING = int(10.67 * GIB)
M1_MEMSIZE_USABLE = int(15.4 * GIB)


# ---- red-first case: known M1 16 GiB box ---------------------------------


def test_m1_16gib_hand_computed():
    """M1 16 GB, default wired limit, no other engines, 1 GiB KV reserve.

    budget     = min(10.67 GiB, ceiling) = 10.67 GiB
    os_reserve = max(3 GiB, 0.15 * 15.4 GiB) = 3 GiB (2.31 GiB < 3 GiB)
    result     = 10.67 GiB - 3 GiB - 0 - 1 GiB = 7161857966 bytes
    """
    from app.models.capacity import unified_weights_budget

    got = unified_weights_budget(M1_DEVICE_CEILING, 0, M1_MEMSIZE_USABLE, 0, GIB)
    assert got == M1_DEVICE_CEILING - 3 * GIB - GIB
    assert got == 7161857966


# ---- (a) wired_limit_mb 0 -> device ceiling used --------------------------


def test_wired_zero_falls_back_to_ceiling():
    """wired_limit_mb 0 means "default" — the ceiling stands alone."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 16 * GIB
    got = unified_weights_budget(ceiling, 0, memsize, 0, 0)
    assert got == ceiling - 3 * GIB  # 15% of 16 GiB = 2.4 GiB < 3 GiB floor


# ---- (b) wired vs ceiling: below wins, above loses, equal is the bound ----


def test_wired_below_ceiling_wins():
    """A lowered wired cap tightens the budget below the ceiling."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 16 * GIB
    wired_mb = 4 * 1024  # 4 GiB wired < 10 GiB ceiling
    got = unified_weights_budget(ceiling, wired_mb, memsize, 0, 0)
    assert got == 4 * GIB - 3 * GIB


def test_wired_above_ceiling_ceiling_wins():
    """A raised wired cap cannot grant more than the device ceiling."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 32 * GIB
    wired_mb = 12 * 1024  # 12 GiB wired > 10 GiB ceiling
    got = unified_weights_budget(ceiling, wired_mb, memsize, 0, 0)
    # 15% of 32 GiB = 4.8 GiB > 3 GiB floor, so the fraction wins here.
    assert got == 10 * GIB - math.ceil(0.15 * 32 * GIB)


def test_wired_equal_to_ceiling_bound():
    """wired == ceiling: the min is exactly the ceiling (bound leg)."""
    from app.models.capacity import unified_weights_budget

    ceiling = 8 * GIB
    memsize = 8 * GIB  # 15% = 1.2 GiB < 3 GiB floor
    wired_mb = 8 * 1024  # wired == ceiling exactly
    got = unified_weights_budget(ceiling, wired_mb, memsize, 0, 0)
    assert got == 8 * GIB - 3 * GIB
    # One MiB above the ceiling and the ceiling must still win.
    got_above = unified_weights_budget(ceiling, wired_mb + 1, memsize, 0, 0)
    assert got_above == 8 * GIB - 3 * GIB
    # One MiB below and wired must win — the two legs disagree by 1 MiB.
    got_below = unified_weights_budget(ceiling, wired_mb - 1, memsize, 0, 0)
    assert got_below == 8 * GIB - MIB - 3 * GIB


# ---- (c) os_reserve: floor below 20 GiB usable, fraction above, equal at 20
#
# The max(3 GiB, 0.15 * memsize) crossover is at exactly 20 GiB usable:
# 0.15 * 20 GiB == 3 GiB. Below it the floor wins, above it the fraction.


def test_os_reserve_floor_wins_below_crossover():
    """memsize_usable just below 20 GiB: 15% < 3 GiB, the floor stands."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 19 * GIB  # 15% = 2.85 GiB < 3 GiB
    got = unified_weights_budget(ceiling, 0, memsize, 0, 0)
    assert got == 10 * GIB - 3 * GIB


def test_os_reserve_fraction_wins_above_crossover():
    """memsize_usable just above 20 GiB: 15% > 3 GiB, the fraction wins."""
    from app.models.capacity import unified_weights_budget

    ceiling = 32 * GIB
    memsize = 21 * GIB  # 15% = 3.15 GiB > 3 GiB
    got = unified_weights_budget(ceiling, 0, memsize, 0, 0)
    assert got == 32 * GIB - math.ceil(0.15 * 21 * GIB)


def test_os_reserve_exactly_at_crossover_bound():
    """memsize_usable == 20 GiB: 15% == 3 GiB — both arms equal (bound leg)."""
    from app.models.capacity import unified_weights_budget

    ceiling = 32 * GIB
    memsize = 20 * GIB
    got = unified_weights_budget(ceiling, 0, memsize, 0, 0)
    # Either arm gives 3 GiB here; the result is 32 GiB - 3 GiB either way.
    assert got == 32 * GIB - 3 * GIB
    assert 0.15 * memsize == 3 * GIB  # sanity: the legs really do meet here


def test_os_reserve_fraction_rounds_up_so_budget_never_overstates():
    """15% of 21 GiB is 3382286745.6 bytes: the reserve rounds UP to
    3382286746, so the budget rounds DOWN — capacity never overstated."""
    from app.models.capacity import unified_weights_budget

    memsize = 21 * GIB
    assert 0.15 * memsize == 3382286745.6  # really fractional
    got = unified_weights_budget(32 * GIB, 0, memsize, 0, 0)
    assert got == 32 * GIB - 3382286746
    # Huge box: 15% of 512 GiB is 82463372083.2 -> reserve 82463372084.
    huge = unified_weights_budget(1024 * GIB, 0, 512 * GIB, 0, 0)
    assert huge == 1024 * GIB - 82463372084


def test_os_reserve_floor_at_zero_and_tiny_memsize():
    """0 or a few bytes of usable RAM: the 3 GiB floor still applies, the
    fraction never drops the reserve below it."""
    from app.models.capacity import unified_weights_budget

    assert unified_weights_budget(10 * GIB, 0, 0, 0, 0) == 7 * GIB
    assert unified_weights_budget(10 * GIB, 0, 7, 0, 0) == 7 * GIB
    # A zero device ceiling is a negative budget, not an error or a clamp.
    assert unified_weights_budget(0, 0, 0, 0, 0) == -3 * GIB


def test_wired_limit_is_mib_not_mb():
    """iogpu.wired_limit_mb is in MiB (2^20 bytes), not 10^6-byte MB."""
    from app.models.capacity import unified_weights_budget

    got = unified_weights_budget(64 * GIB, 5000, 16 * GIB, 0, 0)
    assert got == 5000 * MIB - 3 * GIB


# ---- (d) other_engines and kv_reserve are each subtracted -----------------


def test_other_engines_subtracted():
    """A second loaded engine's footprint comes off the budget (leg: others)."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 16 * GIB
    got = unified_weights_budget(ceiling, 0, memsize, 2 * GIB, 0)
    assert got == 10 * GIB - 3 * GIB - 2 * GIB


def test_kv_reserve_subtracted():
    """The KV reserve comes off the budget (leg: kv)."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 16 * GIB
    got = unified_weights_budget(ceiling, 0, memsize, 0, 512 * MIB)
    assert got == 10 * GIB - 3 * GIB - 512 * MIB


def test_other_and_kv_subtracted_together():
    """Both subtractions compose; order is irrelevant (pure arithmetic)."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 16 * GIB
    got = unified_weights_budget(ceiling, 0, memsize, 2 * GIB, 512 * MIB)
    assert got == 10 * GIB - 3 * GIB - 2 * GIB - 512 * MIB


# ---- (e) negative result is returned, not clamped --------------------------


def test_negative_result_returned_not_clamped():
    """KV + others exceeding the budget yields a negative int, like
    fit.weights_budget_bytes — classify_fit reads it as the red signal."""
    from app.models.capacity import unified_weights_budget

    ceiling = 10 * GIB
    memsize = 16 * GIB
    got = unified_weights_budget(ceiling, 0, memsize, 4 * GIB, 6 * GIB)
    assert got == 10 * GIB - 3 * GIB - 4 * GIB - 6 * GIB
    assert got == -3 * GIB
    assert got < 0


# ---- (f) integration with fit.classify_fit — fit.py unchanged --------------


def test_classify_fit_consumes_unified_budget():
    """The budget feeds classify_fit as-is: positive -> ratio ladder,
    non-positive -> red, exactly like the CUDA path."""
    from app.models.capacity import unified_weights_budget

    ceiling = M1_DEVICE_CEILING
    memsize = M1_MEMSIZE_USABLE
    kv = GIB
    budget = unified_weights_budget(ceiling, 0, memsize, 0, kv)
    assert budget > 0

    assert classify_fit(0, budget) == "green"
    # 50% of the budget is safely under the 0.55 green line.
    assert classify_fit(budget // 2, budget) == "green"
    # 70% sits between the 0.55 green and 0.80 yellow lines.
    assert classify_fit(int(0.7 * budget), budget) == "yellow"
    # 90% sits between the 0.80 yellow and 1.0 orange lines.
    assert classify_fit(int(0.9 * budget), budget) == "orange"
    # At the budget itself the ratio is 1.0 -> red (the < 1.0 bound).
    assert classify_fit(budget, budget) == "red"
    # And a budget that goes negative classifies as red at ANY file size.
    negative = unified_weights_budget(ceiling, 0, memsize, 0, 20 * GIB)
    assert negative < 0
    assert classify_fit(0, negative) == "red"
    assert classify_fit(1, negative) == "red"


def test_kv_reserve_bytes_feeds_unified_budget():
    """The existing pure KV formula composes with the unified budget — the
    two modules share one arithmetic core, no Metal in either."""
    from app.models.capacity import unified_weights_budget

    kv = kv_reserve_bytes(
        hidden_size=4096,
        num_layers=32,
        num_kv_heads=8,
        num_attention_heads=32,
        max_model_len=4096,
        dtype_bytes=2,
    )
    budget = unified_weights_budget(M1_DEVICE_CEILING, 0, M1_MEMSIZE_USABLE, 0, kv)
    assert budget == M1_DEVICE_CEILING - 3 * GIB - kv


# ---- the defaults are judgement calls: overridable, labelled ---------------


def test_defaults_are_parameters():
    """The 3 GiB floor and 15% fraction are module constants (judgement
    calls per §7) exposed as overridable parameters — the settings layer
    will override them without touching the formula."""
    from app.models.capacity import (
        OS_RESERVE_FLOOR_BYTES,
        OS_RESERVE_FRACTION,
        unified_weights_budget,
    )

    assert OS_RESERVE_FLOOR_BYTES == 3 * GIB
    assert OS_RESERVE_FRACTION == 0.15

    ceiling = 10 * GIB
    memsize = 16 * GIB
    # Overriding the floor to 8 GiB (above the 2.4 GiB fraction for 16 GiB)
    # changes the answer by exactly 5 GiB versus the 3 GiB default floor.
    got_floor = unified_weights_budget(ceiling, 0, memsize, 0, 0, os_reserve_floor=8 * GIB)
    assert got_floor == 10 * GIB - 8 * GIB
    # Overriding the fraction to 0.5 makes it dominate the floor.
    got_frac = unified_weights_budget(ceiling, 0, memsize, 0, 0, os_reserve_fraction=0.5)
    assert got_frac == 10 * GIB - 8 * GIB


# ---- UnifiedCapacity: resolving probe readings, fail closed --------------
#
# ``UnifiedCapacity`` holds the raw probe readings (any of them may be None)
# and resolves them into the two numbers the Mac consumers need: the device
# budget (GPU table total, shell feed usable_bytes, admission ceiling) and
# the reserve basis. Unknown never means bigger: an unreadable hw.memsize is
# budget 0, an unknown ceiling falls back to hw.memsize (the pre-P6
# behaviour), and nothing ever exceeds hw.memsize.

M1_MEMSIZE = 16 * GIB
M1_REAL_CEILING = 12713115648  # recommendedMaxWorkingSetSize on the farm M1


def test_capacity_m1_real_ceiling():
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(
        memsize=M1_MEMSIZE,
        memsize_usable=M1_MEMSIZE_USABLE,
        wired_limit_mb=0,
        device_ceiling=M1_REAL_CEILING,
    )
    assert cap.device_budget == M1_REAL_CEILING
    assert cap.reserve_basis == M1_MEMSIZE_USABLE
    assert cap.has_device_ceiling
    # The Metal ceiling already leaves the OS its share of unified memory:
    # no OS reserve on top of it (~11.84 GiB, not the old ~8.84 GiB).
    assert cap.weights_budget() == M1_REAL_CEILING
    assert cap.weights_budget() == unified_weights_budget(
        M1_REAL_CEILING, 0, M1_MEMSIZE_USABLE, 0, 0, os_reserve=False
    )
    # The real ceiling is smaller than the hw.memsize stand-in it replaces.
    assert cap.weights_budget() < unified_weights_budget(M1_MEMSIZE, 0, M1_MEMSIZE, 0, 0)


def test_capacity_known_ceiling_no_os_reserve_live_m1_values():
    """The farm M1's live readings: ceiling 12713115648, hw.memsize
    17179869184, hw.memsize_usable 16595435520. The weights budget is the
    ceiling itself (minus KV / other engines), never ceiling - 3 GiB."""
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(
        memsize=17179869184,
        memsize_usable=16595435520,
        wired_limit_mb=0,
        device_ceiling=12713115648,
    )
    assert cap.weights_budget() == 12713115648
    assert cap.weights_budget() != 12713115648 - 3 * GIB  # the pre-fix 9491890176
    assert cap.weights_budget(other_engines_bytes=GIB, kv_reserve=2 * GIB) == (
        12713115648 - 3 * GIB
    )
    # A wired limit below the ceiling lowers it; still no OS reserve on top.
    low = UnifiedCapacity(
        memsize=17179869184,
        memsize_usable=16595435520,
        wired_limit_mb=8192,
        device_ceiling=12713115648,
    )
    assert low.weights_budget() == 8 * GIB


def test_capacity_no_ceiling_keeps_floor_reserve():
    """hw.memsize fallback (no ceiling readable): the floor reserve applies."""
    from app.models.capacity import UnifiedCapacity

    for ceiling in (None, 0, -1):
        cap = UnifiedCapacity(
            memsize=17179869184, memsize_usable=16595435520, device_ceiling=ceiling
        )
        assert not cap.has_device_ceiling
        # max(3 GiB, ceil(0.15 * 16595435520) = 2489315328) = 3 GiB
        assert cap.weights_budget() == 17179869184 - 3 * GIB


def test_capacity_ceiling_above_memsize_is_not_trusted():
    """A ceiling above hw.memsize is not a sane reading: fail closed to the
    hw.memsize fallback, floor reserve included."""
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(memsize=M1_MEMSIZE, device_ceiling=M1_MEMSIZE + GIB)
    assert not cap.has_device_ceiling
    assert cap.weights_budget() == M1_MEMSIZE - 3 * GIB


def test_capacity_ceiling_equal_to_memsize_is_trusted():
    """The sane-ceiling boundary is inclusive (0 < ceiling <= memsize): a
    ceiling exactly at hw.memsize is the Metal reading, so no OS reserve."""
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(memsize=M1_MEMSIZE, device_ceiling=M1_MEMSIZE)
    assert cap.has_device_ceiling
    assert cap.weights_budget() == M1_MEMSIZE


def test_capacity_memsize_unreadable_fails_closed():
    from app.models.capacity import UnifiedCapacity

    for memsize in (None, 0, -1):
        cap = UnifiedCapacity(
            memsize=memsize, memsize_usable=M1_MEMSIZE_USABLE, device_ceiling=M1_REAL_CEILING
        )
        assert cap.device_budget == 0
        assert cap.weights_budget() == 0


def test_capacity_unknown_ceiling_is_pre_p6_memsize_behaviour():
    from app.models.capacity import UnifiedCapacity

    for ceiling in (None, 0, -5):
        cap = UnifiedCapacity(memsize=M1_MEMSIZE, device_ceiling=ceiling)
        assert cap.device_budget == M1_MEMSIZE
        assert cap.reserve_basis == M1_MEMSIZE
        assert cap.weights_budget() == unified_weights_budget(M1_MEMSIZE, 0, M1_MEMSIZE, 0, 0)


def test_capacity_ceiling_above_memsize_is_clamped():
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(memsize=M1_MEMSIZE, device_ceiling=M1_MEMSIZE + GIB)
    assert cap.device_budget == M1_MEMSIZE


def test_capacity_wired_limit_lowers_but_never_raises():
    from app.models.capacity import UnifiedCapacity

    low = UnifiedCapacity(memsize=M1_MEMSIZE, wired_limit_mb=8192, device_ceiling=M1_REAL_CEILING)
    assert low.device_budget == 8 * GIB
    # A user-raised wired limit above the device ceiling is NOT honoured on
    # its own (design §7: budget = min(device_ceiling, wired_ceiling)) ...
    high = UnifiedCapacity(
        memsize=M1_MEMSIZE, wired_limit_mb=15 * 1024, device_ceiling=M1_REAL_CEILING
    )
    assert high.device_budget == M1_REAL_CEILING
    # ... nor above hw.memsize when the ceiling is unknown.
    huge = UnifiedCapacity(memsize=M1_MEMSIZE, wired_limit_mb=64 * 1024)
    assert huge.device_budget == M1_MEMSIZE


def test_capacity_wired_limit_unknown_is_os_default():
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(memsize=M1_MEMSIZE, wired_limit_mb=None, device_ceiling=M1_REAL_CEILING)
    assert cap.device_budget == M1_REAL_CEILING


def test_capacity_reserve_basis_never_below_usable_reading_bounds():
    from app.models.capacity import UnifiedCapacity

    # memsize_usable unknown, 0, or above memsize -> hw.memsize (the larger
    # basis: a bigger OS reserve, a smaller budget).
    for usable in (None, 0, M1_MEMSIZE + 1):
        cap = UnifiedCapacity(memsize=M1_MEMSIZE, memsize_usable=usable)
        assert cap.reserve_basis == M1_MEMSIZE


def test_capacity_weights_budget_subtracts_engines_and_kv():
    from app.models.capacity import UnifiedCapacity

    cap = UnifiedCapacity(
        memsize=M1_MEMSIZE, memsize_usable=M1_MEMSIZE_USABLE, device_ceiling=M1_REAL_CEILING
    )
    assert cap.weights_budget(other_engines_bytes=GIB, kv_reserve=2 * GIB) == (
        unified_weights_budget(
            M1_REAL_CEILING, 0, M1_MEMSIZE_USABLE, GIB, 2 * GIB, os_reserve=False
        )
    )
    assert cap.weights_budget(other_engines_bytes=GIB, kv_reserve=2 * GIB) == (
        M1_REAL_CEILING - 3 * GIB
    )
