"""Comparing what two run modes actually establish.

A cached measurement may be reused only when it answers the question being
asked. The cooldown originally asked "is there a recent measurement for this
fingerprint?" and never "does that measurement answer THIS request?", so a
`conservative` run blocked a `thorough` one and the operator was handed back a
result that could not contain what they had asked for.

The modes differ in what they establish, not merely in how long they take
(``PROFILES`` in ``runner.py``):

* ``conservative`` — crash budget 0, confirm 5/5, one needle offset, no sweep
* ``quick``        — crash budget 2, otherwise as conservative
* ``thorough``     — crash budget 4, confirm 7/7, three needle offsets, and the
  configuration sweep, which is the ONLY source of ``recommended_config``

So `thorough` is the only mode that can answer "what should I set
``max_model_len`` to?" at all.
"""

from __future__ import annotations

#: Ranked by what a mode ESTABLISHES, which is not the same as how long it
#: takes or how much it risks.
#:
#: `conservative` and `quick` differ ONLY in crash budget -- 0 against 2. Both
#: confirm 5/5, both probe the single 0.5 needle offset, neither runs the
#: sweep. A crash budget is a willingness to break the engine, not a stronger
#: claim about the model, so the two produce the same kind of measurement and
#: either satisfies the other. Ranking them apart would send an operator to the
#: GPU for a number they already had.
#:
#: `thorough` is strictly more: 7/7 confirmation, three needle offsets, and the
#: configuration sweep -- the only source of `recommended_config`.
_RANK = {"conservative": 0, "quick": 0, "thorough": 1}


def at_least_as_thorough(cached: str | None, requested: str | None) -> bool:
    """True when a ``cached`` run's mode can stand in for ``requested``.

    Unknown modes are never treated as sufficient, on either side, except for
    an exact match. The asymmetry of cost drives that: reusing wrongly withholds
    the answer the operator asked for and reports success, while re-running
    wrongly costs GPU time and tells them the truth.
    """
    if cached == requested:
        return True
    c, r = _RANK.get(cached or ""), _RANK.get(requested or "")
    if c is None or r is None:
        return False
    return c >= r
