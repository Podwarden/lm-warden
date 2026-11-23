"""Cache observation for an isolated key: its own lens only, never the operator lens.

An isolated key's prefixes can never serve another key (its GPU salt and its RAM and
NVMe scope are its own), so recording them under the operator-wide ``"*"`` lens would
make the panels predict hits that cannot happen.
"""

from __future__ import annotations

from app.cache_obs.canonical import chain_of
from app.cache_obs.index import PrefixIndex

BODY = {"messages": [{"role": "user", "content": "x" * 5000}], "cache_salt": "s"}


def test_an_isolated_observation_is_on_its_own_lens_only() -> None:
    idx = PrefixIndex()
    chain = chain_of(BODY)
    assert chain is not None
    idx.observe("m", "e", None, chain, "tok-a", operator_lens=False)
    # B (another key, same chain): the operator lens finds nothing from A.
    assert idx.match("m", "e", chain).best_bytes == 0
    assert idx.match("m", "e", chain, token_id="tok-b").best_bytes == 0
    # A's own lens finds it.
    own = idx.match("m", "e", chain, token_id="tok-a")
    assert own.best_bytes > 0 and own.best_bytes == chain.total_bytes


def test_a_sharing_observation_is_on_both_lenses_as_before() -> None:
    idx = PrefixIndex()
    chain = chain_of(BODY)
    assert chain is not None
    idx.observe("m", "e", None, chain, "tok-b")
    assert idx.match("m", "e", chain).best_bytes == chain.total_bytes
    assert idx.match("m", "e", chain, token_id="tok-b").best_bytes == chain.total_bytes


def test_no_token_and_no_operator_lens_records_nothing() -> None:
    idx = PrefixIndex()
    chain = chain_of(BODY)
    assert chain is not None
    idx.observe("m", "e", None, chain, None, operator_lens=False)
    assert len(idx) == 0
