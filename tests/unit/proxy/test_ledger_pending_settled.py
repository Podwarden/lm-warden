"""pending() is blind while a flush holds its batch out; pending_settled reads under the ledger lock."""

import asyncio
import sqlite3

import pytest

from app.proxy.ledger import Ledger


@pytest.fixture
def fresh_ledger(tmp_path):
    return Ledger(tmp_path / "l.db")


async def test_raw_pending_is_blind_while_a_flush_holds_the_batch(fresh_ledger, monkeypatch):
    from tests.unit.proxy.ledger_helpers import pending_settled

    led = fresh_ledger
    entered, gate = asyncio.Event(), asyncio.Event()

    async def fake_write(self, batch):
        entered.set()
        await gate.wait()
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Ledger, "_write", fake_write)

    led.record_request(
        model_id="m",
        token_id=None,
        prompt_tokens=1,
        completion_tokens=1,
        cached_tokens=None,
        variant_id=None,
    )
    flush = asyncio.create_task(led.flush())
    await entered.wait()
    # The flush has swapped the batch for an empty one and not restored it yet.
    assert led.pending() is False
    settled = asyncio.create_task(pending_settled(led))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not settled.done()
    gate.set()
    assert await flush == 0
    assert await settled is True


async def test_pending_settled_false_when_empty(fresh_ledger):
    from tests.unit.proxy.ledger_helpers import pending_settled

    assert await pending_settled(fresh_ledger) is False


async def test_pending_settled_true_with_a_batch_and_no_flush(fresh_ledger):
    from tests.unit.proxy.ledger_helpers import pending_settled

    fresh_ledger.record_request(
        model_id="m",
        token_id=None,
        prompt_tokens=1,
        completion_tokens=1,
        cached_tokens=None,
        variant_id=None,
    )
    assert await pending_settled(fresh_ledger) is True
