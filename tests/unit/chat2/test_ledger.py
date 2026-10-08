import pytest

from app.chat2.budget import AlwaysAllow
from app.chat2.ledger import LedgerRepo, RateCard, Usage, compute_cost
from app.db.database import open_db


def _card(**over):
    base = dict(
        id="rc1",
        org_id=None,
        provider="openrouter",
        model="m",
        input_per_m_micros=1_000_000,
        output_per_m_micros=3_000_000,
        reasoning_per_m_micros=None,
        cache_read_per_m_micros=None,
        cache_write_per_m_micros=None,
        markup_pct=10,
        currency="USD",
        effective_from="2026-01-01T00:00:00.000Z",
    )
    base.update(over)
    return RateCard(**base)


def test_compute_cost_rounds_once_with_markup() -> None:
    # 1000 in @ $1/M = 1000 micros; 500 out @ $3/M = 1500; sum 2500 * 1.10 = 2750
    r = compute_cost(Usage(prompt=1000, completion=500), _card())
    assert (r.cost_micros, r.cost_status) == (2750, "priced")


def test_compute_cost_half_up_and_statuses() -> None:
    # 1 token @ $1/M = 1 micro; *1.10 = 1.1 -> 1 ; 5 tokens -> 5.5 -> 6 (half up)
    assert compute_cost(Usage(prompt=5, completion=0), _card()).cost_micros == 6
    assert compute_cost(Usage(prompt=5, completion=0), None).cost_status == "unpriced"
    r = compute_cost(Usage(prompt=5, completion=0, reasoning=10), _card())
    assert (r.cost_micros, r.cost_status) == (None, "incomplete_card")


@pytest.mark.asyncio
async def test_record_picks_latest_card_and_is_idempotent(tmp_data_dir, client) -> None:
    client.get("/healthz")
    async with open_db(tmp_data_dir / "vllm-warden.db") as db:
        for cid, eff, price in (
            ("old", "2026-01-01T00:00:00.000Z", 1_000_000),
            ("new", "2026-06-01T00:00:00.000Z", 2_000_000),
        ):
            await db.execute(
                "INSERT INTO rate_cards(id,org_id,provider,model,input_per_m_micros,"
                "output_per_m_micros,markup_pct,currency,effective_from) "
                "VALUES (?,NULL,'openrouter','m',?,0,0,'USD',?)",
                (cid, price, eff),
            )
        await db.commit()
        repo = LedgerRepo(db)
        card = await repo.find_card("openrouter", "m", None, "2026-08-01T00:00:00.000Z")
        assert card and card.id == "new"
        lid, cost = await repo.record(
            request_id="req1",
            user_id=1,
            org_id=None,
            chat_id="c",
            message_id="m1",
            purpose="turn",
            provider="openrouter",
            model="m",
            usage=Usage(prompt=1_000_000, completion=0),
            outcome="ok",
        )
        assert cost.cost_micros == 2_000_000
        assert await repo.find("req1", user_id=1) == ("ok", "m1")
        assert await repo.find("req1", user_id=2) is None
        cur = await db.execute("SELECT COUNT(*) FROM usage_ledger")
        assert (await cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_always_allow() -> None:
    d = await AlwaysAllow().check(1, None)
    assert d.allowed and d.blocked_until is None
