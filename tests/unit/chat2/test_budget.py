from app.chat2.budget import BudgetDecision, BudgetWindow
from app.chat2.routes_chats import serialize_budget


def test_budget_window_serialises_as_the_contract_object() -> None:
    d = BudgetDecision(
        allowed=False,
        blocked_until="2026-08-25T00:00:00.000Z",
        window=BudgetWindow(
            used_micros=1200, limit_micros=1000, resets_at="2026-08-25T00:00:00.000Z"
        ),
    )
    assert serialize_budget(d) == {
        "blocked_until": "2026-08-25T00:00:00.000Z",
        "window": {
            "used_micros": 1200,
            "limit_micros": 1000,
            "resets_at": "2026-08-25T00:00:00.000Z",
        },
    }


def test_allowed_decision_serialises_as_null() -> None:
    assert serialize_budget(BudgetDecision(allowed=True)) is None
