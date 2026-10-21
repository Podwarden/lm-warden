"""/api/stats/v2/overview says whether this platform can report power at all.

The page used to infer "unsupported" from power_w == null with an empty series,
which a Linux GPU that has just started also shows. The server knows the
telemetry source (Apple silicon needs root for powermetrics), so it says so.
"""

import pytest

from app import platform_profile
from tests.conftest import jwt_login
from tests.unit.stats.test_stats_api import _seed_done


@pytest.mark.parametrize(("profile", "expect"), [("linux", True), ("apple", False)])
def test_overview_reports_power_supported_from_the_platform(
    tmp_data_dir, client, monkeypatch, profile, expect
):
    client.get("/healthz")
    _seed_done(tmp_data_dir / "vllm-warden.db")
    monkeypatch.setattr(platform_profile, "PROFILE", profile)
    auth = jwt_login(client, username="admin", password="hunter2")
    r = client.get("/api/stats/v2/overview?range=1h", headers=auth)
    assert r.status_code == 200, r.text
    cur = r.json()["current"]
    assert cur["power_w"] is None  # nothing sampled either way
    assert cur["power_supported"] is expect
