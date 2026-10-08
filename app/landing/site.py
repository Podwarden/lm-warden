"""Per-request website state: is the site on, and which Google IDs are set.

One DB open per request for all four keys. Tracking IDs are re-validated
here with the settings coercer's patterns: a value written straight into the
settings table (bypassing PATCH) must still never reach a public page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import Request

from app.db.database import open_db
from app.db.repos.settings import SettingsRepo
from app.settings.tracking_ids import ADS_ID_RE, GA4_ID_RE, GSC_TOKEN_RE

_TRUTHY = {"true", "1", "yes", "on"}


@dataclass(frozen=True)
class Analytics:
    ga4_id: str = ""
    ads_id: str = ""
    gsc_token: str = ""

    @property
    def tag_on(self) -> bool:
        return bool(self.ga4_id or self.ads_id)


@dataclass(frozen=True)
class SiteState:
    enabled: bool
    analytics: Analytics


def _valid(raw: str | None, pattern: re.Pattern[str]) -> str:
    s = (raw or "").strip()
    return s if pattern.match(s) else ""


async def site_state(request: Request) -> SiteState:
    """Stored as canonical 'true' / 'false' by the settings coercer; any
    common truthy spelling is tolerated for manual edits. A missing row means
    off: the website is opt-in (spec 2026-10-07 §9)."""
    settings = request.app.state.settings
    async with open_db(settings.db_path) as db:
        repo = SettingsRepo(db)
        enabled = await repo.get("landing_page_enabled")
        ga4 = await repo.get("analytics_ga4_id")
        ads = await repo.get("analytics_google_ads_id")
        gsc = await repo.get("search_console_verification")
    return SiteState(
        enabled=(enabled or "").strip().lower() in _TRUTHY,
        analytics=Analytics(
            ga4_id=_valid(ga4, GA4_ID_RE),
            ads_id=_valid(ads, ADS_ID_RE),
            gsc_token=_valid(gsc, GSC_TOKEN_RE),
        ),
    )
