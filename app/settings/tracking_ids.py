"""Google ID shapes for the public website's tracking settings (spec
2026-10-07 §8). Their own module, with no app imports, because two places
need them: the settings PATCH coercer (app/settings/routes_api.py), and
app/landing/site.py, which re-validates whatever it reads from the DB before
echoing it into a public page."""

import re

GA4_ID_RE = re.compile(r"^G-[A-Z0-9]{4,12}$")
ADS_ID_RE = re.compile(r"^AW-[0-9]{6,12}$")
GSC_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
