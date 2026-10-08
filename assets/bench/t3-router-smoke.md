| Asked for | Where it went | Status or reason | TTFB p50 (s) | Measured by |
|---|---|---|---|---|
| Haiku, matched by a rule | local | 200 | 0.108 | router smoke test |
| A model with no rule | passthrough | 200 | 0.607 | live Claude Code session |
| Haiku, local leg failed before the first byte, fall back on | fallback | status_400 | – | live Claude Code session |
| Opus, key without “May relay” | refused | 403 relay_not_allowed | – | router smoke test |
