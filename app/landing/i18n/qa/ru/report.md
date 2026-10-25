# Translation QA: ru

- Date: 2026-10-25
- Back-translation model: claude-sonnet-5-5; fixer: claude-opus-5-5
- Segments covered: 667 of 667 published (14402 English words)
- Issues found (critical/major/minor): 0/126/428
- Issues remaining (critical/major/minor): 0/0/2
- MQM score: 0.14 per 1,000 words (pass at <= 10.0, no critical or major)
- Result: **PASS**

## Remaining issues

| Key | Category | Severity | Status | Note |
|---|---|---|---|---|
| `chrome.docs.short` | style | minor | accepted | [meaning review] «Доки» is informal; accepted because the label has a hard 6-character limit and «Справка»/«Документация» do not fit, while «Докум.» (old) was an awkward abbreviation |
| `coding-agents.agents.h1-1` | style | minor | accepted | [meaning review] would read better with a dash («Ваш агент… — на ваших GPU»), but the exact string is pinned by tests/unit/landing/test_i18n_pages.py::test_russian_pages_are_russian; left as is |
