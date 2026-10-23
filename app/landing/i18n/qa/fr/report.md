# Translation QA: fr

- Date: 2026-10-23
- Back-translation model: claude-sonnet-5-5; fixer: claude-opus-5-5
- Segments covered: 667 of 667 published (14402 English words)
- Issues found (critical/major/minor): 0/29/313
- Issues remaining (critical/major/minor): 0/0/3
- MQM score: 0.21 per 1,000 words (pass at <= 10.0, no critical or major)
- Result: **PASS**

## Remaining issues

| Key | Category | Severity | Status | Note |
|---|---|---|---|---|
| `chrome.nav.plat-upgrade` | omission | minor | accepted | [meaning review] "existing" not rendered ("Mettre à jour une installation"): the nav label is limited to ~27 characters and the meaning is clear from context |
| `chrome.install` | omission | minor | accepted | [meaning review] "Download & Install" rendered "Installer": the label has an 18-character limit; the full "Télécharger et installer" is used where space allows |
| `chrome.docs.aria-label` | addition | minor | accepted | [meaning review] "Documentation du projet" adds "du projet" to a screen-reader label; harmless and disambiguates it from the visible "Documentation" link |
