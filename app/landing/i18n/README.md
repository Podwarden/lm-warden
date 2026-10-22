# Translating the home page

Only the public **home page** (`/`) and the chrome rendered with it (top bar,
menu, language switcher, footer, cookie banner, carousel / lightbox / copy
labels, `<title>`, meta description, og tags, JSON-LD) are translated. Every
other page (`/faq`, `/coding-agents`, client pages, `/cache-aware-routing`,
`/privacy`, …) stays English and is linked from every language. `llms.txt`
stays English. The console UI is not part of this.

English is the single source of truth. Translations are generated artifacts:
produce them with the pipeline below, never hand-edit a catalog outside it.

## Files

| File | What it is |
|---|---|
| `en.json` | The source catalog: `{key: {text, note, hash}}`, written by `scripts/landing_i18n.py write-source` from the marked HTML. Never edit by hand. |
| `<locale>.json` | One translation: `{key: {text, source_hash}}`. `source_hash` is en.json's `hash` of the English you translated. |
| `glossary.json` | Do-not-translate terms, preferred renderings of key terms per locale, and per-locale allowlists. |

Locales and their URLs (`app/landing/pages.py`, `LOCALES`): `fr` `/fr/`, `de`
`/de/`, `es` `/es/`, `pt-BR` `/pt-BR/`, `ru` `/ru/`, `uk` `/uk/`, `pl` `/pl/`,
`zh-CN` `/zh-CN/`, `ja` `/ja/`, `hi` `/hi/`. **A locale is published the moment
its file exists** (URL, switcher, hreflang, sitemap); without it its URL 404s.
So add a file only when it is complete and its checks pass.

## How to translate one locale (the hand-off)

You add exactly one file: `app/landing/i18n/<locale>.json`. Nothing else.

1. Get the segments: `../../.venv/bin/python scripts/landing_i18n.py todo <locale> > /tmp/todo.json`
   (all 284 for a new locale; only missing or stale ones for a refresh).
2. Translate each `text`, guided by its `note` (where it appears; a length
   limit for bar / button labels) and `glossary.json`. Write
   `{key: {"text": <translation>, "source_hash": <the segment's hash>}}`
   for every key, sorted as en.json is. `de.json` is the worked example.
3. Check: `../../.venv/bin/python scripts/landing_i18n.py check <locale>` must print
   `complete`, then `../../.venv/bin/python -m pytest -q tests/unit/landing`.
4. Look at it (below), then commit the one file.

### Rules a translation must keep (the check enforces the first five)

- **Tags and placeholders.** Inline HTML (`<b>`, `<code>`, `<em>`, `<a href="…" rel="noopener">`, `<span class="…">`) and tokens (`{{repo}}`, `{{mac_release}}`, `{n}`, `{m}`) must all survive, attributes byte-identical. You may move them to fit the sentence; never drop, add or edit one.
- **`<code>` spans verbatim**: commands, paths, env vars, endpoints.
- **Numbers keep their value**; separators follow the locale (`2.05` → `2,05` in de/fr/ru/uk/pl/es/pt-BR; thousands `11,178` → `11.178` de, `11 178` fr/ru/uk/pl with a no-break space). Units keep their symbols (GB, GiB, W, s, h). Use `&#8239;` (narrow no-break space) or `&nbsp;` between a number and its unit or `%` where your language puts a space.
- **Do-not-translate terms** (`glossary.json` `do_not_translate`) appear exactly as written whenever the English has them: product and brand names, model ids, and the console's own UI names (`Stats`, `Add model`, `Share on Local Network`, `Connect`, `God mode`, …) because the console itself is English. Do not glue them into compounds with hyphens (`LM-Warden-Server` breaks the check; write it so the name stands alone).
- **Attribute segments** (keys ending `.alt`, `.aria-label`, `.title`, `.resume`, `.dot`, `.done`, `.fail`, and `meta.*`) are plain text: no `"` and no `<`.
- **Nothing left in English by accident.** A segment you deliberately keep identical (e.g. `FAQ`, `APIs`) goes in `glossary.json` → `identical_ok` → your locale.
- **Length limits** in a note are hard: the top bar must still fit a 320px phone. `chrome.install` is the bar's short install button (de: "Installieren"); `home.hero.a-1` is the long one.
- Keep the English page's tone: plain, factual, no marketing superlatives. Do not add or drop claims.
- Inline SVG drawing labels and the `{{fig:…}}` charts stay English (data figures); their captions, `<title>` and `<desc>` are translated.

## Style: Microsoft's localization style guide per language

Follow Microsoft's style guide for your language (index:
<https://learn.microsoft.com/globalization/reference/microsoft-style-guides>)
for address form, capitalization, punctuation, quotes, number/date formats and
spacing. Choices already made:

| Locale | Address | Notes |
|---|---|---|
| fr | *vous* | non-breaking space (`&nbsp;` or `&#8239;`) before `: ; ? !` and inside « » |
| de | *Sie* | „…“ quotes; decimal comma, thousands `.`; `Strg+C`; sentence-style headings |
| es | *tú* (Microsoft Spanish) | ¿…? ¡…!; decimal comma |
| pt-BR | *você* | decimal comma |
| ru | *вы* (lower case) | «…» quotes; decimal comma; no-break space thousands |
| uk | *ви* | «…» quotes; decimal comma |
| pl | per the guide's address section | „…” quotes; decimal comma |
| zh-CN | *你* | full-width punctuation (，。：；（）“”); spacing between Chinese and Latin/digits per the guide |
| ja | です／ます | full-width punctuation (、。「」); spacing between Japanese and Latin/digits per the guide |
| hi | *आप* | Western digits; keep technical terms per glossary |

Preferred renderings of the recurring terms (GPU, API key, prompt, model,
inference, cache hit, prefix cache, KV cache, tensor / data parallel, replica,
beta, coming soon, coding agent, token, time to first byte, air-gapped) are in
`glossary.json` → `preferred`; use them consistently. If one reads wrong in
context, change it there (for your locale) and use the new one everywhere.

## Fonts and layout

No web fonts for CJK or Devanagari: `layout.html` sets `:lang()` system stacks
for `zh-CN`, `ja`, `hi` (Latin stays Mona Sans) with more line height. Mona Sans
has no Cyrillic, so ru/uk Cyrillic falls back glyph by glyph to the system
stack. Buttons and labels may wrap to two lines; nothing is clipped. Check:

```sh
# a server with the site on, then headless Chrome at 1440 and phone widths
../../.venv/bin/python -m pytest -q tests/unit/landing   # first
```

and look at `/<locale>/` at 1440px, 1280px, 390px and 320px (menu open too):
the top bar must not overflow, the hero headline must wrap cleanly.

## When the English changes

Edit the English in `pages/home.html`, `layout.html` or `render.py`
(`PY_SEGMENTS`, `nav_html`, `_BANNER`). New visible text needs a marker
(`data-t="home.<section>.<tag>-<n>"`, or `data-t-alt=` / `data-t-aria-label=`
for attributes; `scripts/landing_i18n.py unmarked FILE` lists what lacks one,
`mark FILE PREFIX` drafts keys). Then `write-source`. Every locale whose
segments changed now fails `check` as **stale** (still served, with the old
translation) until a translation pass refreshes those keys via `todo`.
