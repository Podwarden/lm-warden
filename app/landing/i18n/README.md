# Translating the website

Every public page is translated: Home, `/coding-agents` and each
`/coding-agents/<client>` page, `/agent-tuning`, `/gpu-monitoring`,
`/cache-aware-routing`, `/api-keys`, `/features`, `/faq`, `/privacy` and the
404 page, with the chrome on all of them (top bar, menu, language switcher,
breadcrumbs, footer, cookie banner, copy labels, `<title>`, meta description,
og tags, JSON-LD). `llms.txt`, `/agent-guide.md`, the console UI, code
snippets and the tuning prompts stay English.

English is the single source of truth. Translations are generated artifacts:
produce them with the pipeline below, never hand-edit a catalog outside it.

## Parts and files

The text is split into **parts**, one catalog each. A segment's key starts
with its part (Home's with `chrome.`, `home.`, `meta.` or `jsonld.`).

| Part | Segments | Pages that need it |
|---|---|---|
| `home` | 282 | every page (Home, and the chrome all pages share) |
| `common` | 25 | every page but Home (breadcrumbs, "Copy", client-page headings, support labels) |
| `coding-agents` | 10 | `/coding-agents` |
| `clients` | 105 | every `/coding-agents/<client>` page (prose from `app/connect/catalog.py`) |
| `claude-code-router` | 17 | `/coding-agents/claude-code` (with `clients`) |
| `agent-tuning` | 38 | `/agent-tuning` |
| `gpu-monitoring` | 49 | `/gpu-monitoring` |
| `cache-aware-routing` | 33 | `/cache-aware-routing` |
| `api-keys` | 13 | `/api-keys` |
| `features` | 46 | `/features` |
| `faq` | 28 | `/faq` (also its FAQ JSON-LD) |
| `privacy` | 16 | `/privacy` (only with a Google tag configured) |
| `not-found` | 4 | the 404 page under `/<locale>/...` |

| File | What it is |
|---|---|
| `en.json` | Source catalog of part `home`: `{key: {text, note, hash}}`, written by `scripts/landing_i18n.py write-source`. Never edit by hand. |
| `en/<part>.json` | Source catalog of every other part, the same way. |
| `<locale>.json` | Translation of part `home`: `{key: {text, source_hash}}`. `source_hash` is the source's `hash` of the English you translated. |
| `<locale>/<part>.json` | Translation of every other part. `ru/` is the worked example. |
| `glossary.json` | Do-not-translate terms, preferred renderings per locale, and per-locale allowlists. |

Locales and their URLs (`app/landing/pages.py`, `LOCALES`): English has no
prefix; `fr` `de` `es` `pt-BR` `ru` `uk` `pl` `zh-CN` `ja` `hi` serve
`/<tag>/` (Home) and `/<tag><path>` (`/ru/faq`, `/ru/coding-agents/aider`).

**A page is published in a locale when every part it needs is complete
there** (Home: `home`; the others: `home` + `common` + their own). Then its
URL answers, and the switcher, hreflang and sitemap offer it; until then it
404s and the page simply stays English in that language (links to it from
translated pages go to the English page). So the site ships with partial
coverage: commit a part only when it is complete and its checks pass (the
tests fail on a committed part file with a missing or broken key).

## How to translate one locale (the hand-off)

You add files for exactly one locale: `app/landing/i18n/<locale>/<part>.json`
(and, if Home changed, `<locale>.json`). Nothing else, except your locale's
lists in `glossary.json` when an allowance below is needed.

1. See what is left: `../../.venv/bin/python scripts/landing_i18n.py check <locale>`
   and `... pages` (which pages each locale publishes, and which parts hold
   the rest back).
2. Take one part at a time, `common` first (every page needs it):
   `../../.venv/bin/python scripts/landing_i18n.py todo <locale> common > /tmp/todo.json`
   (all its segments for a new part; only missing or stale ones for a
   refresh; `todo <locale>` with no part gives every part's, merged).
3. Translate each `text`, guided by its `note` (where it appears; a length
   limit for labels) and `glossary.json`. Write
   `{key: {"text": <translation>, "source_hash": <the segment's hash>}}`
   for every key of the part, in source order, to `<locale>/<part>.json`.
4. Check: `../../.venv/bin/python scripts/landing_i18n.py check <locale> --part <part>`
   must print `complete`; then `../../.venv/bin/python -m pytest -q tests/unit/landing`.
5. Look at the pages (below), then commit (one commit per locale, or per part).

### Rules a translation must keep (the check enforces the first five)

- **Tags and placeholders.** Inline HTML (`<b>`, `<code>`, `<em>`, `<a href="…" rel="noopener">`, `<span class="…">`) and tokens (`{{repo}}`, `{{mac_release}}`, `{{docs_link}}`, `{{docs_date}}`, `{{privacy_ads}}`, `{n}`, `{m}`, `{name}`, `{summary}`) must all survive, attributes byte-identical. You may move them to fit the sentence; never drop, add or edit one.
- **Links keep their English href** (`<a href="/faq">`): the server points every link to a site page at the same page in the reader's language where it is published, and leaves it English where it is not.
- **`<code>` spans verbatim**: commands, paths, env vars, endpoints.
- **Numbers keep their value**; separators follow the locale (`2.05` → `2,05` in de/fr/ru/uk/pl/es/pt-BR; thousands `11,178` → `11.178` de, `11 178` fr/ru/uk/pl with a no-break space). Units keep their symbols (GB, GiB, W, s, h). Use `&#8239;` (narrow no-break space) or `&nbsp;` between a number and its unit or `%` where your language puts a space.
- **Do-not-translate terms** (`glossary.json` `do_not_translate`) appear exactly as written whenever the English has them: product and brand names, model ids, and the console's own UI names (`Stats`, `Add model`, `Share on Local Network`, `Connect`, `God mode`, …) because the console itself is English. Do not glue them into compounds with hyphens (`LM-Warden-Server` breaks the check; write it so the name stands alone).
- **Attribute segments** (keys ending `.alt`, `.aria-label`, `.title`, `.data-h`, `.resume`, `.dot`, `.done`, `.fail`, and `meta.*` / `<part>.meta.*` / `<part>.jsonld.*`) are plain text: no `"` and no `<`.
- **Client pages** (`clients.*`): the catalogue's prose with every setting, env var, path, command and flag already in `<code>` (keep them verbatim). Names of the clients' own menus and settings (`API Provider`, `OpenAI Compatible`, `Cursor Settings`, `Override OpenAI Base URL`) and of the console's (`Settings → Access Tokens`, `Router`, `Stats`, `In flight`, `May relay`) stay English, as on screen. `privacy.ads` and `privacy.contact-*` start with a space.
- **Nothing left in English by accident.** A segment you deliberately keep identical (e.g. `FAQ`, `APIs`, a row of config names) goes in `glossary.json` → `identical_ok` → your locale. Likewise `numbers_ok` (digits legitimately differ, or the checker joins two numbers: "bf16 8B") and `glossary_ok` (a do-not-translate term used as an ordinary word, e.g. "Connect" as a verb).
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

and look at `/<locale>/` and the pages you translated (at least
`/<locale>/coding-agents`, `/<locale>/coding-agents/claude-code`,
`/<locale>/faq`, `/<locale>/cache-aware-routing`) at 1440px, 1280px, 390px
and 320px (menu open too): the top bar must not overflow, headlines and
table cells must wrap cleanly. Then the click path: `/<locale>/` → Coding
agents → the switcher's English → the same page in English → your language
→ back.

## How a visitor gets a language

- `?lang=<tag>` on any page (the switcher's links, each to the same page in
  that language) sets the `lw_lang` cookie for a year; it serves the page if
  it is already in `<tag>`, else 302s to the same page in `<tag>`.
- The English Home sends a browser to `/<locale>/` by the cookie, else
  Accept-Language. Every other English page honours the cookie only, and
  only where that page is published in it. Crawlers and HEAD are never
  redirected; `?lang=en` always keeps a visitor on English.

## When the English changes

Edit the English in `pages/*.html`, `layout.html`, `render.py`
(`PY_SEGMENTS`, `PART_PY_SEGMENTS`, `nav_html`, `crumbs_html`, `_BANNER`),
`clients.py` / `diagrams.py`, or the catalogue prose in
`app/connect/catalog.py` (summaries, capability, requirements, mode titles
and descriptions). New visible text in a fragment needs a marker
(`data-t="<part>.<tag>-<n>"`, or `data-t-alt=` / `data-t-aria-label=` /
`data-t-data-h=` for attributes; `scripts/landing_i18n.py unmarked FILE` lists
what lacks one, `mark FILE PREFIX` drafts keys). Then `write-source`. Every locale whose
segments changed now fails `check` as **stale** (still served, with the old
translation) until a translation pass refreshes those keys via `todo`.

## Translation QA

The `check` above proves a translation is *well-formed* (tags, code, numbers,
glossary). It cannot tell whether it says the same thing, or reads like a
native wrote it. Every published locale therefore goes through QA (owner,
2026-10-21: "translate back and forth, see if back translation shows the same
meaning, use other approaches to control quality, for all languages"), and
the result is recorded in `qa/<locale>/report.json` (+ a human `report.md`).

### Method

1. **Blind back-translation.** An independent agent sees ONLY the target
   text (never the English) and renders each segment literally into English,
   adding a `"<key>#note"` entry where the target is unclear or unnatural.
2. **Monolingual native-editor review.** Another agent, also target only,
   flags fluency, grammar, terminology, style and locale-convention issues
   (quotes, spacing, decimal separators, address form per the style table
   above) with a suggested rewrite, plus a terminology-consistency table.
3. **Bilingual meaning review.** Compares English vs back-translation (and
   the target) per segment; flags accuracy issues (omission, addition,
   mistranslation, changed number or claim, left untranslated) with severity.
4. **Fix.** Apply the rewrites (`apply`); re-run the blind back-translation
   on the changed segments only; repeat until 0 critical and 0 major.
5. **Score**, MQM-style: penalty points minor = 1, major = 5, critical = 25
   per 1,000 English words covered. Issues marked `fixed` are listed but do
   not score; `open` and `accepted` ones do.

**Pass rule:** 0 critical and 0 major issues remaining, MQM score <= 10,
every published segment back-translated, and no stale translation.

**Independence rules:** the back-translator never sees the English; the
native editor never sees the English; the back-translation runs on a
different model from the fixer (`record` refuses when `--back-model` equals
`--fix-model`). Each agent is a fresh session that has not seen the others'
output except where its brief hands it over.

Severity: **critical** = the reader would act wrongly or the claim is false
(wrong number, command, price, negation, a promise the English does not
make); **major** = meaning lost or changed, a sentence a native would stumble
on, a wrong term for a product concept; **minor** = style, punctuation,
slightly unnatural but correct wording.

### Commands

`scripts/landing_i18n_qa.py` (run from the repo root with `../../.venv/bin/python`):

```sh
Q="../../.venv/bin/python scripts/landing_i18n_qa.py"
W=/tmp/qa-{loc}                                         # any scratch dir
$Q extract {loc} --out $W                               # {loc}.target.json, {loc}.source.json (published keys)
#   -> back-translator writes $W/back.json ; native editor writes $W/editor.json
$Q diff {loc} --back $W/back.json --out $W/diff.md      # + diff.json, for the meaning reviewer
#   -> meaning reviewer + fixer writes $W/fixes.json and $W/issues.json
$Q apply {loc} --fixes $W/fixes.json                    # writes the catalogs, then runs check {loc}
$Q extract {loc} --out $W/r1 --keys $W/fixes.json       # changed segments only, for the re-verifier
#   -> re-verifier writes $W/back-r1.json and updates issues.json
$Q record {loc} --issues $W/issues.json --back $W/back.json --back $W/back-r1.json \
   --back-model <model> --fix-model <model>             # qa/{loc}/report.json + report.md
$Q status                                               # every locale: published, covered, stale, score, pass
```

File formats: back-translation `{key: "literal English", "<key>#note":
"..."}` (later `--back` files override earlier ones); fixes `{key: "new
target text"}` (refused as a whole when a key is unknown or a fix breaks
tags, code, numbers or glossary terms); issues a list of `{"key",
"category", "severity", "status", "note"}` with category one of `omission`,
`addition`, `mistranslation`, `number`, `untranslated`, `terminology`,
`fluency`, `grammar`, `style`, `locale-convention`; severity `minor` /
`major` / `critical`; status `open` / `fixed` / `accepted`.

`tests/unit/landing/test_i18n_qa.py` fails for a locale whose report did not
pass or no longer covers what it publishes (a segment's English or
translation changed since, or a newly published part): "re-run translation
QA for {loc}". A refresh only needs the stale/new segments put through the
loop, then `record` again with all back-translation files. A published locale
with no report is only a warning while QA is rolled out; flipping
`REQUIRE_QA_REPORT = True` in that test makes it a failure.

### Agent briefs (copy; replace `{loc}`, `{language}`, `{W}` and, in brief 2, `{address form}` from the style table)

**1. Blind back-translator** (a different model from the fixer; give it
only `{W}/{loc}.target.json`):

```text
You are a professional translator from {language} into English. Read {W}/{loc}.target.json:
a JSON object of website text segments (key -> {language} text) from a product site for a
self-hosted LLM gateway. You do NOT have, and must not look for, any English original; do not
open any other file in the repository.

Translate every segment into English as LITERALLY as is still grammatical: keep the
{language} sentence structure, word choice, hedges, numbers, units and emphasis, even where it
sounds odd in English. Do not improve, smooth or guess what the author "meant". Keep inline
HTML tags, {{tokens}}, {n}-style placeholders and <code> spans exactly as they are.

Where the {language} is unclear, ambiguous, unnatural, grammatically wrong, mixes languages
oddly or uses a term a native speaker would not, add an entry "<key>#note": "<what is odd
and why>". Write {W}/back.json: {key: "English", "<key>#note": "..."} covering every key, in
the same order. Report only the count of segments and notes.
```

**2. Monolingual native editor** (sees only the target):

```text
You are a native {language} copy editor for software documentation and product websites,
following Microsoft's {language} localization style guide. Read {W}/{loc}.target.json
(key -> {language} text of a product site for a self-hosted LLM gateway). You do not have the
English and must not look for it.

Review every segment as a native reader would: fluency and grammar, natural word order,
terminology (the right and consistent term for GPU, API key, prompt, model, inference, token,
cache, replica, coding agent, ...), register and address form ({address form}), punctuation,
quotes, spacing between scripts, number/decimal/thousands format and units. Product names,
model ids, commands, paths and English UI names (Stats, Add model, Settings → Access Tokens)
are deliberately left in English: do not flag them. Do not touch tags, {{tokens}}, {n} or
<code> spans.

Write {W}/editor.json: {"issues": [{"key", "category" (fluency|grammar|terminology|style|
locale-convention), "severity" (minor|major|critical), "note", "rewrite": "<full new segment
text>"}], "terminology": [{"concept", "renderings_seen": {"<rendering>": [keys]},
"recommended"}]}. Flag only real problems; a clean segment gets no entry.
```

**3. Bilingual meaning reviewer + fixer** (sees English, target, back-translation, editor notes):

```text
You are a bilingual English/{language} reviewer. Inputs: {W}/diff.md (or diff.json): per
segment the English source, the {language} target, an independent blind back-translation and
the back-translator's notes; {W}/editor.json from a native editor who did not see the English;
the rules in app/landing/i18n/README.md and app/landing/i18n/glossary.json.

1. For every segment compare the English with the back-translation (and the target): flag
   omission, addition, mistranslation, number (a changed number, unit or claim) and
   untranslated text. A literal back-translation that sounds clumsy is not an error when the
   meaning is intact. Severity: critical = a reader would act wrongly or a claim becomes false;
   major = meaning lost or changed, or a native would stumble; minor = style.
2. Judge each editor issue against the English (an editor rewrite must not change meaning)
   and keep the valid ones.
3. Write {W}/fixes.json: {key: "<full corrected {language} text>"} for every critical and
   major issue and every minor one worth fixing; keep tags, {{tokens}}, {n}, <code> spans and
   do-not-translate terms byte-identical. Run:
   ../../.venv/bin/python scripts/landing_i18n_qa.py apply {loc} --fixes {W}/fixes.json
   and correct anything it refuses until the check is clean.
4. Write {W}/issues.json: a list of {"key", "category", "severity", "status", "note"} for
   every issue found by you or the editor; status "fixed" for those in fixes.json, "accepted"
   for a minor one deliberately left, "open" otherwise. Then extract the changed segments:
   ../../.venv/bin/python scripts/landing_i18n_qa.py extract {loc} --out {W}/r1 --keys {W}/fixes.json
Do not touch any other locale's files, en*.json or glossary.json.
```

**4. Re-verifier** (a fresh blind back-translator, same model family as brief 1, then a
meaning check):

```text
Step A (blind): run brief 1 on {W}/r1/{loc}.target.json only, writing {W}/back-r1.json.
Step B (a separate agent, after A): read {W}/r1/{loc}.source.json (English), {W}/back-r1.json
and {W}/issues.json. For each "fixed" issue confirm the new back-translation now carries the
English meaning; if not, set its status back to "open" with a note and raise any new issue the
fix introduced. If anything critical or major is open, hand it back to brief 3 (round r2, ...).
Otherwise record:
../../.venv/bin/python scripts/landing_i18n_qa.py record {loc} --issues {W}/issues.json \
  --back {W}/back.json --back {W}/back-r1.json --back-model <back model> --fix-model <fix model>
It exits 0 on PASS. Commit app/landing/i18n/qa/{loc}/ together with the fixed catalogs.
```
