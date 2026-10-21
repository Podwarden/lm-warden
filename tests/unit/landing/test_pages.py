"""Registry-wide checks over every website page (spec 2026-10-07 §3, §6, §10)."""

import html
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.auth.csrf import _mints_cookie
from app.auth.policy import PUBLIC_ROUTES
from app.landing.pages import PAGES
from tests.unit.landing.conftest import all_pages, fetch

GOLDEN = Path(__file__).parents[2] / "fixtures/landing/landing-a4f06434.html"


def _disable(tmp_data_dir: Path) -> None:
    from tests.unit.landing.test_routes import _disable as off

    off(tmp_data_dir)


def test_expected_pages_exist() -> None:
    paths = {p.path for p in PAGES}
    assert {
        "/",
        "/coding-agents",
        "/agent-tuning",
        "/gpu-monitoring",
        "/cache-aware-routing",
        "/api-keys",
        "/features",
        "/faq",
    } <= paths


def test_titles_and_descriptions_unique_and_short(client: TestClient) -> None:
    titles, descs = set(), set()
    for path, body in all_pages(client).items():
        t = html.unescape(re.search(r"<title>(.*?)</title>", body, re.S).group(1))
        d = html.unescape(re.search(r'<meta name="description" content="([^"]+)"', body).group(1))
        assert len(t) < 60 and len(d) < 160, path
        assert t not in titles and d not in descs, path
        titles.add(t)
        descs.add(d)


def test_one_h1_per_page(client: TestClient) -> None:
    for path, body in all_pages(client).items():
        assert len(re.findall(r"<h1[\s>]", body)) == 1, path


def test_every_internal_link_resolves(client: TestClient) -> None:
    known = {p.path for p in PAGES} | {
        "/ui/login",
        "/llms.txt",
        "/llms-full.txt",
        "/agent-guide.md",
    }
    for path, body in all_pages(client).items():
        for href in re.findall(r'href="(/[^"#?]*)', body):
            if href.startswith("/_landing/assets/") or href.startswith("/ui/icon"):
                continue
            assert href in known or href.startswith("/coding-agents/"), (path, href)


def test_every_page_route_is_public_and_cookieless() -> None:
    for p in PAGES:
        assert ("GET", p.path) in PUBLIC_ROUTES, p.path
        assert not _mints_cookie(p.path), p.path


def test_menu_marks_the_current_page(client: TestClient) -> None:
    for path, body in all_pages(client).items():
        nav = re.search(r'<nav aria-label="Primary">(.*?)</nav>', body, re.S).group(1)
        assert nav.count('aria-current="page"') <= 1, path
        assert '<details class="menu"' in nav  # the phone menu


def test_non_home_pages_show_breadcrumbs(client: TestClient) -> None:
    for path, body in all_pages(client).items():
        if path == "/":
            assert 'aria-label="Breadcrumb"' not in body
        else:
            assert '<nav class="crumbs" aria-label="Breadcrumb">' in body, path


def test_no_accordion_hides_page_content(client: TestClient) -> None:
    """The phone menu is a <details>; nothing inside <main> is."""
    for path, body in all_pages(client).items():
        main = re.search(r"<main\b.*?</main>", body, re.S).group(0)
        assert "<details" not in main, path


def test_trailing_slash_redirects_to_the_canonical_path(client: TestClient) -> None:
    r = client.get("/faq/", follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "/faq"


def test_disabled_site_404s_every_page(client: TestClient, tmp_data_dir: Path) -> None:
    _disable(tmp_data_dir)
    for p in PAGES:
        if p.path != "/":
            assert client.get(p.path).status_code == 404, p.path


def test_old_anchors_map_to_their_new_pages(client: TestClient) -> None:
    home = fetch(client, "/")
    for anchor, page in (
        ("#faq", "/faq"),
        ("#router-title", "/coding-agents/claude-code"),
        ("#cards-title", "/gpu-monitoring"),
        ("#replica-title", "/cache-aware-routing"),
        ("#keys-title", "/api-keys"),
        ("#small-title", "/features"),
    ):
        assert f'"{anchor}": "{page}"' in home, anchor
    assert '"#faq"' not in fetch(client, "/faq")  # the map runs on Home only


def _blocks(body: str) -> list[str]:
    main = re.search(r"<main\b.*?</main>", body, re.S).group(0)
    main = re.sub(r"<!--.*?-->", "", main, flags=re.S)
    text = re.sub(r"\s+", " ", main)
    # A section's first heading may now be its page's <h1>; a link to a
    # section now on Home reads "/#id" instead of "#id".
    text = re.sub(r"<h1\b", "<h2", text).replace("</h1>", "</h2>")
    text = text.replace('href="/#', 'href="#')
    tags = r"p|h2|h3|li|figcaption|td|th|dt|dd"
    return [m.strip() for m in re.findall(rf"<(?:{tags})\b[^>]*>(.*?)</(?:{tags})>", text)]


#: Blocks of the pre-split page that were deliberately rewritten, with why.
REWRITTEN = {
    # Hero lede named only Claude Code; now names the other agents too (spec §3).
    "LM Warden runs vLLM and llama.cpp on your NVIDIA cards behind one port. Each app gets "
    "its own key, and each card reports for itself. Point Claude Code at it, and a model on "
    "your cards answers for the Claude models you choose, under their names, while the rest "
    "go to Anthropic on your own login.": "multi-agent hero, spec §3",
    # Promoted to the GPU page's <h1> with the topic in front, for search.
    "The box behind these numbers.": "GPU monitoring: the box behind these numbers.",
    # Engine-bump: the features page's source-of-truth sentence was rewritten
    # (the old one named the shipped version explicitly; the new one does not).
    'Checked against vLLM&nbsp;0.20.0, before the engine moved to 0.26.0, the version this release ships and the replica measurements above used. The rules have held across those upgrades; the exact numbers are worth rechecking on your engine. The full table and the reasons behind it are in <a href="https://github.com/Podwarden/lm-warden/blob/main/documents/HAZARDS.md#what-fits-on-what" rel="noopener">HAZARDS.md</a>. Not on NVIDIA? <a href="#roadmap">Other GPUs are on the roadmap</a>.': "engine-bump: features.html source sentence rewritten to drop the version pin",
    # Tokens-per-second rename: the chart now plots tokens per second; the
    # Sep 23 capture row is labelled as the old per-minute panel.
    "Tokens per minute": "tokens-per-second rename, audit item E",
    "VRAM in use, GPU utilisation (the busiest card), power (the sum of the cards) and tokens per minute": "tokens-per-second rename, audit item E",
    # Spill-threshold fix: the landing said 8 was a constant; the default is a
    # quarter of max_num_seqs, 8 is the auto value on the four-card box.
    "An agent resends its whole history every turn, so a long conversation is mostly a prefix the engine has already read. vLLM keeps that work in its prefix cache (the attention state of prompts it has seen) and skips it the next time the same prefix arrives, but only on the replica that did the work: each full copy of the model has its own cache. vLLM’s own balancer sends each turn to whichever replica has the shortest queue, which usually has never seen the conversation, so it reads the whole history again. LM Warden places each new conversation on the least-loaded replica (requests in flight, how full its cache is, and how many live conversations it holds) and keeps it there, using the session id Claude Code already sends. It moves a request only when that replica has 8 requests in flight (the spill threshold).": "spill-threshold fix, audit item G",
    # Third lesson note added: the heading now says "Three things".
    "Two things we learned the hard way.": "third lesson note added, audit item C",
    # Engine versions added to the FAQ answer for consistency with the lede.
    "Mainline vLLM and llama.cpp, chosen per model and run unmodified: no patched image, no fork, no monkeypatching. Picking a <code>.gguf</code> file selects llama.cpp, which fits larger models onto smaller cards with GGUF quantisation and keeps older GPUs useful. TensorRT-LLM, SGLang, MLX and Ollama’s runtime are not shipped.": "engine versions added to FAQ, audit item 5",
    # BACKLOG-macos P9: Apple silicon moved from the roadmap to "Mac app in
    # development, no download yet" (home roadmap div, no-NVIDIA item, GPU FAQ).
    "<strong>You have no NVIDIA GPU.</strong> Today it runs on NVIDIA cards with CUDA only. There is no ROCm, Apple silicon, Intel GPU or CPU path yet; see the roadmap below.": "Mac app in development, BACKLOG-macos P9",
    "Other GPUs are next on the roadmap.": "Mac app in development, BACKLOG-macos P9",
    "Apple silicon (Macs), AMD Radeon and Intel GPUs are on our immediate roadmap, with no dates. Which comes first depends on who asks, so tell us which one you would run it on:": "Mac app in development, BACKLOG-macos P9",
    "NVIDIA GPUs on a Linux x86_64 host. The floor is Turing (sm_75, the RTX 20-series generation): the base image ships CUDA 13, which dropped Maxwell, Pascal and Volta, so a GTX 1080 Ti, a Titan X or a V100 will not work. Turing through Blackwell is covered, including Ada, Hopper and the RTX 50-series. There is no ROCm, Apple Silicon, Intel GPU or CPU path today; Apple silicon, AMD Radeon and Intel GPUs are on the immediate roadmap, and which comes first depends on the feedback we get in the project’s GitHub Discussions. Several GPUs in one box are supported; one model split across several machines is not.": "Mac app in development, BACKLOG-macos P9",
}


def test_nothing_from_the_single_page_was_lost(client: TestClient) -> None:
    """Spec §1: reorganise, do not remove. Every heading, paragraph, list
    item, caption, table cell and definition of the pre-split page appears
    on some page of the site."""
    old = _blocks(GOLDEN.read_text())
    new = " ".join(" ".join(_blocks(b)) for b in all_pages(client).values())
    new = re.sub(r"\s+", " ", new)
    missing = [b for b in old if b and b not in new and b not in REWRITTEN]
    assert not missing, missing[:5]
