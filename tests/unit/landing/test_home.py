"""The redesigned Home (mockup B, "Console-first", owner's choice 2026-10-21),
ported 1:1 from the approved mockup. These pin what the port must not lose:
every section, the install chooser with nothing pre-selected, the carousel's
ten slides and its accessible controls, the lightbox, both cuts of each
drawing, the top bar, and every stand-in link mapped to a real route."""

import json
import re

from fastapi.testclient import TestClient

from app.landing.downloads import MAC_RELEASE_URL
from app.landing.render import REPO_URL, asset_url, inline_scripts
from tests.unit.landing.conftest import all_pages, fetch
from tests.unit.landing.test_routes import CANON, _canonical


def _main(body: str) -> str:
    return re.search(r'<main id="main">.*?</main>', body, re.S).group(0)


def test_sections_in_the_approved_order(client: TestClient) -> None:
    main = _main(fetch(client, "/"))
    ids = [
        'class="hero" aria-labelledby="hero-title"',
        'id="use-cases" aria-roledescription="carousel"',
        'id="install" aria-labelledby="install-title"',
        'id="why" aria-labelledby="why-title"',
        'id="agents" aria-labelledby="agents-title"',
        'id="why-agent" aria-labelledby="tune-title"',
        'id="why-keys" aria-labelledby="share-title"',
        'id="hardware" aria-labelledby="hw-title"',
        'id="features" aria-labelledby="features-title"',
        'id="open-source" aria-labelledby="oss-title"',
    ]
    at = [main.index(i) for i in ids]
    assert at == sorted(at)
    assert '<h1 id="hero-title">Your GPUs, served like a hosted API.</h1>' in main
    assert len(re.findall(r"<h1[\s>]", main)) == 1
    # The ladder holds five problems; coding agents, tuning and sharing are
    # sections of their own (decisions D-B15 to D-B17).
    ladder = main[
        main.index('<ol class="ladder">') : main.index("</ol>", main.index('<ol class="ladder">'))
    ]
    assert re.findall(r'<li id="(why-[a-z]+)">', ladder) == [
        "why-fit",
        "why-cache",
        "why-stats",
        "why-chat",
        "why-god",
    ]


def test_install_chooser_has_four_options_and_no_default(client: TestClient) -> None:
    main = _main(fetch(client, "/"))
    radios = re.findall(r'<input type="radio" name="plat" id="(t-[a-z]+)"([^>]*)>', main)
    assert [r[0] for r in radios] == ["t-linux", "t-mac", "t-win", "t-app"]
    assert not any("checked" in attrs for _, attrs in radios)  # nothing chosen for the visitor
    labels = re.findall(
        r'<label for="t-[a-z]+"[^>]*><b>([^<]+)</b>.*?<span class="st">([^<]+)</span>', main
    )
    assert labels == [
        ("Linux + NVIDIA GPUs", "main release"),
        ("macOS, Apple silicon", "beta"),
        ("Windows", "next"),
        ("Appliance", "coming soon"),
    ]
    for panel in ("p-linux", "p-mac", "p-win", "p-appliance"):
        assert f'<div class="panel-p" id="{panel}">' in main, panel
    assert '<a href="#install-all">show all four</a>' in main
    assert '<div class="panels" id="install-all">' in main
    # Nothing is guessed from the browser either: no script reads the platform.
    for script in inline_scripts(fetch(client, "/")):
        assert "userAgent" not in script and "platform" not in script.lower()


def test_carousel_ten_slides_and_controls(client: TestClient) -> None:
    body = fetch(client, "/")
    main = _main(body)
    slides = re.findall(
        r'<li class="uc-slide" role="group" aria-roledescription="slide" aria-label="(\d+) of 10"><a href="([^"]+)">',
        main,
    )
    assert [int(n) for n, _ in slides] == list(range(1, 11))
    # Every slide links to its block on this page.
    for _, href in slides:
        assert href.startswith("#") and f'id="{href[1:]}"' in main, href
    assert (
        '<div class="uc-ctl" hidden>' in main
        and '<div class="uc-dots" role="group" aria-label="Choose a use case" hidden></div>' in main
    )
    assert 'aria-label="Pause automatic rotation" aria-pressed="false"' in main
    script = next(s for s in inline_scripts(body) if 'querySelector(".uc")' in s)
    # Reading-time interval (never under 6 s), stop for good on manual action,
    # reduced motion never starts, live region only while not rotating.
    assert "Math.max(6000, 1000 + w / 3.5 * 1000)" in script
    assert 'matchMedia("(prefers-reduced-motion: reduce)")' in script
    assert (
        'track.setAttribute("aria-live", (stopped || hovering || focused) ? "polite" : "off")'
        in script
    )
    assert 'e.pointerType === "touch"' in script and "Math.abs(dx) >= 40" in script
    assert ".js .uc-view{overflow:hidden;margin-top:20px;touch-action:pan-y}" in body


def test_every_screenshot_opens_in_the_lightbox(client: TestClient) -> None:
    body = fetch(client, "/")
    main = _main(body)
    zooms = re.findall(r'<a class="zoom" href="([^"]+)"><img src="([^"]+)"[^>]*></a>', main)
    assert len(zooms) == 20  # every product screenshot on the page
    assert len(re.findall(r'<a class="zoom"', main)) == len(zooms)
    for href, src in zooms:
        assert href == src and href.startswith("/_landing/assets/"), href
        assert client.get(href).status_code == 200, href
    assert '<dialog class="lb" aria-labelledby="lb-cap">' in main
    assert '<button type="button" class="lb-close" aria-label="Close enlarged screenshot">' in main
    script = next(s for s in inline_scripts(body) if 'querySelector("dialog.lb")' in s)
    assert "dlg.showModal()" in script and "opener.focus()" in script
    assert "e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1" in script


def test_requests_by_key_figure_ships_with_its_screenshot(client: TestClient) -> None:
    """The per-key Stats chart (owner-provided 2026-10-21) replaced the
    mockup's placeholder: no placeholder box, no hidden figure."""
    main = _main(fetch(client, "/"))
    fig = main[
        main.index('<figure class="bykey">') : main.index(
            "</figure>", main.index('<figure class="bykey">')
        )
    ]
    assert asset_url("shot-requests-by-key.webp") in fig
    assert 'width="2464" height="884"' in fig
    assert 'class="ph"' not in main and "Screenshot pending" not in main
    assert '<figure class="bykey" hidden' not in main


def test_stand_in_links_are_real_routes(client: TestClient) -> None:
    body = fetch(client, "/")
    main = _main(body)
    assert not re.findall(
        r'href="#(?:github|docs|mac|console|faq|tuning-page|client-[a-z-]*)"', body
    )
    for cid in (
        "codex-cli",
        "opencode",
        "aider",
        "grok-cli",
        "openai-sdk",
        "continue",
        "cline",
        "cursor",
    ):
        assert f'<li><a href="/coding-agents/{cid}"><b>' in main, cid
    assert '<a href="/coding-agents">All coding agents</a>' in main
    assert '<a href="/agent-tuning">The agent-tuning page' in main
    assert '<a class="btn go" href="/download/mac">Download for Mac (beta)</a>' in main
    assert f'href="{MAC_RELEASE_URL}"' in main
    assert f'Apache-2.0 on <a href="{REPO_URL}" rel="noopener">GitHub</a>' in main
    assert (
        main.count(f'href="{REPO_URL}/releases" rel="noopener">Watch releases on GitHub</a>') == 2
    )
    # Every in-page fragment on Home names an element on Home.
    for frag in set(re.findall(r'href="/?#([a-z0-9-]+)"', body)):
        assert f'id="{frag}"' in body, frag


def test_top_bar_on_every_page(client: TestClient) -> None:
    for path, body in all_pages(client).items():
        header = body[body.index('<header class="masthead">') : body.index("</header>")]
        assert (
            f'<a class="btn doc" href="{REPO_URL}#readme" rel="noopener" aria-label="Documentation">'
            '<span class="full">Documentation</span><span class="short" aria-hidden="true">Docs</span></a>'
        ) in header, path
        assert '<a class="btn go" href="/#install">Download &amp; Install</a>' in header, path
        assert (
            f'<li class="gh"><a href="{REPO_URL}" rel="noopener" title="Source on GitHub, Apache-2.0">'
            in header
        )
        assert 'GitHub<span class="lic"> · Apache-2.0</span></a>' in header, path
        assert '<li class="signin"><a href="/ui/login">Console sign-in</a></li>' in header, path
        for frag in ("p-linux", "p-mac", "p-win", "p-appliance", "upgrade"):
            assert f'href="/#{frag}"' in header, (path, frag)
        assert '<details class="menu">' in header  # closed: the script opens it from 1280px


def test_structured_data_names_the_platforms_and_features(client: TestClient) -> None:
    _canonical(client, CANON)
    blocks = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>', fetch(client, "/"), re.S
    )
    app = next(d for d in map(json.loads, blocks) if d["@type"] == "SoftwareApplication")
    assert app["operatingSystem"] == "Linux, macOS (beta)"
    assert app["downloadUrl"][0].endswith("/lm-warden/main/install.sh")
    assert app["downloadUrl"][1].endswith(".dmg")
    assert app["installUrl"] == CANON + "/#install"
    assert len(app["featureList"]) >= 8


def test_llms_txt_names_every_platform_and_use_case(client: TestClient) -> None:
    for path in ("/llms.txt", "/llms-full.txt"):
        body = client.get(path).text
        plat = body[body.index("## Platforms") : body.index("## Use cases")]
        for fact in (
            "Linux + NVIDIA GPUs (the main release)",
            "macOS, Apple silicon (beta)",
            "Windows (next)",
            "Appliance (coming soon, no date)",
            "AMD ROCm and Intel GPU support are coming soon",
        ):
            assert fact in plat, (path, fact)
        uses = body[body.index("## Use cases") :]
        for fact in (
            "Will this model fit my GPU?",
            "Coding agents on your own GPUs",
            "56 → 86%",
            "tune your own GPUs",
            "god mode",
            "strict priority",
            "8 × B300",
        ):
            assert fact in uses, (path, fact)
