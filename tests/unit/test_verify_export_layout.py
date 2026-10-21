"""Layout check for the frontend static export (plan-8 p4d).

``scripts/macos/verify-export-layout.sh`` inspects a built ``frontend/out/``
tree and fails when it does not match what ``app/webui`` mounts at ``/ui``:
``index.html`` at the root, ``_next/static/`` present, no ``ui/`` directory
level (basePath must not become a path component on disk), every
``/ui/_next/...`` src/href in every html file resolving on disk, no bare
``/_next/...`` reference (basePath not baked), and no ``api/`` server-route
artifact.

Run directly from Linux; no macOS tools needed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "macos" / "verify-export-layout.sh"

OK_LINE = "export layout ok"


def _run_script(out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(out_dir)],
        capture_output=True,
        text=True,
    )


def _write_asset_layout(out: Path) -> tuple[Path, Path]:
    """Write the minimal valid export layout: index.html + _next/static file."""
    index = out / "index.html"
    index.write_text(
        "<html><head>"
        '<link href="/ui/_next/static/css/app.css" rel="stylesheet"/>'
        '<script src="/ui/_next/static/main.js"></script>'
        "</head><body></body></html>\n"
    )
    css = out / "_next" / "static" / "css" / "app.css"
    css.parent.mkdir(parents=True)
    css.write_text("body{}\n")
    js = out / "_next" / "static" / "main.js"
    js.write_text("// main\n")
    return index, js


def test_good_layout_passes_and_reports_counts(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _write_asset_layout(out)

    res = _run_script(out)
    assert res.returncode == 0, f"rc={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got {lines!r}"
    assert lines[0].startswith(OK_LINE), f"unexpected ok line: {lines[0]!r}"
    # 1 html, 2 assets (app.css + main.js).
    assert "1 html" in lines[0], lines[0]
    assert "2 assets" in lines[0], lines[0]


def test_missing_index_html_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "_next").mkdir()
    (out / "_next" / "static").mkdir()

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "index.html" in res.stdout, res.stdout


def test_missing_next_static_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text("<html></html>\n")

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "_next/static" in res.stdout, res.stdout


def test_ui_directory_level_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _write_asset_layout(out)
    (out / "ui").mkdir()

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "ui/" in res.stdout, res.stdout


def test_api_directory_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    _write_asset_layout(out)
    (out / "api").mkdir()

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "api/" in res.stdout, res.stdout


def test_unreferenced_missing_asset_passes(tmp_path: Path) -> None:
    # An html that references nothing: only the structural rules apply.
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text("<html><body></body></html>\n")
    (out / "_next").mkdir()
    (out / "_next" / "static").mkdir()

    res = _run_script(out)
    assert res.returncode == 0, f"rc={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
    assert "1 html" in res.stdout
    assert "0 assets" in res.stdout


def test_missing_referenced_asset_fails_naming_it(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        '<html><script src="/ui/_next/static/missing.js"></script></html>\n'
    )
    (out / "_next").mkdir()
    (out / "_next" / "static").mkdir()

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "_next/static/missing.js" in res.stdout, res.stdout


def test_referenced_asset_present_passes(tmp_path: Path) -> None:
    # The bound: the same html as test_missing_referenced_asset_fails_naming_it
    # with the referenced file now on disk must pass.
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        '<html><script src="/ui/_next/static/missing.js"></script></html>\n'
    )
    static = out / "_next" / "static"
    static.mkdir(parents=True)
    (static / "missing.js").write_text("// here\n")

    res = _run_script(out)
    assert res.returncode == 0, f"rc={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
    assert "1 html" in res.stdout
    assert "1 assets" in res.stdout


def test_href_attribute_is_checked_too(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        '<html><head><link href="/ui/_next/static/gone.css"/></head></html>\n'
    )
    (out / "_next" / "static").mkdir(parents=True)

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "gone.css" in res.stdout, res.stdout


def test_non_ui_prefix_reference_fails(tmp_path: Path) -> None:
    # basePath not baked: a bare /_next/... reference would 404 behind the
    # /ui mount.
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text('<html><script src="/_next/static/main.js"></script></html>\n')
    static = out / "_next" / "static"
    static.mkdir(parents=True)
    (static / "main.js").write_text("// main\n")

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "_next/static/main.js" in res.stdout, res.stdout


def test_non_ui_non_next_reference_is_ignored(tmp_path: Path) -> None:
    # Only /ui/_next/... and bare /_next/... references are checked; other
    # absolute refs (e.g. the /ui page links themselves) are not.
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text('<html><a href="/ui/models/detail?id=7">m</a></html>\n')
    (out / "_next" / "static").mkdir(parents=True)

    res = _run_script(out)
    assert res.returncode == 0, f"rc={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"


def test_multiple_problems_reported_one_per_line(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        "<html>"
        '<script src="/ui/_next/static/one.js"></script>'
        '<script src="/ui/_next/static/two.js"></script>'
        "</html>\n"
    )
    (out / "_next" / "static").mkdir(parents=True)
    (out / "ui").mkdir()

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    # Both missing assets and the ui/ dir must be named, one per line.
    assert any("one.js" in line for line in lines), lines
    assert any("two.js" in line for line in lines), lines
    assert any("ui/" in line for line in lines), lines
    assert len(lines) >= 3, lines


def test_two_refs_on_one_line_both_checked(tmp_path: Path) -> None:
    # Both references on the same html line; only the second is missing.
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        '<html><script src="/ui/_next/static/a.js"></script>'
        '<script src="/ui/_next/static/b.js"></script></html>\n'
    )
    static = out / "_next" / "static"
    static.mkdir(parents=True)
    (static / "a.js").write_text("// a\n")

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    assert any("b.js" in line for line in lines), lines
    assert not any("a.js" in line for line in lines), lines


def test_href_before_src_on_one_line_is_checked(tmp_path: Path) -> None:
    # A Next export is one long line with <link href> before <script src>:
    # the href must not be skipped because a src follows it.
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        '<html><link href="/ui/_next/static/gone.css"/>'
        '<script src="/ui/_next/static/a.js"></script></html>\n'
    )
    static = out / "_next" / "static"
    static.mkdir(parents=True)
    (static / "a.js").write_text("// a\n")

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    assert any("gone.css" in line for line in lines), lines
    assert not any("a.js" in line for line in lines), lines


def test_good_layout_with_its_css_removed_fails(tmp_path: Path) -> None:
    # The good layout's href (before its src) is really checked.
    out = tmp_path / "out"
    out.mkdir()
    _write_asset_layout(out)
    (out / "_next" / "static" / "css" / "app.css").unlink()

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "app.css" in res.stdout, res.stdout


def test_dot_dot_ref_escaping_the_export_fails(tmp_path: Path) -> None:
    # The target exists on disk outside <out>; the ref still must fail.
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "secret.js").write_text("// outside\n")
    (out / "index.html").write_text(
        '<html><script src="/ui/_next/../../secret.js"></script></html>\n'
    )
    (out / "_next" / "static").mkdir(parents=True)

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "/ui/_next/../../secret.js" in res.stdout, res.stdout


def test_dot_dot_ref_staying_inside_the_export_fails(tmp_path: Path) -> None:
    # Resolves to <out>/index.html, outside _next/: still a dot segment.
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        '<html><script src="/ui/_next/../index.html"></script></html>\n'
    )
    (out / "_next" / "static").mkdir(parents=True)

    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "/ui/_next/../index.html" in res.stdout, res.stdout


def test_query_and_fragment_are_not_part_of_the_file(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "index.html").write_text(
        '<html><script src="/ui/_next/static/a.js?dpl=abc"></script>'
        '<link href="/ui/_next/static/b.css#x"/></html>\n'
    )
    static = out / "_next" / "static"
    static.mkdir(parents=True)
    (static / "a.js").write_text("// a\n")
    (static / "b.css").write_text("b{}\n")

    res = _run_script(out)
    assert res.returncode == 0, f"rc={res.returncode}\nstdout={res.stdout}\nstderr={res.stderr}"
    assert "1 html, 2 assets" in res.stdout, res.stdout

    (static / "a.js").unlink()
    res = _run_script(out)
    assert res.returncode == 1, f"rc={res.returncode}, stderr={res.stderr}"
    assert "a.js?dpl=abc" in res.stdout, res.stdout


def test_missing_out_dir_exits_2(tmp_path: Path) -> None:
    res = _run_script(tmp_path / "does-not-exist")
    assert res.returncode == 2, f"rc={res.returncode}, stderr={res.stderr}"
