"""Keep requirements-macos.{in,txt} consistent with requirements.txt.

requirements-macos.txt is the hash-locked dependency set for the Python
embedded in the macOS app bundle (BACKLOG-macos P8; Metal design §3). It is
generated from requirements-macos.in by scripts/macos/lock-requirements.sh
(`make macos-requirements`). requirements.txt is the Linux source of truth:
a pin bumped there must land in the macOS files too, or the two products
drift onto different FastAPI/uvicorn/... versions.

Pure file parsing — runs on Linux in the normal unit suite, no network.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LINUX_REQS = REPO_ROOT / "requirements.txt"
MACOS_IN = REPO_ROOT / "requirements-macos.in"
MACOS_LOCK = REPO_ROOT / "requirements-macos.txt"
LOCK_SCRIPT = REPO_ROOT / "scripts" / "macos" / "lock-requirements.sh"
MACOS_MK = REPO_ROOT / "make" / "macos.mk"

# requirements.txt packages deliberately absent from the bundle, with the reason.
MACOS_EXCLUDED = {
    # Docker cannot pass Metal into a container; `import docker` is lazy inside
    # _build_engine_driver, so omitting it is safe (Metal design §9).
    "docker": "Linux sibling-container engine driver only",
}

# macOS-only packages the bundle must carry.
MACOS_REQUIRED = {
    "mlx-lm",  # the MLX engine (mlx_lm.server)
    "pyobjc-framework-metal",  # MTLDevice.recommendedMaxWorkingSetSize probe
    "transformers",  # app/proxy/tokenizers.py; on Linux it comes from the vllm base image
    "huggingface-hub",  # app/models/discovery.py, pull_task.py; same
}

_REQ_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"\s*(?P<spec>[^\s;\\#]*)"
)


def _canon(name: str) -> str:
    """PEP 503 normalised project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse(path: Path) -> dict[str, tuple[frozenset[str], str]]:
    """Map canonical name -> (extras, specifier) for each requirement line.

    Skips comments, blank lines, `-r`/`--hash` option lines and the indented
    continuation lines of a hash-locked file.
    """
    out: dict[str, tuple[frozenset[str], str]] = {}
    for raw in path.read_text().splitlines():
        line = raw.split(" #", 1)[0].rstrip()
        if not line or line[0] in "#- \t":
            continue
        m = _REQ_RE.match(line)
        assert m, f"{path.name}: unparseable requirement line {raw!r}"
        extras = frozenset(_canon(e) for e in (m["extras"] or "").split(",") if e.strip())
        name = _canon(m["name"])
        assert name not in out, f"{path.name}: {name} listed twice"
        out[name] = (extras, m["spec"])
    return out


def _pin(spec: str) -> str | None:
    return spec[2:] if spec.startswith("==") and "," not in spec else None


def test_every_linux_pin_is_repeated_verbatim_in_the_macos_inputs():
    linux = _parse(LINUX_REQS)
    macos_in = _parse(MACOS_IN)
    for name, (extras, spec) in linux.items():
        if name in MACOS_EXCLUDED:
            assert (
                name not in macos_in
            ), f"{name} is excluded from macOS but listed in {MACOS_IN.name}"
            continue
        assert name in macos_in, (
            f"{name}{spec} is in requirements.txt but missing from {MACOS_IN.name}; "
            f"copy the pin (or add it to MACOS_EXCLUDED with a reason) and re-lock"
        )
        assert macos_in[name] == (extras, spec), (
            f"{name}: requirements.txt has extras={sorted(extras)} {spec}, "
            f"{MACOS_IN.name} has extras={sorted(macos_in[name][0])} {macos_in[name][1]}"
        )


def test_every_shared_package_locks_the_same_version_as_linux():
    linux = _parse(LINUX_REQS)
    lock = _parse(MACOS_LOCK)
    for name, (_extras, spec) in linux.items():
        if name in MACOS_EXCLUDED:
            assert (
                name not in lock
            ), f"{name} is excluded from macOS but present in {MACOS_LOCK.name}"
            continue
        pin = _pin(spec)
        assert pin, f"requirements.txt: {name} is not an exact == pin ({spec!r})"
        assert name in lock, f"{name} missing from {MACOS_LOCK.name}; run make macos-requirements"
        assert _pin(lock[name][1]) == pin, (
            f"{name}: requirements.txt pins {pin}, {MACOS_LOCK.name} locks "
            f"{lock[name][1]} — re-run make macos-requirements"
        )


def test_lock_covers_every_macos_input_and_honours_its_pins():
    macos_in = _parse(MACOS_IN)
    lock = _parse(MACOS_LOCK)
    for name, (_extras, spec) in macos_in.items():
        assert (
            name in lock
        ), f"{name} is in {MACOS_IN.name} but not locked; run make macos-requirements"
        pin = _pin(spec)
        if pin:
            assert (
                _pin(lock[name][1]) == pin
            ), f"{name}: {MACOS_IN.name} pins {pin}, lock has {lock[name][1]}; re-lock"


def test_macos_only_packages_are_inputs_and_locked():
    macos_in = _parse(MACOS_IN)
    lock = _parse(MACOS_LOCK)
    for name in MACOS_REQUIRED:
        assert name in macos_in, f"{name} must be a direct input in {MACOS_IN.name}"
        assert name in lock, f"{name} must be locked in {MACOS_LOCK.name}"


def test_lock_is_exact_and_hash_locked():
    """Every locked requirement is `==` with at least one sha256 hash, so the
    bundle build can install with --require-hashes."""
    text = MACOS_LOCK.read_text()
    assert text.startswith(
        "# GENERATED by scripts/macos/lock-requirements.sh"
    ), f"{MACOS_LOCK.name} must be produced by the lock script, not edited by hand"
    # Split into one block per requirement: a block starts at a non-indented,
    # non-comment line and runs until the next one.
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    assert blocks, f"{MACOS_LOCK.name} has no requirements"
    for block in blocks:
        head = block[0]
        m = _REQ_RE.match(head)
        assert m and _pin(m["spec"]), f"{MACOS_LOCK.name}: not an exact pin: {head!r}"
        assert any(
            re.search(r"--hash=sha256:[0-9a-f]{64}", line) for line in block
        ), f"{MACOS_LOCK.name}: {m['name']} has no sha256 hash"


def test_regeneration_entry_points_exist():
    assert LOCK_SCRIPT.is_file(), LOCK_SCRIPT
    assert LOCK_SCRIPT.stat().st_mode & 0o111, f"{LOCK_SCRIPT} must be executable"
    script = LOCK_SCRIPT.read_text()
    for flag in (
        "uv pip compile",
        "--exclude-newer",
        "--generate-hashes",
        "--no-build",
        '"3.12"',
        "aarch64-apple-darwin",
    ):
        assert flag in script, f"{LOCK_SCRIPT.name} lost {flag!r}"
    mk = MACOS_MK.read_text()
    assert re.search(r"^macos-requirements:\n\tscripts/macos/lock-requirements.sh$", mk, re.M)
    assert re.search(
        r"^macos-requirements-check:\n\tscripts/macos/lock-requirements.sh --check$", mk, re.M
    )


def test_inputs_carry_no_index_options_or_direct_references():
    """uv pip compile honours option lines inside the .in (--index-url,
    --extra-index-url, --find-links, -e, -r/-c); an extra index there would
    silently re-source the bundle's wheels, and the lock (no --emit-index-url)
    would not show it. The .in holds only comments and named requirements."""
    for raw in MACOS_IN.read_text().splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        assert not line.startswith("-"), f"{MACOS_IN.name}: option line not allowed: {raw!r}"
        assert (
            "@" not in line and "://" not in line and not line.startswith((".", "/"))
        ), f"{MACOS_IN.name}: direct/URL/path reference not allowed: {raw!r}"


def test_lock_script_passes_a_fixed_utc_exclude_newer_cutoff():
    """The resolve is cut off at a fixed UTC instant (EXCLUDE_NEWER), so
    --check and re-locks stay reproducible; bumping it is a reviewed edit.
    A stub `uv` records its argv; the script must pass --exclude-newer with
    exactly the cutoff parsed from its own text, in full RFC 3339 UTC form."""
    import os
    import subprocess
    import tempfile
    from datetime import datetime, timedelta

    script_text = LOCK_SCRIPT.read_text()
    assign = re.search(r'^EXCLUDE_NEWER="([^"]*)"\s*$', script_text, re.M)
    assert assign, f'{LOCK_SCRIPT.name}: no EXCLUDE_NEWER="..." assignment found'
    value = assign.group(1)

    # The cutoff must be full RFC 3339 UTC: date, time, and a literal Z
    # (not an offset, not a bare date) so uv sees one exact instant.
    cutoff_re = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    assert re.fullmatch(cutoff_re, value), (
        f"{LOCK_SCRIPT.name}: EXCLUDE_NEWER {value!r} is not a full UTC " "timestamp ending in Z"
    )
    # Boundary legs: a naive timestamp (exact bound, no trailing Z) and a
    # date-only value (just past it) must both be rejected by the same
    # regex, so a future edit cannot drop the UTC marker.
    assert cutoff_re.fullmatch("2026-10-21T01:39:02") is None
    assert cutoff_re.fullmatch("2026-10-21") is None
    # And the parse must really be UTC, not just Z-shaped.
    assert datetime.fromisoformat(value.replace("Z", "+00:00")).utcoffset() == timedelta(0)

    lock_before = MACOS_LOCK.read_bytes()

    # Stub `uv`: records each argv item on its own line, and creates the
    # --output-file target empty so the --check diff runs and reports drift
    # (exit 1), not a crash (exit 2) from a missing file.
    stub_body = "\n".join(
        [
            "#!/bin/sh",
            'for a in "$@"; do',
            '  printf "%s\\n" "$a" >> "$UV_ARGV_FILE"',
            "done",
            'out=""',
            'prev=""',
            'for a in "$@"; do',
            '  if [ "$prev" = "--output-file" ]; then out="$a"; fi',
            '  prev="$a"',
            "done",
            'if [ -n "$out" ]; then : > "$out"; fi',
        ]
    )

    with tempfile.TemporaryDirectory(prefix="lock-uv-") as td:
        tmp_path = Path(td)
        stub = tmp_path / "uv"
        stub.write_text(stub_body + "\n")
        stub.chmod(0o755)
        argv_file = tmp_path / "uv-argv.txt"

        env = dict(os.environ)
        env["PATH"] = f"{tmp_path}:{env['PATH']}"
        env["UV_ARGV_FILE"] = str(argv_file)

        proc = subprocess.run(
            ["bash", str(LOCK_SCRIPT), "--check"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

        argv_lines = argv_file.read_text().splitlines()
        assert argv_lines[:2] == [
            "pip",
            "compile",
        ], f"stub uv argv starts with {argv_lines[:2]!r}, expected ['pip', 'compile']"
        assert (
            argv_lines.count("--exclude-newer") == 1
        ), f"--exclude-newer occurs {argv_lines.count('--exclude-newer')} times, expected exactly 1"
        idx = argv_lines.index("--exclude-newer")
        assert (
            argv_lines[idx + 1] == value
        ), f"value after --exclude-newer is {argv_lines[idx + 1]!r}, expected {value!r}"

        assert proc.returncode == 1, (
            f"exit code {proc.returncode}, expected 1 (drift from empty stub output)\n"
            f"stderr: {proc.stderr}"
        )

        # The --check temp file must have been cleaned up by the EXIT trap.
        leftovers = list(REPO_ROOT.glob(".lock-check-*.txt"))
        assert not leftovers, f"leftover --check temp files: {leftovers}"

        assert (
            MACOS_LOCK.read_bytes() == lock_before
        ), f"{MACOS_LOCK.name} changed during --check run"
