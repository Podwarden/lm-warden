#!/usr/bin/env python3
"""Fill the marker-delimited shared regions of the public docs and the catalogue copy.

The operational hazards (the /dev/shm floor, one loaded model per GPU, what
`gpu_memory_utilization` really reserves, what fits on a 16 GiB card, why the
first load is slow, the mandatory cookie secret, the headless first run) are
true of this product however it was deployed, so both surfaces that document
it need the same words:

  * the public documents that ship verbatim to the public GitHub mirror --
    `documents/HAZARDS.md` (the five load-time hazards),
    `documents/INSTALL.md` (the cookie secret) and `documents/API.md` (the
    headless first run); `README.md` used to carry
    all seven before it was split by section, and
  * the PodWarden Hub catalogue row's long `content_md`, whose shared regions
    this script assembles into `docs/catalog-shared-regions.md`.

Kept as two hand-written copies they drift, and the copy that is wrong is
always the one somebody is reading when it matters. `deploy/hub/` was this
repo's worked example of that failure (deleted 2026-09-24): an unpublished mirror of the live
catalogue row that quietly fell to 5 `env_schema` entries against the row's 16.
One fragment per hazard under `docs/shared/`, pulled into both surfaces, is the
fix.

MARKER BLOCKS, DELIBERATELY NOT A GENERATED README
--------------------------------------------------
Each managed document carries marker pairs:

    <!-- shared:shm-sigbus -->
    ...body, written by this script from docs/shared/shm-sigbus.md...
    <!-- /shared:shm-sigbus -->

Only the region between a pair is machine-managed. Every other line of the
managed documents is hand-written and stays that way.

This is on purpose, and it is the part not to "simplify" later. A wholly
generated README would make every edit -- a typo, a broken link, a sentence
that reads badly -- require first knowing that an assembler exists and where
its input lives. Contributors do not know that, they edit the file in front of
them, and their fix is silently reverted by the next sync. That is how a
documentation pipeline stops being used: not by breaking, but by making the
cheap contribution expensive. With markers, a contributor who opens HAZARDS.md
to fix a typo just fixes it, unless the typo happens to be inside one of the
clearly fenced regions -- and a comment above the first of them in each file
names this script, so the answer to "why did my edit come back" is one grep
away.

The same contract as `frontend/src/lib/api-types.generated.ts`, and the same
warning: the marked regions are generated, they must never be hand-edited, and
CI computes the diff rather than trusting anyone to remember.

HEADINGS ARE NOT SHARED
-----------------------
A fragment holds the BODY of a hazard, never its heading. Each surface owns its
own headings: the public documents' are linked from the README and from each
other (`documents/HAZARDS.md#one-loaded-model-per-gpu`,
`documents/API.md#first-run-without-a-browser`), so an assembler
that rewrote them would silently break in-page navigation, and the catalogue
words some of its headings for a reader who is looking at a deployment form
rather than a repository. The prose underneath is identical; the sign above the
door is local.

WHAT IS NOT SHARED
------------------
Surface-specific material stays out of `docs/shared/`. Pushing it in would
trade "stale and duplicated" for "accurate and confusing":

  * Catalogue only -- the `x-podwarden` block, `storage_class`/Longhorn, "the
    GPU lands on the primary service because Caddy owns the published port",
    the compose-stack topology table.
  * GitHub only -- building from source, the chat-ui pinned SHA, the `make`
    targets, the contributing section.

A hazard that is mostly shared but has a surface-specific tail is split: the
shared fragment, then a local paragraph AFTER the closing marker. The `/dev/shm`
hazard is the worked example -- the SIGBUS and the 2 GiB floor are shared; "put
`shm_size` on the api service, because PodWarden's stack-level field lands on
Caddy" is catalogue-local and lives outside the markers.

`docs/` IS STRIPPED FROM THE PUBLIC SNAPSHOT
--------------------------------------------
`publish/exclude.txt` drops `/docs/`, so the fragments never reach GitHub. They
are a build input, not published reading, and NOTHING in a published document may
link to `docs/shared/*` -- the link would 404 for every public reader. The markers name
a slug, not a path, for exactly that reason.

That also means this script has no fragments to read in a public clone. It says
so and exits 0 there rather than failing a checkout it was never meant to run
in.

USAGE
-----
    python scripts/sync-shared-docs.py            # rewrite the managed regions
    python scripts/sync-shared-docs.py --check    # exit 1 if anything differs

`make sync-shared-docs` / `make check-shared-docs` wrap both. `--check` runs in
CI (`lint:shared-docs`).
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# One self-contained fragment per hazard.
FRAGMENT_DIR = REPO_ROOT / "docs" / "shared"

# Every document whose marked regions this script owns. All are checked by
# CI. The public documents ship to GitHub (publish/exclude.txt strips /docs/
# but not /documents/, and not the root README);
# docs/catalog-shared-regions.md is the hand-off to the Hub row. README.md
# carries no region today -- the hazards moved out to the sectioned documents
# -- but stays listed so a marker added back to it is still filled and checked
# rather than silently ignored.
MANAGED_DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "documents" / "INSTALL.md",
    REPO_ROOT / "documents" / "API.md",
    REPO_ROOT / "documents" / "HAZARDS.md",
    REPO_ROOT / "docs" / "catalog-shared-regions.md",
)

OPEN_RE = re.compile(r"^<!-- shared:([a-z0-9][a-z0-9-]*) -->$")
CLOSE_RE = re.compile(r"^<!-- /shared:([a-z0-9][a-z0-9-]*) -->$")


class SyncError(Exception):
    """A malformed document or a missing fragment. Always fatal."""


def load_fragment(slug: str) -> list[str]:
    """Return the fragment body as lines, with no leading/trailing blanks."""
    path = FRAGMENT_DIR / f"{slug}.md"
    if not path.is_file():
        raise SyncError(
            f"no fragment for marker 'shared:{slug}': expected "
            f"{path.relative_to(REPO_ROOT)}"
        )
    body = path.read_text(encoding="utf-8").split("\n")
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        raise SyncError(f"fragment {path.relative_to(REPO_ROOT)} is empty")
    return body


def assemble(doc: Path) -> tuple[str, set[str]]:
    """Return (document with every marked region refilled, slugs it consumed)."""
    text = doc.read_text(encoding="utf-8")
    lines = text.split("\n")
    out: list[str] = []
    consumed: set[str] = set()

    i = 0
    while i < len(lines):
        line = lines[i]
        opened = OPEN_RE.match(line.strip())
        if not opened:
            if CLOSE_RE.match(line.strip()):
                raise SyncError(
                    f"{doc.relative_to(REPO_ROOT)}:{i + 1}: closing marker "
                    f"{line.strip()} with no opening marker"
                )
            out.append(line)
            i += 1
            continue

        slug = opened.group(1)
        if slug in consumed:
            raise SyncError(
                f"{doc.relative_to(REPO_ROOT)}:{i + 1}: 'shared:{slug}' appears "
                "twice in one document; a slug may be consumed once per file"
            )

        # Find the matching close, so a stray marker cannot swallow the rest of
        # the file.
        close_at = None
        for j in range(i + 1, len(lines)):
            closed = CLOSE_RE.match(lines[j].strip())
            if closed:
                if closed.group(1) != slug:
                    raise SyncError(
                        f"{doc.relative_to(REPO_ROOT)}:{j + 1}: expected "
                        f"<!-- /shared:{slug} --> but found {lines[j].strip()}"
                    )
                close_at = j
                break
            if OPEN_RE.match(lines[j].strip()):
                raise SyncError(
                    f"{doc.relative_to(REPO_ROOT)}:{j + 1}: 'shared:{slug}' "
                    "opened at line "
                    f"{i + 1} was never closed"
                )
        if close_at is None:
            raise SyncError(
                f"{doc.relative_to(REPO_ROOT)}:{i + 1}: 'shared:{slug}' has no "
                "closing marker"
            )

        out.append(line)
        out.extend(load_fragment(slug))
        out.append(lines[close_at])
        consumed.add(slug)
        i = close_at + 1

    return "\n".join(out), consumed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fill the shared marker regions of README.md and the catalogue "
            "hand-off from docs/shared/."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 (with a diff) if any managed region is stale",
    )
    args = parser.parse_args(argv)

    if not FRAGMENT_DIR.is_dir():
        # The public snapshot: publish/exclude.txt strips /docs/, so the
        # fragments this script reads are not part of a GitHub clone. There is
        # nothing to check and nothing is wrong.
        print(
            f"sync-shared-docs: {FRAGMENT_DIR.relative_to(REPO_ROOT)} is not "
            "present -- this is the public snapshot, where docs/ is stripped. "
            "Nothing to do."
        )
        return 0

    declared = {p.stem for p in FRAGMENT_DIR.glob("*.md")} - {"README"}
    consumed_anywhere: set[str] = set()
    stale: list[Path] = []

    for doc in MANAGED_DOCUMENTS:
        if not doc.is_file():
            raise SyncError(f"managed document missing: {doc}")
        assembled, consumed = assemble(doc)
        consumed_anywhere |= consumed
        current = doc.read_text(encoding="utf-8")
        if assembled == current:
            continue
        if args.check:
            stale.append(doc)
            rel = doc.relative_to(REPO_ROOT)
            sys.stdout.writelines(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    assembled.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel} (assembled from docs/shared/)",
                )
            )
        else:
            doc.write_text(assembled, encoding="utf-8")
            print(f"sync-shared-docs: rewrote {doc.relative_to(REPO_ROOT)}")

    orphans = sorted(declared - consumed_anywhere)
    if orphans:
        raise SyncError(
            "fragments nobody consumes: "
            + ", ".join(f"docs/shared/{s}.md" for s in orphans)
            + ". Add the marker pair to the surfaces that need them, or delete "
            "the fragment -- an unconsumed fragment is drift with a head start."
        )

    if stale:
        names = ", ".join(str(p.relative_to(REPO_ROOT)) for p in stale)
        print(
            f"\nsync-shared-docs: {names} differ(s) from docs/shared/.\n"
            "The regions between <!-- shared:SLUG --> markers are generated, "
            "like frontend/src/lib/api-types.generated.ts: never hand-edit "
            "them.\nEdit docs/shared/<slug>.md and run "
            "`make sync-shared-docs`, or if the edit above is the one you "
            "meant, move it into the fragment.",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print("sync-shared-docs: all shared regions are in sync.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SyncError as exc:
        print(f"sync-shared-docs: {exc}", file=sys.stderr)
        sys.exit(2)
