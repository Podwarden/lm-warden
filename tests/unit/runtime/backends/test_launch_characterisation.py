"""Sub-project B's regression net.

Every case's argv and env must match the frozen snapshot EXACTLY. A failure
here means behaviour changed — which, for a refactor whose stated property is
"byte-identical", is always a bug and never a reason to regenerate the golden
file.
"""

import pytest

from app.runtime.backends.vllm import VllmBackend
from tests.unit.runtime.backends.corpus import CASES, load_golden

GOLDEN = load_golden()
BACKEND = VllmBackend()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_argv_matches_golden(case):
    plan = BACKEND.plan(
        case.model,
        port=case.port,
        bind_host=BACKEND.bind_host(case.driver),
        overrides=case.overrides,
        hf_token=case.hf_token,
        hf_cache_dir=case.hf_cache_dir,
        engine_driver=case.driver,
    )
    assert plan.argv[2:] == GOLDEN[case.name]["argv"]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_env_matches_golden(case):
    plan = BACKEND.plan(
        case.model,
        port=case.port,
        bind_host=BACKEND.bind_host(case.driver),
        overrides=case.overrides,
        hf_token=case.hf_token,
        hf_cache_dir=case.hf_cache_dir,
        engine_driver=case.driver,
    )
    assert plan.env == GOLDEN[case.name]["env"]


def test_every_case_is_in_the_golden_file():
    """A case added without regenerating the golden file must fail loudly,
    not be silently skipped."""
    assert sorted(k for k in GOLDEN if k != "_meta") == sorted(c.name for c in CASES)


def test_stub_mirrors_model_row():
    """A ratchet on the corpus's own honesty.

    StubModel exists to stand in for ModelRow. Where the two diverge, the
    corpus stops describing production. Today there are exactly two known
    divergences, both from the same latent bug: `quantization` and
    `max_num_seqs` are real columns (migration 0011) that ModelRow, _MODEL_COLS
    and _decode_row never learned about, so a production row has no such
    attribute and cmd_builder's getattr(..., None) silently drops the flag.

    WHEN THAT BUG IS FIXED, THIS TEST FAILS -- and the fix is to delete the
    name from KNOWN_DIVERGENCES, not to widen the exception. That is the point:
    the corpus should find out.

    Any OTHER divergence is a bug in the corpus and must be fixed here.
    """
    from dataclasses import fields as dc_fields

    from app.db.repos.models import ModelRow
    from tests.unit.runtime.backends.corpus import StubModel

    KNOWN_DIVERGENCES = {"quantization", "max_num_seqs"}

    stub = {f.name for f in dc_fields(StubModel)}
    row = {f.name for f in dc_fields(ModelRow)}

    extra = stub - row
    assert extra == KNOWN_DIVERGENCES, (
        f"StubModel declares {sorted(extra)} that ModelRow does not. If the "
        f"0011-column bug was just fixed, remove the name from "
        f"KNOWN_DIVERGENCES. Otherwise the corpus has drifted from production."
    )
    # Fields the corpus does not model are fine (status, pulled_bytes, ...);
    # they are not builder inputs. Only stub-side extras are dangerous.


def test_no_case_sets_trust_remote_code():
    """Deliberate coverage GAP, and it must stay deliberate.

    `trust_remote_code` is persisted (column, ModelRow, _decode_row, schemas)
    but cmd_builder NEVER emits `--trust-remote-code` -- verified 2026-09-01 by
    grep. So the row is served without the flag.

    No case sets it True, which means fixing that bug changes NO golden value
    and B's byte-identity claim is safe either way. Adding such a case is
    exactly the collision to avoid: captured BEFORE the fix it freezes "no
    flag", and the fix then breaks B's corpus for a reason unrelated to B's
    refactor. Add the case only on a tree where the bug is already fixed --
    Task 1 Step 6 precondition 5.
    """
    assert not any(c.model.trust_remote_code for c in CASES)


def test_the_golden_file_records_where_it_was_captured_from():
    """The corpus is only a regression net for the tree it was frozen against,
    and B's branch point is several commits downstream of the design spec's
    reference point -- an operator release, its backmerge, and all of
    sub-project A. Recording the SHA is what lets a later reader tell."""
    assert GOLDEN["_meta"]["captured_from"] != "unknown"
