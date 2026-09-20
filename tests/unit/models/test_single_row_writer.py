"""There is one operator-facing writer of the ``models`` table, and the state
machine's columns have their own.

The recurring defect behind #256 / #260 / #261 / #262 / #265 was not a count of
HTTP doors, it was that the ``ModelRow`` had six producers and only one of them
carried the ruleset. The count is what these tests hold down: a new raw ``UPDATE
models`` anywhere outside the repo is a seventh producer, and it will arrive
without the policy table, ``ModelSpec``, the principal check, the loaded guard or
the cross-row checks -- exactly as ``try_stack`` and stress-apply did.

Source-text tests, deliberately. There is no runtime hook that can observe "a
module wrote SQL", and the failure mode is a future edit, so the check has to
read the code.
"""
import ast
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[3] / "app"

#: The only module allowed to write the ``models`` table. Its named methods --
#: ``update_status``, ``set_prior_status``, ``update_pull_progress``,
#: ``mark_runtime_dead_on_startup`` -- are the sole writers of the RUNTIME
#: columns, and ``update_fields`` (which only ``app/models/writer.py`` calls) is
#: the sole writer of the operator columns.
WRITER_MODULE = "db/repos/models.py"

_UPDATE_MODELS = re.compile(r"UPDATE\s+models\b", re.IGNORECASE)

#: Modules that produce rows with NO principal at the keyboard. They own only
#: RUNTIME columns and must not be able to reach ``apply_model_change`` -- the
#: writer refuses ``SYSTEM`` at runtime, and this keeps the import out too, so
#: the refusal cannot be dodged by inventing a principal.
NO_PRINCIPAL_MODULES = (
    "runtime/watchdog.py",
    "runtime/boot_reconcile.py",
    "runtime/supervisor.py",
    "models/pull_task.py",
)


def _python_files():
    return sorted(p for p in APP.rglob("*.py"))


def test_only_the_models_repo_writes_the_models_table():
    offenders = []
    for path in _python_files():
        rel = path.relative_to(APP).as_posix()
        if rel == WRITER_MODULE:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not _UPDATE_MODELS.search(line):
                continue
            # Prose, not SQL: a comment, or a docstring quoting the statement in
            # ``double backticks`` to explain why it must not be written by hand.
            if line.lstrip().startswith("#") or "``" in line:
                continue
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert offenders == [], (
        "a raw UPDATE of the models table outside app/db/repos/models.py:\n"
        + "\n".join(offenders)
        + "\n\nOperator columns go through app/models/writer.py::"
        "apply_model_change; a state-machine column needs a named ModelRepo "
        "method of its own."
    )


def test_the_repo_still_owns_the_statements():
    """The inverse of the test above: if the statements moved rather than being
    removed, the guard above would pass while proving nothing."""
    text = (APP / WRITER_MODULE).read_text()
    assert len(_UPDATE_MODELS.findall(text)) >= 4


def test_the_no_principal_modules_cannot_call_the_writer():
    offenders = []
    for rel in NO_PRINCIPAL_MODULES:
        text = (APP / rel).read_text()
        if "apply_model_change" in text or "models.writer" in text:
            offenders.append(rel)
    assert offenders == [], (
        f"{offenders} imported the operator writer. These run with no principal "
        f"-- the watchdog restarts a row with nobody at the keyboard -- so they "
        f"may write only the RUNTIME columns, through the named ModelRepo "
        f"methods that own them."
    )


def test_update_fields_is_only_called_by_the_writer():
    callers = []
    for path in _python_files():
        rel = path.relative_to(APP).as_posix()
        if rel in (WRITER_MODULE, "models/writer.py"):
            continue
        if "update_fields(" in path.read_text():
            callers.append(rel)
    assert callers == [], (
        f"{callers} called ModelRepo.update_fields directly, which skips the "
        f"policy table, ModelSpec and the principal check. Call "
        f"app/models/writer.py::apply_model_change instead."
    )


# ---------------------------------------------------------------------------
# Destructive steps and validation
# ---------------------------------------------------------------------------

#: Calls that take an engine down or start one. A module that does one of these
#: AND writes a model row has to think about the order of the two.
_DISRUPTIVE = re.compile(r"\b(sup|self\._sup|supervisor)\.unload\(|\b_restart\(")

#: Functions that legitimately do both. Exactly one, and it is the one that got
#: this wrong: `POST /{id}/stress/apply` unloaded the engine and then called
#: `apply_model_change`, which validates the WHOLE merged row and can refuse
#: over a column the request never touched. A stale `extra_env` key -- the shape
#: #266's sweep leaves alone and `filter_extra_env` drops at launch, so the
#: model serves fine -- turned an ordinary "Apply and reload" into an outage the
#: watchdog would not recover.
#:
#: The rule, for whoever adds the next entry: **never put a destructive step
#: before a validation that can fail for reasons unrelated to the request.**
#: Run `dry_run_model_change` first, while the refusal is still free.
DISRUPT_AND_WRITE = {"stress/routes_api.py::apply_stress_recommendation"}


def _functions_with_source():
    """Every function/method in ``app/``, as ``(label, source text)``."""
    for path in _python_files():
        rel = path.relative_to(APP).as_posix()
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - app/ must parse
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield f"{rel}::{node.name}", ast.get_source_segment(text, node) or ""


def test_a_function_that_disrupts_an_engine_and_writes_a_row_validates_first():
    """A convention lint, not a guard — read it as one.

    ``ast`` only enumerates the functions; the check itself is a regex and a
    ``str.find`` over each function's source text. There is no call graph and no
    control flow, so it is blind to a renamed supervisor variable, to the write
    moved one call into a helper, and to a ``dry_run_model_change(`` sitting in a
    branch that never runs. It does catch the obvious offender, which is how the
    rule gets remembered.

    The assertion with real weight is the roster below: a function that both
    disrupts an engine and writes a row has to be added to
    ``DISRUPT_AND_WRITE`` by hand, which puts a human in front of the question.
    """
    offenders = []
    found = set()
    for label, source in _functions_with_source():
        disruption = _DISRUPTIVE.search(source)
        if disruption is None or "apply_model_change(" not in source:
            continue
        found.add(label)
        dry = source.find("dry_run_model_change(")
        if dry == -1 or dry > disruption.start():
            offenders.append(label)
    assert offenders == [], (
        f"{offenders} take an engine down before validating the change they are "
        f"about to write. A merged-row refusal can be about a column the request "
        f"never touched, so after the unload it is an outage rather than an "
        f"error message. Call dry_run_model_change FIRST, while the refusal is "
        f"still free."
    )
    # And the one function that legitimately does both is still the one that
    # taught us the rule -- a new entry here is a review checkpoint.
    assert found == DISRUPT_AND_WRITE
