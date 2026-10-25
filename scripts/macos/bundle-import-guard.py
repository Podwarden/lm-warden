#!/usr/bin/env python3
"""bundle-import-guard.py — the P11 import net.

Run by the BUNDLED interpreter, it imports every module the server needs on
the Mac and exits 1 naming the first failure:

    <interp> -I -B scripts/macos/bundle-import-guard.py [--list] [--only a,b] [--walk pkg]

It is the regression net the P11 prune (transformers ``modeling_*`` /
``modular_*`` / image-video files, PyObjCTest, Tk, ensurepip, headers) is
gated on: if the prune drops a file the server still imports, this script
turns the bundle build red and names the module.

Order of imports:
  (1) REQUIRED — an explicit tuple of third-party and stdlib modules;
  (2) every submodule of the ``app`` package (``pkgutil.walk_packages``);
  (3) every submodule of ``mlx_lm.models`` (all families users can load);
  (4) transformers' tokenizer/config resolution: ``auto.tokenization_auto``
      and ``auto.configuration_auto`` plus, for every model dir that has
      them, its ``configuration_*`` and ``tokenization_*`` modules (the
      files the prune must keep). ``transformers.AutoTokenizer`` /
      ``AutoConfig`` attribute access is forced (lazy module).

Stdlib only at top level: the interpreter running us may not yet be able to
import the very modules this script exists to check.

``-I -B`` (as the bundled interpreter is always run) keeps this script from
writing ``.pyc`` anywhere; it writes nothing at all.

Exit codes: 0 all imports ok (final line ``IMPORT-GUARD: ok <N> modules``),
1 at least one import failed (one ``IMPORT-GUARD: FAIL <module>: <ExcType>:
<msg>`` line per failure, then ``IMPORT-GUARD: failed <k> of <N>``), 2 usage.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import pkgutil
import sys

# (1) The explicit required modules, in import order.
REQUIRED = (
    "ssl",
    "sqlite3",
    "ctypes",
    "asyncio",
    "multiprocessing",
    "uvicorn",
    "uvloop",
    "fastapi",
    "starlette",
    "pydantic",
    "httpx",
    "cryptography",
    "bcrypt",
    "PIL.Image",
    "jinja2",
    "yaml",
    "safetensors",
    "tokenizers",
    "sentencepiece",
    "huggingface_hub",
    "mlx.core",
    "mlx.nn",
    "mlx_lm",
    "mlx_lm.server",
    "mlx_lm.utils",
    "mlx_lm.tokenizer_utils",
    "transformers",
    "Metal",
)

# (2) The server package: every submodule via walk_packages.
APP_PACKAGE = "app"

# (3) The MLX model families users can load.
MLX_MODELS_PACKAGE = "mlx_lm.models"

# (4) transformers' tokenizer/config resolution.
TRANSFORMERS_MODELS = "transformers.models"
TRANSFORMERS_AUTO_TOKENIZER = "transformers.models.auto.tokenization_auto"
TRANSFORMERS_AUTO_CONFIG = "transformers.models.auto.configuration_auto"

# Modules that fail on the CURRENT untrimmed bundle for reasons unrelated to
# the prune. Every entry is reviewed; a module listed here that now imports
# is fine (the list is an allow-list, not a pin).
ALLOWED_FAILURES = {
    # mlx_lm's olmo family requires the separate ai2-olmo package (not
    # shipped in the bundle); importing the module prints "To run olmo
    # install ai2-olmo" and calls sys.exit(1).
    "mlx_lm.models.olmo": "needs the ai2-olmo package, not shipped",
    # pop2piano's tokenizer module raises NameError(pretty_midi): it depends
    # on the optional midi-tokenizer package, which the bundle does not ship.
    "transformers.models.pop2piano.tokenization_pop2piano": (
        "needs the optional midi-tokenizer package"
    ),
}


def _spec_for(package: str):
    """The ModuleSpec for ``package`` (finders only, no import), or None.

    ``-I`` drops the working directory from sys.path, so it is tried as a
    final entry: the guard must find ``app`` (and a ``--walk`` probe package)
    in the working directory. A dotted package resolves only under a sys.path
    entry that holds its whole parent chain.
    """
    try:
        for path_entry in list(sys.path) + [os.getcwd()]:
            if not path_entry:
                continue
            entry = os.path.abspath(path_entry)
            if not os.path.isdir(entry):
                continue
            if "." in package:
                root, _, rest = package.partition(".")
                spec = importlib.machinery.PathFinder.find_spec(root, [entry])
                if spec is None:
                    continue
                ok = True
                for part in rest.split("."):
                    spec = importlib.machinery.PathFinder.find_spec(
                        part, list(spec.submodule_search_locations or [])
                    )
                    if spec is None:
                        ok = False
                        break
                if ok:
                    return spec
            else:
                spec = importlib.machinery.PathFinder.find_spec(package, [entry])
                if spec is not None:
                    return spec
    except BaseException:
        pass
    return None


def _sys_path_root(package: str, pkg_dir: str) -> str:
    """The sys.path entry that contains ``package`` located at ``pkg_dir``."""
    # One level up per dotted part: ``a`` at <root>/a, ``a.b`` at <root>/a/b.
    up = os.path.join(pkg_dir, *[".."] * len(package.split(".")))
    return os.path.abspath(up)


def _walk_with_imports(package: str, spec) -> tuple[list[str], list[str]]:
    """``walk_packages`` descent: it IMPORTS every package it descends into.

    Used only where that is safe -- ``--list`` and the ``--walk`` test leg:
    the rooted directory (the repo for ``app``, the cwd for a probe, a
    site-packages package dir) is trusted, and ``-B`` (the bundled
    interpreter is always run with it) keeps the descent from writing
    ``.pyc``. ``onerror`` is called with the NAME of a package whose import
    failed; those names come back as discovery errors, never raised, so one
    unreadable file cannot hide the modules after it.
    """
    errors: list[str] = []
    names: list[str] = []
    root = _sys_path_root(package, os.path.abspath(spec.submodule_search_locations[0]))
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        for info in pkgutil.walk_packages(
            spec.submodule_search_locations, package + ".", onerror=errors.append
        ):
            if info.name.rpartition(".")[2] != "__init__":
                names.append(info.name)
    finally:
        # Only what we added: site-packages is already there for mlx_lm.
        if added:
            sys.path.remove(root)
    return names, errors


def _descend(package: str, spec) -> tuple[list[str], list[str]]:
    """Finder-only descent: nothing is imported while discovering."""

    errors: list[str] = []
    names: list[str] = []
    stack: list[tuple[str, list[str]]] = [
        (package, [str(p) for p in spec.submodule_search_locations])
    ]
    seen: set[str] = set()
    while stack:
        prefix, locations = stack.pop()
        for loc in locations:
            key = os.path.abspath(loc)
            if key in seen:
                continue
            seen.add(key)
        for info in pkgutil.iter_modules(locations, prefix + "."):
            leaf = info.name.rpartition(".")[2]
            if leaf == "__init__":
                continue
            names.append(info.name)
            if info.ispkg:
                try:
                    sub_spec = importlib.machinery.PathFinder.find_spec(leaf, locations)
                except BaseException as exc:
                    errors.append(f"{info.name} discovery: {type(exc).__name__}: {exc}")
                    continue
                if sub_spec is None or not sub_spec.submodule_search_locations:
                    continue
                stack.append((info.name, [str(p) for p in sub_spec.submodule_search_locations]))
    return names, errors


def _recorded_py(package: str, pkg_dir: str) -> list[str]:
    """Posix paths of the ``.py`` files under ``package`` that the installing
    wheel's ``RECORD`` lists (the dist-info next to the package), or ``[]``.

    The disk walk only sees what is still on disk: a prune that deletes a
    whole family file (``mlx_lm/models/phi3.py``, a ``configuration_*.py``)
    leaves nothing to import and nothing to fail. ``RECORD`` was written by
    pip before any prune, so its entries keep a deleted file on the list and
    its import fails, naming it. Without a dist-info (``app`` is copied from
    ``git archive``, not installed) the disk walk alone stands.
    """
    # stdlib; deferred so --only stays minimal
    import csv
    import importlib.metadata

    root = _sys_path_root(package, pkg_dir)
    prefix = package.replace(".", "/") + "/"
    out: list[str] = []
    try:
        for dist in importlib.metadata.distributions(path=[root]):
            # read_text, not dist.files: files drops entries missing on disk
            # (CPython 3.12+), which are exactly the ones this exists for.
            text = dist.read_text("RECORD") or ""
            for row in csv.reader(text.splitlines()):
                rel = row[0].replace(os.sep, "/") if row else ""
                if rel.startswith(prefix) and rel.endswith(".py"):
                    out.append(rel)
    except Exception:
        return []
    return sorted(set(out))


def _module_name(rel: str) -> str | None:
    """``a/b/c.py`` -> ``a.b.c``; ``a/b/__init__.py`` -> ``a.b``; None if not a module path."""
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def _discover(package: str, allow_imports: bool) -> tuple[list[str], list[str]]:
    """``[package] + every non-``__init__`` submodule, in walk order, then
    any submodule the package's ``RECORD`` lists that the walk did not find."""
    spec = _spec_for(package)
    if spec is None or not spec.submodule_search_locations:
        return [package], []
    names, errors = _walk_with_imports(package, spec) if allow_imports else _descend(package, spec)
    seen = {package, *names}
    pkg_dir = os.path.abspath(spec.submodule_search_locations[0])
    for rel in _recorded_py(package, pkg_dir):
        name = _module_name(rel)
        if name is not None and name not in seen:
            seen.add(name)
            names.append(name)
    return [package] + names, errors


def _transformers_tok_cfg() -> list[str]:
    """(4) The tokenizer/config modules the prune must keep, in sorted order."""
    pkg = importlib.import_module(TRANSFORMERS_MODELS)
    root = pkg.__path__[0]
    per_model: list[str] = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path) or entry.startswith("_"):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".py"):
                continue
            stem = fname[:-3]
            if stem.startswith("configuration_") or stem.startswith("tokenization_"):
                per_model.append(f"{TRANSFORMERS_MODELS}.{entry}.{stem}")
    # RECORD keeps a file the prune deleted on the list (see _recorded_py).
    on_disk = set(per_model)
    for rel in _recorded_py(TRANSFORMERS_MODELS, root):
        parts = rel[:-3].split("/")
        if len(parts) != 4 or parts[2].startswith("_"):
            continue
        if parts[3].startswith(("configuration_", "tokenization_")):
            name = _module_name(rel)
            if name is not None and name not in on_disk:
                on_disk.add(name)
                per_model.append(name)
    return [TRANSFORMERS_AUTO_TOKENIZER, TRANSFORMERS_AUTO_CONFIG] + per_model


def resolve_names(list_only: bool) -> list[str]:
    """The full module list, in import order, de-duplicated.

    ``--list`` roots the discovery at the repo (imports allowed: it is the
    guard's own tree, and nothing else imports user code in that mode); a
    real run (bundled or farm) discovers finder-only so a broken file in
    site-packages cannot import half a package just to be discovered -- the
    explicit imports report it instead.
    """
    names: list[str] = list(REQUIRED)
    names.extend(_discover(APP_PACKAGE, allow_imports=list_only)[0])
    names.extend(_discover(MLX_MODELS_PACKAGE, allow_imports=list_only)[0])
    names.extend(_transformers_tok_cfg())
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _import_one(name: str, allowed: set[str]) -> BaseException | None:
    try:
        importlib.import_module(name)
    except BaseException as exc:
        if name in allowed:
            return None
        return exc
    return None


def run(names: list[str], force_auto: bool = True) -> tuple[int, list[str]]:
    """Import each name once. Returns (count imported, failure lines).

    ``force_auto`` (the full run only): also force the lazy
    ``transformers.AutoTokenizer`` / ``AutoConfig``; ``--only`` / ``--walk``
    import exactly their names.
    """
    state = {"ok": 0}
    failures: list[str] = []
    for name in names:
        exc = _import_one(name, ALLOWED_FAILURES)
        if exc is None:
            state["ok"] += 1
        else:
            failures.append(f"IMPORT-GUARD: FAIL {name}: {type(exc).__name__}: {exc}")
    if not force_auto:
        return state["ok"], failures
    # transformers.AutoTokenizer / AutoConfig are lazy attributes: force them.
    try:
        tf = importlib.import_module("transformers")
        _ = tf.AutoTokenizer
        _ = tf.AutoConfig
    except BaseException as exc:
        failures.append(
            f"IMPORT-GUARD: FAIL transformers.AutoTokenizer/AutoConfig: {type(exc).__name__}: {exc}"
        )
    return state["ok"], failures


def main(argv: list[str]) -> int:
    list_only = False
    only: str | None = None
    walk_pkg: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--list":
            list_only = True
        elif arg in ("--only", "--walk"):
            i += 1
            if i >= len(argv):
                return 2
            value = argv[i]
            if arg == "--only":
                only = value
            else:
                walk_pkg = value
        else:
            return 2
        i += 1
    if only is not None and walk_pkg is not None:
        return 2

    smoke_model = os.environ.get("LMW_MLX_SMOKE_MODEL", "")

    if only is not None:
        names = [n for n in only.split(",") if n]
        if not names:
            return 2
    elif walk_pkg is not None:
        names, _errors = _discover(walk_pkg, allow_imports=True)
        # The walk takes its root back off sys.path; the imports below need
        # it, or every name fails as ModuleNotFoundError of the package and a
        # clean package reads the same as a broken one.
        spec = _spec_for(walk_pkg)
        if spec is not None and spec.submodule_search_locations:
            root = _sys_path_root(walk_pkg, os.path.abspath(spec.submodule_search_locations[0]))
            if root not in sys.path:
                sys.path.insert(0, root)
    else:
        names = resolve_names(list_only)

    if list_only:
        for name in sorted(set(names)):
            print(name)
        return 0

    total = len(names)
    ok, failures = run(names, force_auto=only is None and walk_pkg is None)

    if smoke_model and not failures:
        # Tokenizer only, no weights, no GPU: no live-model lock needed.
        # load_tokenizer lives in mlx_lm.utils in mlx-lm 0.32 (the bundled
        # pin); tokenizer_utils holds its TokenizerWrapper/detokenizer pieces.
        try:
            import mlx_lm.utils as mlx_lm_utils

            mlx_lm_utils.load_tokenizer(smoke_model)
        except BaseException as exc:
            failures.append(f"IMPORT-GUARD: FAIL LMW_MLX_SMOKE_MODEL: {type(exc).__name__}: {exc}")
            total += 1

    if failures:
        for line in failures:
            print(line)
        print(f"IMPORT-GUARD: failed {len(failures)} of {total}")
        return 1
    print(f"IMPORT-GUARD: ok {ok} modules")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
