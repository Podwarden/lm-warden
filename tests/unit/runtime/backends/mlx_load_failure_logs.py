"""Real mlx_lm.server model-load failure logs, captured on the M1.

Captured 2026-10-21 with mlx_lm 0.32.2 (the farm venv) by running
``python -m mlx_lm.server --model <dir> --port <p>`` against three throwaway
dirs: an EMPTY dir, a copy of Qwen2.5-0.5B-Instruct-4bit whose
``model.safetensors`` was overwritten with 4 KiB of random bytes (CORRUPT),
and a copy whose config.json says ``"model_type": "frobnicator"`` (ARCH).
In all three the process did NOT exit: /health answered 503 for the full 60 s
observed, and the log ended in an endless run of ``GET /health ... 503``.

The lines are verbatim except for two substitutions: the throwaway dir prefix
is ``/tmp/mlx-broken`` and the venv prefix is ``/opt/lmw/venv`` (the captured
paths named a home directory). Three /health lines stand in for the endless
run each log ended in.
"""

from __future__ import annotations

_SP = "/opt/lmw/venv/lib/python3.12/site-packages"

_LOAD_FRAMES = f"""\
  File "{_SP}/mlx_lm/server.py", line 544, in _run_generate
    self._generate()
  File "{_SP}/mlx_lm/server.py", line 782, in _generate
    self.model_provider.load_default()
  File "{_SP}/mlx_lm/server.py", line 475, in load_default
    self.load("default_model", None, "default_model")
  File "{_SP}/mlx_lm/server.py", line 484, in load
    self._load(*model_key)
  File "{_SP}/mlx_lm/server.py", line 436, in _load
    model, tokenizer, config = load(
                               ^^^^^
  File "{_SP}/mlx_lm/utils.py", line 690, in load
    model, config = load_model(
                    ^^^^^^^^^^^
"""


def _server_tail(port: int) -> str:
    return f"""\
{_SP}/mlx_lm/server.py:1894: UserWarning: mlx_lm.server is not recommended for production as it only implements basic security checks.
  warnings.warn(
2026-10-21 01:18:41,122 - INFO - Starting httpd at 127.0.0.1 on port {port}...
127.0.0.1 - - [21/Oct/2026 01:18:41] "GET /health HTTP/1.1" 503 -
127.0.0.1 - - [21/Oct/2026 01:18:42] "GET /health HTTP/1.1" 503 -
127.0.0.1 - - [21/Oct/2026 01:18:43] "GET /health HTTP/1.1" 503 -
"""


# The fatal line of each, exactly as the engine wrote it (timestamp included).
EMPTY_DIR_FATAL = (
    "2026-10-21 01:18:41,118 - ERROR - mlx_lm.server generation thread died: "
    "[Errno 2] No such file or directory: '/tmp/mlx-broken/empty/config.json'"
)
CORRUPT_SAFETENSORS_FATAL = (
    "2026-10-21 01:19:45,761 - ERROR - mlx_lm.server generation thread died: "
    "[load_safetensors] Invalid json header length file "
    "/tmp/mlx-broken/corrupt/model.safetensors"
)
UNSUPPORTED_ARCH_FATAL = (
    "2026-10-21 01:20:50,414 - ERROR - mlx_lm.server generation thread died: "
    "Model type frobnicator not supported."
)

EMPTY_DIR_LOG = f"""\
{EMPTY_DIR_FATAL}
Traceback (most recent call last):
{_LOAD_FRAMES}\
  File "{_SP}/mlx_lm/utils.py", line 445, in load_model
    config = load_config(model_path)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "{_SP}/mlx_lm/utils.py", line 364, in load_config
    with open(model_path / "config.json", "r") as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/mlx-broken/empty/config.json'
{_server_tail(18951)}"""

CORRUPT_SAFETENSORS_LOG = f"""\
{CORRUPT_SAFETENSORS_FATAL}
Traceback (most recent call last):
{_LOAD_FRAMES}\
  File "{_SP}/mlx_lm/utils.py", line 456, in load_model
    weights.update(mx.load(wf))
                   ^^^^^^^^^^^
RuntimeError: [load_safetensors] Invalid json header length file /tmp/mlx-broken/corrupt/model.safetensors
{_server_tail(18952)}"""

UNSUPPORTED_ARCH_LOG = f"""\
{UNSUPPORTED_ARCH_FATAL}
Traceback (most recent call last):
  File "{_SP}/mlx_lm/utils.py", line 222, in _get_classes
    arch = importlib.import_module(f"mlx_lm.models.{{model_type}}")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'mlx_lm.models.frobnicator'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
{_LOAD_FRAMES}\
  File "{_SP}/mlx_lm/utils.py", line 474, in load_model
    model_class, model_args_class = get_model_classes(config=config)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "{_SP}/mlx_lm/utils.py", line 225, in _get_classes
    raise ValueError(msg) from e
ValueError: Model type frobnicator not supported.
{_server_tail(18953)}"""

# case name -> (the captured log, its fatal line)
CAPTURED = {
    "empty_dir": (EMPTY_DIR_LOG, EMPTY_DIR_FATAL),
    "corrupt_safetensors": (CORRUPT_SAFETENSORS_LOG, CORRUPT_SAFETENSORS_FATAL),
    "unsupported_arch": (UNSUPPORTED_ARCH_LOG, UNSUPPORTED_ARCH_FATAL),
}
