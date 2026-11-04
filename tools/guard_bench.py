"""Guard latency bench: cold scan of a ~1 MB code-heavy body, and a warm (memoised) repeat."""

import json
import pathlib
import statistics
import time

from app.guard.config import compile_guard
from app.guard.walk import _MEMO, inspect_body

ROOT = pathlib.Path(__file__).resolve().parents[1]
files = sorted((ROOT / "app").rglob("*.py"))
text_parts, size = [], 0
for f in files:
    t = f.read_text(errors="replace")
    text_parts.append(t)
    size += len(t)
    if size > 1_000_000:
        break
msgs = [{"role": "user", "content": "review this"}]
for i, t in enumerate(text_parts):
    msgs.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "read", "arguments": json.dumps({"path": f"f{i}.py"})},
                }
            ],
        }
    )
    msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": t})
body = {"model": "m", "messages": msgs}
g = compile_guard({"mode": "sanitize"})
cold, warm = [], []
for _ in range(7):
    _MEMO.clear()
    t0 = time.perf_counter()
    r = inspect_body(body, "openai", g, token_id="t", secret=b"k" * 32)
    cold.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    inspect_body(body, "openai", g, token_id="t", secret=b"k" * 32)
    warm.append(time.perf_counter() - t0)
print(
    f"body {size/1e6:.2f} MB, {len(msgs)} messages, findings {len(r.findings)}: {sorted({h.rule_id for h in r.findings})}"
)
print(
    f"cold p50 {statistics.median(cold)*1000:.1f} ms  max {max(cold)*1000:.1f} ms | warm p50 {statistics.median(warm)*1000:.2f} ms"
)
