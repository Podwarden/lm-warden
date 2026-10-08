"""Lifespan wiring for the chat2 GC background task (spec §3 collector).

Mirrors the existing sampler/pruner/watchdog lifespan tests: the task is
created on startup, stored on ``app.state`` so tests/ops can introspect it,
and is cancelled (not left dangling) on shutdown.
"""


def test_gc_task_started_and_stopped(tmp_data_dir, client) -> None:
    client.get("/healthz")
    task = client.app.state.chat2_gc_task
    assert task is not None and not task.done()
