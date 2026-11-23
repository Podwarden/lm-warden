"""The Prompt Lens switch (app/proxy/prompt_lens_switch.py).

Precedence: the runtime setting ``prompt_lens_enabled`` when it is set, else
the env (``VW_PROMPT_LENS_ENABLED`` or its deprecated alias
``VW_GODMODE_ENABLED``), else off. Turning it off drops what is held in memory
(ring, key windows, sections, images) and ends every open stream; kept captures
on disk are not touched.
"""

import dataclasses
from types import SimpleNamespace

from app.config import Settings
from app.proxy import prompt_lens_switch as sw
from app.proxy.prompt_lens import STOP, PromptLensHub, PromptLensMediaStore
from app.proxy.prompt_lens_sections import PromptLensSections


def _app(*, env_on: bool = False, env_var: str | None = None, runtime=None):
    settings = Settings(
        data_dir=None,  # type: ignore[arg-type]
        hf_cache_dir=None,  # type: ignore[arg-type]
        cookie_secret="x" * 32,
        container_gpu_count=0,
        prompt_lens_enabled=env_on,
        prompt_lens_env_var=env_var,
    )
    sections = PromptLensSections()
    state = SimpleNamespace(
        settings=settings,
        prompt_lens_sections=sections,
        prompt_lens_hub=PromptLensHub(sections=sections),
        prompt_lens_media=PromptLensMediaStore(),
    )
    if runtime is not None:
        state.prompt_lens_runtime = runtime
    return SimpleNamespace(state=state)


def test_default_is_off():
    app = _app()
    assert sw.enabled(app) is False
    assert sw.status(app) == {
        "enabled": False,
        "source": "default",
        "setting": None,
        "env_var": None,
        "env_value": None,
    }


def test_env_when_no_setting():
    app = _app(env_on=True, env_var="VW_PROMPT_LENS_ENABLED")
    assert sw.enabled(app) is True
    st = sw.status(app)
    assert st["source"] == "env" and st["env_var"] == "VW_PROMPT_LENS_ENABLED"
    assert st["env_value"] is True


def test_deprecated_env_alias_is_named():
    app = _app(env_on=True, env_var="VW_GODMODE_ENABLED")
    assert sw.enabled(app) is True
    assert sw.status(app)["env_var"] == "VW_GODMODE_ENABLED"


def test_setting_wins_over_env_both_ways():
    app = _app(env_on=True, env_var="VW_PROMPT_LENS_ENABLED", runtime=False)
    assert sw.enabled(app) is False
    assert sw.status(app)["source"] == "setting"
    app = _app(env_on=False, env_var="VW_PROMPT_LENS_ENABLED", runtime=True)
    assert sw.enabled(app) is True


def test_no_hub_is_off():
    app = _app(runtime=True)
    app.state.prompt_lens_hub = None
    assert sw.enabled(app) is False


def test_non_bool_runtime_is_ignored():
    # A MagicMock app.state (unit tests) answers any attribute: only a real bool counts.
    app = _app(env_on=False)
    app.state.prompt_lens_runtime = object()
    assert sw.enabled(app) is False


def test_env_read_live_from_settings():
    # Tests (and nothing else) swap app.state.settings; the switch reads it each time.
    app = _app()
    app.state.settings = dataclasses.replace(app.state.settings, prompt_lens_enabled=True)
    assert sw.enabled(app) is True


async def test_off_drops_memory_and_stops_streams():
    app = _app(runtime=True)
    hub = app.state.prompt_lens_hub
    hub.publish({"type": "request_start", "req_id": "r1", "request_id": "q1", "token_id": "t"})
    hub.publish(
        {"type": "delta", "req_id": "r1", "channel": "content", "text": "hi", "token_id": "t"}
    )
    app.state.prompt_lens_media.put("a" * 16, "image/png", "AAAA")
    q_all, _ = hub.subscribe()
    q_key, _ = hub.subscribe_key("t")
    assert hub.events() and app.state.prompt_lens_media.size_chars()
    assert app.state.prompt_lens_sections.size_chars() > 0

    sw.apply(app, False)

    assert sw.enabled(app) is False
    assert hub.events() == [] and hub.key_events("t") == []
    assert hub.restore([], None) == []
    assert app.state.prompt_lens_sections.size_chars() == 0
    assert app.state.prompt_lens_media.size_chars() == 0
    assert app.state.prompt_lens_media.get("a" * 16) is None
    # Every open stream gets the stop mark, last in its queue.
    for q in (q_all, q_key):
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items and items[-1] is STOP


async def test_off_stops_capture_until_on_again():
    app = _app(runtime=True)
    hub = app.state.prompt_lens_hub
    sw.apply(app, False)
    # A request already running when it went off publishes nothing more.
    hub.publish({"type": "delta", "req_id": "r1", "channel": "content", "text": "late"})
    assert hub.events() == []
    sw.apply(app, True)
    hub.publish({"type": "request_start", "req_id": "r2", "request_id": "q2", "token_id": "t"})
    assert [e["req_id"] for e in hub.events()] == ["r2"]


async def test_a_request_from_before_off_is_not_resumed_after_on():
    app = _app(runtime=True)
    hub = app.state.prompt_lens_hub
    old = hub.bound()  # what a request captures when it starts
    old.publish({"type": "request_start", "req_id": "r1", "request_id": "q1", "token_id": "t"})
    sw.apply(app, False)
    sw.apply(app, True)
    # The same request goes on streaming: nothing of it is kept or sent.
    q, _ = hub.subscribe()
    assert old.publish({"type": "delta", "req_id": "r1", "channel": "content", "text": "x"}) == 0
    assert old.publish({"type": "request_end", "req_id": "r1", "token_id": "t"}) == 0
    assert hub.events() == [] and q.empty()
    # A request that starts after it is on again is captured.
    new = hub.bound()
    new.publish({"type": "request_start", "req_id": "r2", "request_id": "q2", "token_id": "t"})
    assert [e["req_id"] for e in hub.events()] == ["r2"]


async def test_a_bound_hub_reads_through():
    app = _app(runtime=True)
    hub = app.state.prompt_lens_hub
    assert hub.bound().subscriber_count() == hub.subscriber_count()


async def test_on_to_on_keeps_memory():
    app = _app(runtime=True)
    hub = app.state.prompt_lens_hub
    hub.publish({"type": "request_start", "req_id": "r1", "request_id": "q1", "token_id": "t"})
    sw.apply(app, True)
    assert len(hub.events()) == 1


async def test_clearing_the_setting_falls_back_to_env():
    app = _app(env_on=True, env_var="VW_GODMODE_ENABLED", runtime=False)
    sw.apply(app, None)
    assert sw.enabled(app) is True
    assert sw.status(app)["source"] == "env"


async def test_load_reads_the_kv(tmp_path):
    from app.db.database import open_db
    from app.db.migrations import apply_migrations
    from app.db.repos.settings import SettingsRepo

    app = _app(env_on=True, env_var="VW_PROMPT_LENS_ENABLED")
    app.state.settings = dataclasses.replace(app.state.settings, data_dir=tmp_path)
    db_path = app.state.settings.db_path
    async with open_db(db_path) as db:
        await apply_migrations(db)
    await sw.load(app)
    assert getattr(app.state, "prompt_lens_runtime", None) is None
    async with open_db(db_path) as db:
        await SettingsRepo(db).set("prompt_lens_enabled", "false")
    await sw.load(app)
    assert app.state.prompt_lens_runtime is False
    assert sw.enabled(app) is False


def test_stop_is_the_stream_sentinel():
    from app.proxy import prompt_lens_stream

    assert prompt_lens_stream._STOP is STOP
