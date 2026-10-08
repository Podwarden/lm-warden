import json
import os
import sqlite3

import pytest

from app.chat2.catalog import (
    chat_template_suggests_reasoning,
    config_suggests_vision,
    list_catalog,
    read_hf_chat_template,
    read_hf_config,
    read_max_position_embeddings,
    reasoning_efforts_from_template,
)

THINKING_TEMPLATE = (
    "{%- if enable_thinking is undefined or enable_thinking is true %}\n"
    "    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
    "{%- endif %}\n"
)
PLAIN_TEMPLATE = "{{ messages[0]['content'] }}"


def _insert_model(
    db_path, *, served="qwen", status="loaded", max_model_len=None, hf_repo="org/qwen", tools=None
):
    c = sqlite3.connect(db_path)
    c.execute(
        "INSERT INTO models(id, served_model_name, hf_repo, gpu_indices, tensor_parallel_size, "
        "status, max_model_len, supports_tools, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
        (served + "-id", served, hf_repo, "[0]", 1, status, max_model_len, tools),
    )
    c.commit()


def test_read_max_position_embeddings_from_hf_cache(tmp_path) -> None:
    snap = tmp_path / "models--org--qwen" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    assert read_max_position_embeddings(tmp_path, "org/qwen") == 32768
    assert read_max_position_embeddings(tmp_path, "org/missing") is None


@pytest.mark.asyncio
async def test_list_catalog_resolves_window_and_filters_loaded(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _insert_model(db_path, served="a", max_model_len=8192, tools=1)
    _insert_model(db_path, served="b", max_model_len=None, hf_repo="org/b")
    _insert_model(db_path, served="c", status="pulled")
    snap = tmp_data_dir / "hf-cache" / "models--org--b" / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    cat = await list_catalog(client.app.state.settings)
    by_id = {m.id: m for m in cat}
    assert set(by_id) == {"a", "b"}
    assert by_id["a"].context_window == 8192 and by_id["a"].supports_tools is True
    assert by_id["b"].context_window == 4096 and by_id["b"].supports_tools is False


def _set_vision(db_path, served, value):
    c = sqlite3.connect(db_path)
    c.execute("UPDATE models SET supports_vision = ? WHERE served_model_name = ?", (value, served))
    c.commit()


def _write_config(tmp_data_dir, repo, config):
    snap = tmp_data_dir / "hf-cache" / f"models--{repo.replace('/', '--')}" / "snapshots" / "s"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(config))


def _set_reasoning(db_path, served, value):
    c = sqlite3.connect(db_path)
    c.execute(
        "UPDATE models SET supports_reasoning = ? WHERE served_model_name = ?", (value, served)
    )
    c.commit()


def _snap_dir(tmp_data_dir, repo):
    snap = tmp_data_dir / "hf-cache" / f"models--{repo.replace('/', '--')}" / "snapshots" / "s"
    snap.mkdir(parents=True, exist_ok=True)
    return snap


def _write_chat_template_jinja(tmp_data_dir, repo, template):
    (_snap_dir(tmp_data_dir, repo) / "chat_template.jinja").write_text(template)


def _write_tokenizer_config(tmp_data_dir, repo, *, chat_template=None):
    data = {} if chat_template is None else {"chat_template": chat_template}
    (_snap_dir(tmp_data_dir, repo) / "tokenizer_config.json").write_text(json.dumps(data))


def test_config_suggests_vision_reads_both_signals() -> None:
    assert config_suggests_vision({"vision_config": {"hidden_size": 1024}}) is True
    assert config_suggests_vision({"architectures": ["Gemma3ForConditionalGeneration"]}) is True
    assert config_suggests_vision({"architectures": ["Qwen3ForCausalLM"]}) is False
    assert config_suggests_vision({"max_position_embeddings": 4096}) is False
    # unreadable / absent config is not evidence of anything
    assert config_suggests_vision(None) is False


@pytest.mark.asyncio
async def test_supports_vision_is_auto_detected_only_when_the_column_is_null(
    tmp_data_dir, client
) -> None:
    """The 2026-08 incident: a genuinely multimodal model served every pasted
    image as "[image omitted]" because nothing ever wrote `supports_vision`, and
    the catalog collapsed NULL and 0 into the same False.

    NULL now means "ask the on-disk HF config"; an explicit 0 or 1 is an
    operator decision and always wins over whatever the config says.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    for served, repo in (
        ("auto-vlm", "org/auto-vlm"),
        ("auto-arch", "org/auto-arch"),
        ("auto-text", "org/auto-text"),
        ("forced-off", "org/forced-off"),
        ("forced-on", "org/forced-on"),
        ("no-config", "org/no-config"),
    ):
        _insert_model(db_path, served=served, max_model_len=8192, hf_repo=repo)
    _write_config(tmp_data_dir, "org/auto-vlm", {"vision_config": {"hidden_size": 8}})
    _write_config(
        tmp_data_dir, "org/auto-arch", {"architectures": ["Llama4ForConditionalGeneration"]}
    )
    _write_config(tmp_data_dir, "org/auto-text", {"architectures": ["Qwen3ForCausalLM"]})
    _write_config(tmp_data_dir, "org/forced-off", {"vision_config": {"hidden_size": 8}})
    _write_config(tmp_data_dir, "org/forced-on", {"architectures": ["Qwen3ForCausalLM"]})
    _set_vision(db_path, "forced-off", 0)
    _set_vision(db_path, "forced-on", 1)

    by_id = {m.id: m for m in await list_catalog(client.app.state.settings)}
    assert by_id["auto-vlm"].supports_vision is True
    assert by_id["auto-arch"].supports_vision is True
    assert by_id["auto-text"].supports_vision is False
    assert by_id["forced-off"].supports_vision is False, "explicit 0 must beat the config"
    assert by_id["forced-on"].supports_vision is True, "explicit 1 needs no config at all"
    assert by_id["no-config"].supports_vision is False, "unreadable config -> False"
    # tools/reasoning stay purely manual: NULL still reads as False
    assert by_id["auto-vlm"].supports_tools is False
    assert by_id["auto-vlm"].supports_reasoning is False


def test_read_hf_config_is_cached_until_the_file_changes(tmp_path) -> None:
    """The models poll hits this every 60s per model; re-parsing an unchanged
    config.json each time is pure waste. The cache key is the file's identity
    (path + mtime + size), so an edited config is still picked up."""
    snap = tmp_path / "models--org--c" / "snapshots" / "s"
    snap.mkdir(parents=True)
    cfg = snap / "config.json"
    cfg.write_text(json.dumps({"vision_config": {"hidden_size": 8}}))

    first = read_hf_config(tmp_path, "org/c")
    assert first is not None and "vision_config" in first
    # same file -> same cached object, not merely an equal one
    assert read_hf_config(tmp_path, "org/c") is first

    cfg.write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    os.utime(cfg, ns=(1_000_000_000, 1_700_000_000_000_000_000))
    fresh = read_hf_config(tmp_path, "org/c")
    assert fresh is not None and "vision_config" not in fresh
    assert config_suggests_vision(fresh) is False


@pytest.mark.asyncio
async def test_vision_detection_prefers_hf_config_repo(tmp_data_dir, client) -> None:
    """#106's `hf_config_repo` exists precisely because some models ship their
    weights and their config under different repos. The detection path has to
    look where the config actually is."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _insert_model(db_path, served="split", max_model_len=8192, hf_repo="org/split-weights")
    c = sqlite3.connect(db_path)
    c.execute(
        "UPDATE models SET hf_config_repo = ? WHERE served_model_name = 'split'",
        ("org/split-config",),
    )
    c.commit()
    # the weights repo has no config at all; the config repo says multimodal
    _write_config(tmp_data_dir, "org/split-config", {"vision_config": {"hidden_size": 8}})

    by_id = {m.id: m for m in await list_catalog(client.app.state.settings)}
    assert by_id["split"].supports_vision is True


def test_chat_template_suggests_reasoning_reads_enable_thinking() -> None:
    assert chat_template_suggests_reasoning(THINKING_TEMPLATE) is True
    assert chat_template_suggests_reasoning(PLAIN_TEMPLATE) is False
    # unreadable / absent template is not evidence of anything
    assert chat_template_suggests_reasoning(None) is False
    assert chat_template_suggests_reasoning("") is False


def test_read_hf_chat_template_prefers_standalone_jinja_file(tmp_path) -> None:
    """Newer repos (e.g. Qwen/Qwen3.8-27B) ship `chat_template.jinja` as a
    separate file and have NO `chat_template` key in `tokenizer_config.json`
    at all. The standalone file must be checked first."""
    snap = tmp_path / "models--org--qwen3" / "snapshots" / "s"
    snap.mkdir(parents=True)
    (snap / "chat_template.jinja").write_text(THINKING_TEMPLATE)
    (snap / "tokenizer_config.json").write_text(json.dumps({}))

    template = read_hf_chat_template(tmp_path, "org/qwen3")
    assert template is not None and "enable_thinking" in template


def test_read_hf_chat_template_falls_back_to_tokenizer_config(tmp_path) -> None:
    """The legacy shape: no standalone file, template embedded in
    `tokenizer_config.json["chat_template"]`."""
    snap = tmp_path / "models--org--legacy" / "snapshots" / "s"
    snap.mkdir(parents=True)
    (snap / "tokenizer_config.json").write_text(json.dumps({"chat_template": THINKING_TEMPLATE}))

    template = read_hf_chat_template(tmp_path, "org/legacy")
    assert template is not None and "enable_thinking" in template


def test_read_hf_chat_template_missing_or_unreadable_repo_is_none(tmp_path) -> None:
    assert read_hf_chat_template(tmp_path, "org/not-in-cache") is None

    snap = tmp_path / "models--org--broken" / "snapshots" / "s"
    snap.mkdir(parents=True)
    (snap / "tokenizer_config.json").write_text("{not json")
    assert read_hf_chat_template(tmp_path, "org/broken") is None

    snap2 = tmp_path / "models--org--no-template-key" / "snapshots" / "s"
    snap2.mkdir(parents=True)
    (snap2 / "tokenizer_config.json").write_text(json.dumps({"some_other_key": 1}))
    assert read_hf_chat_template(tmp_path, "org/no-template-key") is None


@pytest.mark.asyncio
async def test_supports_reasoning_is_auto_detected_only_when_the_column_is_null(
    tmp_data_dir, client
) -> None:
    """#239: the chat's "Enable thinking" toggle was invisible on every newly
    registered reasoning model because NULL collapsed straight to False.

    NULL now means "ask the on-disk chat template"; an explicit 0 or 1 is an
    operator decision and always wins over whatever the template suggests.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    for served, repo in (
        ("auto-thinking-jinja", "org/auto-thinking-jinja"),
        ("auto-thinking-tokcfg", "org/auto-thinking-tokcfg"),
        ("auto-plain", "org/auto-plain"),
        ("forced-off", "org/forced-off"),
        ("forced-on", "org/forced-on"),
        ("no-template", "org/no-template"),
    ):
        _insert_model(db_path, served=served, max_model_len=8192, hf_repo=repo)
    _write_chat_template_jinja(tmp_data_dir, "org/auto-thinking-jinja", THINKING_TEMPLATE)
    _write_tokenizer_config(
        tmp_data_dir, "org/auto-thinking-tokcfg", chat_template=THINKING_TEMPLATE
    )
    _write_chat_template_jinja(tmp_data_dir, "org/auto-plain", PLAIN_TEMPLATE)
    _write_chat_template_jinja(tmp_data_dir, "org/forced-off", THINKING_TEMPLATE)
    _write_chat_template_jinja(tmp_data_dir, "org/forced-on", PLAIN_TEMPLATE)
    _set_reasoning(db_path, "forced-off", 0)
    _set_reasoning(db_path, "forced-on", 1)

    by_id = {m.id: m for m in await list_catalog(client.app.state.settings)}
    assert by_id["auto-thinking-jinja"].supports_reasoning is True
    assert by_id["auto-thinking-tokcfg"].supports_reasoning is True
    assert by_id["auto-plain"].supports_reasoning is False
    assert by_id["forced-off"].supports_reasoning is False, "explicit 0 must beat the template"
    assert by_id["forced-on"].supports_reasoning is True, "explicit 1 needs no template at all"
    assert by_id["no-template"].supports_reasoning is False, "missing template -> False"
    # vision/tools stay unaffected by reasoning detection
    assert by_id["auto-thinking-jinja"].supports_vision is False
    assert by_id["auto-thinking-jinja"].supports_tools is False


@pytest.mark.asyncio
async def test_reasoning_detection_prefers_hf_config_repo(tmp_data_dir, client) -> None:
    """Same split-repo shape as #106's vision case: the weights repo carries
    no template at all, and `hf_config_repo` is where it actually lives."""
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _insert_model(db_path, served="split", max_model_len=8192, hf_repo="org/split-weights")
    c = sqlite3.connect(db_path)
    c.execute(
        "UPDATE models SET hf_config_repo = ? WHERE served_model_name = 'split'",
        ("org/split-config",),
    )
    c.commit()
    _write_chat_template_jinja(tmp_data_dir, "org/split-config", THINKING_TEMPLATE)

    by_id = {m.id: m for m in await list_catalog(client.app.state.settings)}
    assert by_id["split"].supports_reasoning is True


# The membership test Qwen3.8's chat_template.jinja runs before `raise_exception`
# (#241). The tuple is the vocabulary; nothing about it is assumed.
_QWEN38_TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort: ' ~ resolved_reasoning_effort) }}
    {%- endif %}
{%- endif %}
"""


def test_reasoning_efforts_are_read_from_the_template_not_assumed() -> None:
    assert reasoning_efforts_from_template(_QWEN38_TEMPLATE) == ("xhigh", "medium", "low")
    # a template that never reads the kwarg ignores it -- nothing to offer
    assert reasoning_efforts_from_template("{% if enable_thinking %}<think>{% endif %}") == ()
    # one that reads it without stating its accepted set has a vocabulary we
    # cannot enumerate, and a guess could be the value it raises on
    assert reasoning_efforts_from_template("{% if reasoning_effort == 'low' %}x{% endif %}") == ()
    assert reasoning_efforts_from_template(None) == ()
    # a different family with a different vocabulary comes through verbatim
    assert reasoning_efforts_from_template(
        '{% if reasoning_effort not in ("minimal", "max") %}{{ raise_exception(\'x\') }}{% endif %}'
    ) == ("minimal", "max")


@pytest.mark.asyncio
async def test_catalog_publishes_the_template_vocabulary(tmp_data_dir, client) -> None:
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _insert_model(db_path, served="a", max_model_len=8192)
    _insert_model(db_path, served="b", max_model_len=8192, hf_repo="org/b")
    # an operator's explicit "does not reason" -- the template is not even read
    _insert_model(db_path, served="c", max_model_len=8192, hf_repo="org/c")
    c = sqlite3.connect(db_path)
    c.execute("UPDATE models SET supports_reasoning = 0 WHERE served_model_name = 'c'")
    c.commit()
    for repo in ("org/qwen", "org/c"):
        snap = tmp_data_dir / "hf-cache" / f"models--{repo.replace('/', '--')}" / "snapshots" / "x"
        snap.mkdir(parents=True)
        (snap / "chat_template.jinja").write_text(_QWEN38_TEMPLATE)
    by_id = {m.id: m for m in await list_catalog(client.app.state.settings)}
    assert by_id["a"].reasoning_efforts == ("xhigh", "medium", "low")
    assert by_id["a"].as_dict()["reasoning_efforts"] == ["xhigh", "medium", "low"]
    assert by_id["b"].reasoning_efforts == () and by_id["b"].as_dict()["reasoning_efforts"] == []
    assert by_id["c"].reasoning_efforts == ()
