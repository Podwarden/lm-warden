"""``_forward`` pins ``body['model']`` to the backend's ``engine_model_id``.

Every engine registers a model under ONE id: vLLM and llama.cpp under the
row's ``served_model_name`` (the flag ``plan()`` passes), MLX under the local
path it loaded (it has no ``--served-model-name`` at all). A request body
carrying any other id reaches the engine as an unknown name -- for MLX that
triggers a Hub download of ANOTHER model (Metal design s5 A). So the proxy
pins ``body['model']`` to the id the engine serves, re-serialising ONLY when
it differs from what the client sent: a body that already names the served id
(forwarded bytes byte-identical to the incoming body) never pays a
re-serialisation.

The stub backend below reports an id DIFFERENT from the row's
``served_model_name`` (``/models/x``), so a correct rewrite is always a
visible byte change -- no id can pass through verbatim by accident.
"""

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.requests import Request

from app.db.repos.tokens import TokenRow, hash_token

PLAINTEXT = "vw_validtoken1234567890abcdef12345"
ENGINE_ID = "/models/x"
# One exact, one extra whitespace run, one key order a json round-trip cannot
# reproduce: a byte-identical forward must survive them verbatim.
RAW_BODY = json.dumps(
    {"model": "x", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
    indent=2,
).encode()
# The same object's canonical form: what a re-serialisation must produce.
CANONICAL_BODY = json.dumps(
    json.loads(RAW_BODY.replace(b'"model": "x"', b'"model": "' + ENGINE_ID.encode() + b'"'))
).encode()


def _stand_in(engine_id: str, *, record: list):
    """A full ``Backend`` stand-in (every Protocol member) whose
    ``engine_model_id`` answers ``engine_id`` and records the ``model`` it was
    asked for."""

    from app.runtime.backends import BackendCapabilities, LaunchPlan

    class StandIn:
        capabilities = BackendCapabilities(
            name="stub",
            display_name="Stub",
            supports_tensor_parallel=False,
            supports_pipeline_parallel=False,
            max_gpus=None,
            openai_paths=frozenset({"/v1/completions"}),
            health_path="/health",
            metrics_path=None,
            supports_request_priority=True,
            supports_lora=False,
            supports_vision=False,
            supports_version_pin=False,
            needs_local_model_path=False,
            env_prefixes=(),
            env_exact=frozenset(),
            vram_cap_fraction=None,
        )

        def version_pin_available(self, driver):
            return False

        def bind_host(self, driver_name):
            return "127.0.0.1"

        def plan(self, model, *, port, bind_host, overrides=None, resolved=None):
            return LaunchPlan(argv=["x"], env={}, image=None, port=port)

        def health_url(self, host, port):
            return f"http://{host}:{port}/health"

        def diagnose(self, log_tail):
            return None

        def parse_metrics(self, body):
            return None

        def engine_model_id(self, model, resolved=None) -> str:
            record.append((model, resolved))
            return engine_id

    return StandIn


def _ready(client, tmp_data_dir, dp=1):
    client.get("/healthz")
    gpus = list(range(dp))
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute(
            "UPDATE setup_state SET step='done', draft=? WHERE id=1",
            (json.dumps({"allowed_gpu_indices": gpus}),),
        )
        db.execute(
            "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, gpu_indices, "
            "tensor_parallel_size, data_parallel_size, dp_affinity_enabled, dtype, "
            "max_model_len, gpu_memory_utilization, trust_remote_code, extra_args, status, "
            "pulled_bytes, pulled_total, last_error) "
            "VALUES ('m1','x','Qwen/Qwen3.5-9B','main',?,1,?,1,'auto',4096,0.9,0,'[]',"
            "'loaded',0,NULL,NULL)",
            (json.dumps(gpus), dp),
        )
        db.execute(
            "INSERT INTO api_tokens(id, name, prefix, hash, scope) VALUES (?, ?, ?, ?, ?)",
            ("tok1", "test", PLAINTEXT[:8], hash_token(PLAINTEXT), "inference"),
        )
        db.commit()
    client.app.state.supervisor._ports["m1"] = 19099
    cache = MagicMock()
    cache.count = AsyncMock(side_effect=lambda repo, text, *, fallback_repo=None: len(text))
    client.app.state.tokenizers = cache


def _model_row(client):
    from app.db.database import open_db
    from app.db.repos.models import ModelRepo

    async def get():
        async with open_db(client.app.state.settings.db_path) as db:
            return await ModelRepo(db).get("m1")

    return client.portal.call(get)


def _token_row(priority: int = 0) -> TokenRow:
    return TokenRow(
        id="tok1",
        name="test",
        prefix="vw_valid",
        scope="inference",
        allowed_models=None,
        rate_limit_rpm=None,
        rate_limit_tpm=None,
        revoked_at=None,
        last_used_at=None,
        created_at="2026-01-01",
        expires_at=None,
        rotated_at=None,
        rotated_from=None,
        priority=priority,
    )


def _run(
    client,
    raw: bytes,
    *,
    path: str,
    model,
    record: list,
    priority: int = 0,
    chunks: list | None = None,
    backend=None,
    seen: list | None = None,
):
    """Open one ``_forward``; returns (row, upstream request)."""
    from app.proxy.routes import _forward

    chunks = chunks if chunks is not None else [b'{"choices":[{"message":{"content":"a"}}]}\n']
    app = client.app
    seen = seen if seen is not None else []
    backend = backend if backend is not None else _stand_in(ENGINE_ID, record=record)()

    async def _send(request, *, stream=False, **kw):
        seen.append(request)
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/event-stream" if chunks else "application/json"}

        async def aiter():
            for c in chunks:
                yield c

        resp.aiter_bytes = aiter
        resp.aclose = AsyncMock()

        async def aread():
            return b"".join(chunks)

        resp.aread = aread
        return resp

    async def go():
        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"content-type", b"application/json")],
            "app": app,
            "query_string": b"",
            "client": ("127.0.0.1", 1),
        }
        request = Request(scope, receive)
        with (
            patch(
                "app.proxy.routes.backend_registry.get",
                return_value=backend,
            ),
            patch("httpx.AsyncClient.send", new=AsyncMock(side_effect=_send)),
        ):
            await _forward(request, model, "127.0.0.1", 19099, path, _token_row(priority))
        assert len(seen) == 1
        return seen[0]

    return client.portal.call(go)


def _forwarded_json(up) -> dict:
    return json.loads(bytes(up.content))


def test_engine_id_is_pinned_when_it_differs(tmp_data_dir, client):
    """The red-first leg: the client sent the served name ``'x'``; the engine
    serves ``'/models/x'``, so the forwarded body must carry ``'/models/x'``
    (red today: ``'x'`` is forwarded verbatim -- nothing rewrites the id)."""
    _ready(client, tmp_data_dir)
    record: list = []
    model = _model_row(client)
    up = _run(
        client,
        b'{"model": "x", "stream": false, "messages": []}',
        path="/v1/chat/completions",
        model=model,
        record=record,
        chunks=[b'{"choices":[{"message":{"content":"a"}}]}\n'],
    )
    fwd = _forwarded_json(up)
    assert fwd["model"] == ENGINE_ID
    # The pin asked for the ROW's id, passing resolved=None (the forward path
    # resolves no paths; vLLM/llama.cpp ignore it, MLX raises).
    (model, resolved) = record[-1]
    assert model.id == "m1"
    assert resolved is None


def test_equal_id_forwards_byte_identical(tmp_data_dir, client):
    """A body that already names the engine id is forwarded byte for byte:
    exact bytes, original whitespace and key order (a re-serialisation would
    produce the canonical ``json.dumps`` form instead)."""
    _ready(client, tmp_data_dir)
    record: list = []
    model = _model_row(client)
    raw = RAW_BODY.replace(b'"model": "x"', b'"model": "' + ENGINE_ID.encode() + b'"')
    up = _run(
        client,
        raw,
        path="/v1/chat/completions",
        model=model,
        record=record,
        chunks=[b'{"choices":[{"message":{"content":"a"}}]}\n'],
    )
    assert bytes(up.content) == raw


def test_priority_and_model_pin_share_one_serialisation(tmp_data_dir, client):
    """The #173 priority injection and the model-id pin both land in ONE
    re-serialised body (two fields present, no second ``json.dumps``)."""
    _ready(client, tmp_data_dir)
    record: list = []
    model = _model_row(client)
    up = _run(
        client,
        b'{"model": "x", "stream": false, "messages": []}',
        path="/v1/chat/completions",
        model=model,
        record=record,
        priority=7,
        chunks=[b'{"choices":[{"message":{"content":"a"}}]}\n'],
    )
    fwd = _forwarded_json(up)
    assert fwd["model"] == ENGINE_ID
    assert fwd["priority"] == -7
    # One serialisation of the merged object: the bytes are exactly that
    # object's canonical form, with nothing else mutated.
    assert bytes(up.content) == json.dumps(fwd).encode()


def test_missing_model_field_is_pinned(tmp_data_dir, client):
    """A body without ``model`` at all gets the engine id added (``.get`` is
    None, so the pin fires)."""
    _ready(client, tmp_data_dir)
    record: list = []
    model = _model_row(client)
    up = _run(
        client,
        b'{"stream": false, "messages": []}',
        path="/v1/chat/completions",
        model=model,
        record=record,
        chunks=[b'{"choices":[{"message":{"content":"a"}}]}\n'],
    )
    assert _forwarded_json(up)["model"] == ENGINE_ID


def test_streaming_chat_is_rewritten(tmp_data_dir, client):
    """The rewrite applies on a streaming chat body too (``stream: true``)."""
    _ready(client, tmp_data_dir)
    record: list = []
    model = _model_row(client)
    up = _run(
        client,
        b'{"model": "x", "stream": true, "messages": []}',
        path="/v1/chat/completions",
        model=model,
        record=record,
        chunks=[b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n', b"data: [DONE]\n\n"],
    )
    assert _forwarded_json(up)["model"] == ENGINE_ID


def test_completions_path_is_rewritten(tmp_data_dir, client):
    """Same pin on the completions route (its body carries ``prompt``)."""
    _ready(client, tmp_data_dir)
    record: list = []
    model = _model_row(client)
    up = _run(
        client,
        b'{"model": "x", "stream": false, "prompt": "hi"}',
        path="/v1/completions",
        model=model,
        record=record,
        chunks=[b'{"choices":[{"text":"a"}]}\n'],
    )
    assert _forwarded_json(up)["model"] == ENGINE_ID


def test_llamacpp_backed_row_is_pinned_the_same_way(tmp_data_dir, client):
    """The pin is row-driven, not engine-driven: a ``llamacpp`` row goes
    through the SAME call (no backend-name branch in the forward)."""
    _ready(client, tmp_data_dir)
    with sqlite3.connect(tmp_data_dir / "vllm-warden.db") as db:
        db.execute("UPDATE models SET backend='llamacpp' WHERE id='m1'")
        db.commit()
    record: list = []
    model = _model_row(client)
    up = _run(
        client,
        b'{"model": "x", "stream": false, "messages": []}',
        path="/v1/chat/completions",
        model=model,
        record=record,
        chunks=[b'{"choices":[{"message":{"content":"a"}}]}\n'],
    )
    assert _forwarded_json(up)["model"] == ENGINE_ID
    asked, _resolved = record[-1]
    assert asked.backend == "llamacpp"


def test_the_launch_recorded_id_wins_over_the_rows_answer(tmp_data_dir, client):
    """An engine this supervisor launched is forwarded the id the launch
    recorded (MLX's is the resolved local path, which only the launch knows);
    the backend is not asked again on the request path."""
    _ready(client, tmp_data_dir)
    client.app.state.supervisor._engine_ids["m1"] = "/snap/x"
    record: list = []
    model = _model_row(client)
    up = _run(
        client,
        b'{"model": "x", "stream": false, "messages": []}',
        path="/v1/chat/completions",
        model=model,
        record=record,
    )
    assert _forwarded_json(up)["model"] == "/snap/x"
    assert record == []


def test_a_backend_that_cannot_name_its_id_leaks_no_slot(tmp_data_dir, client):
    """MLX's engine_model_id raises ValueError without a resolved path. For an
    engine with no launch record that raise must happen BEFORE the admission
    slot is taken: nothing is forwarded and the slot count is back to zero (a
    raise after admission would hold the slot forever, one per request)."""
    import pytest

    _ready(client, tmp_data_dir)
    record: list = []
    model = _model_row(client)
    raising = _stand_in(ENGINE_ID, record=record)()

    def _no_id(model, resolved=None):
        raise ValueError("MLX needs a resolved model directory")

    raising.engine_model_id = _no_id
    seen: list = []
    with pytest.raises(ValueError, match="resolved model directory"):
        _run(
            client,
            b'{"model": "x", "stream": false, "messages": []}',
            path="/v1/chat/completions",
            model=model,
            record=record,
            backend=raising,
            seen=seen,
        )
    assert seen == []
    assert client.app.state.scheduler._inflight_for_test("m1") == 0


async def test_load_records_the_engine_id_and_unload_clears_it(tmp_path):
    """The supervisor fixes the engine id at launch (with the resolved paths
    plan() used) and forgets it with the rest of the per-engine state."""
    from app.db.repos.models import ModelRow
    from app.runtime.supervisor import Supervisor
    from tests.fakes.fake_engine import FakeDriver

    class _Settings:
        data_dir = str(tmp_path)
        hf_token_path = str(tmp_path / "tok")
        hf_cache_dir = str(tmp_path / "hf-cache")
        load_max_s = 7200.0

    row = ModelRow(
        id="m1",
        served_model_name="qwen",
        hf_repo="Qwen/Qwen3-8B-AWQ",
        hf_revision="main",
        gpu_indices=[0],
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
        extra_args=[],
        status="loaded",
        pulled_bytes=0,
        pulled_total=None,
        last_error=None,
        extra_env={},
        backend="vllm",
    )
    sup = Supervisor(_Settings(), driver=FakeDriver())
    assert sup.get_engine_model_id("m1") is None
    await sup.load(row, port=8001)
    assert sup.get_engine_model_id("m1") == "qwen"
    await sup.unload("m1", force=True)
    assert sup.get_engine_model_id("m1") is None
