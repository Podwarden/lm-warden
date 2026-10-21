"""422 validation errors must not echo request values.

FastAPI's default ``RequestValidationError`` handler returns ``exc.errors()``
verbatim, and each error's ``input`` is the offending value -- for a model-level
validator that is the WHOLE body (``extra_args`` / ``extra_env`` values that may
hold secrets). ``app/utils/validation_errors.py`` keeps exactly ``loc`` /
``msg`` / ``type`` per error, in order, and drops ``input``, ``ctx``, ``url`` and
every other key.
"""


def test_model_level_validator_error_drops_secrets_but_keeps_message() -> None:
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, model_validator

    from app.utils.validation_errors import redacted_validation_exception_handler

    class Body(BaseModel):
        extra_args: list[str]
        extra_env: dict[str, str]

        @model_validator(mode="after")
        def _refuse_api_key(self) -> "Body":
            if "--api-key" in self.extra_args:
                raise ValueError("refused: --api-key in extra_args")
            return self

    app = FastAPI()
    app.add_exception_handler(RequestValidationError, redacted_validation_exception_handler)

    @app.post("/x")
    async def x(body: Body) -> dict:
        return {"ok": True}

    r = TestClient(app).post(
        "/x",
        json={"extra_args": ["--api-key", "sk-SECRET"], "extra_env": {"HF_TOKEN": "hf_SECRET"}},
    )
    assert r.status_code == 422, r.text
    assert "sk-SECRET" not in r.text
    assert "hf_SECRET" not in r.text
    # The message still names the field and the reason.
    assert "refused: --api-key in extra_args" in r.text
    assert r.json() == {
        "detail": [
            {
                "loc": ["body"],
                "msg": "Value error, refused: --api-key in extra_args",
                "type": "value_error",
            }
        ]
    }


def test_field_level_error_names_the_field_without_the_value() -> None:
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    from app.utils.validation_errors import redacted_validation_exception_handler

    class Body(BaseModel):
        token: str

    app = FastAPI()
    app.add_exception_handler(RequestValidationError, redacted_validation_exception_handler)

    @app.post("/y")
    async def y(body: Body) -> dict:
        return {"ok": True}

    r = TestClient(app).post("/y", json={"token": 12345})
    assert r.status_code == 422, r.text
    assert "12345" not in r.text
    assert r.json() == {
        "detail": [
            {
                "loc": ["body", "token"],
                "msg": "Input should be a valid string",
                "type": "string_type",
            }
        ]
    }


def test_missing_required_field_keeps_loc_and_type_with_no_input() -> None:
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    from app.utils.validation_errors import redacted_validation_exception_handler

    class Body(BaseModel):
        name: str
        required: int

    app = FastAPI()
    app.add_exception_handler(RequestValidationError, redacted_validation_exception_handler)

    @app.post("/z")
    async def z(body: Body) -> dict:
        return {"ok": True}

    r = TestClient(app).post("/z", json={"name": "n"})
    assert r.status_code == 422, r.text
    details = r.json()["detail"]
    missing = [d for d in details if d["type"] == "missing"]
    assert len(missing) == 1
    assert missing[0]["loc"] == ["body", "required"]
    assert missing[0] == {
        "loc": ["body", "required"],
        "msg": "Field required",
        "type": "missing",
    }


def test_redactor_drops_everything_but_loc_msg_type_in_order() -> None:
    from app.utils.validation_errors import redact_validation_errors

    two = [
        {
            "loc": ("body", "a"),
            "msg": "first",
            "type": "t1",
            "input": "SECRET-A",
            "ctx": {"k": "v"},
            "url": "http://x",
        },
        {"loc": ["b"], "msg": "second", "type": "t2", "input": {"nested": "SECRET-B"}, "extra": 1},
    ]
    out = redact_validation_errors(two)
    # Exactly the three keys, in the input order.
    assert out == [
        {"loc": ["body", "a"], "msg": "first", "type": "t1"},
        {"loc": ["b"], "msg": "second", "type": "t2"},
    ]
    # Key order inside each error is loc, msg, type.
    assert all(list(e.keys()) == ["loc", "msg", "type"] for e in out)
    # loc is a fresh list, not the caller's tuple.
    assert out[0]["loc"] == ["body", "a"] and isinstance(out[0]["loc"], list)


def test_redactor_on_empty_list() -> None:
    from app.utils.validation_errors import redact_validation_errors

    assert redact_validation_errors([]) == []


def test_redactor_accepts_a_tuple_of_mappings() -> None:
    from app.utils.validation_errors import redact_validation_errors

    out = redact_validation_errors(({"loc": ("a",), "msg": "m", "type": "t", "input": "i"},))
    assert out == [{"loc": ["a"], "msg": "m", "type": "t"}]


def test_writer_validation_detail_is_the_redactor_output() -> None:
    """The PATCH path's hand-built 422 detail goes through the same redactor:
    exactly loc / msg / type, no echoed value."""
    import pydantic
    from pydantic import BaseModel, model_validator

    from app.models.writer import _validation_detail

    class Row(BaseModel):
        extra_args: list[str]
        max_len: int

        @model_validator(mode="after")
        def _refuse(self) -> "Row":
            raise ValueError("refused: --api-key in extra_args")

    try:
        Row.model_validate({"extra_args": ["--api-key", "sk-SECRET"], "max_len": "x-SECRET"})
    except pydantic.ValidationError as exc:
        detail = _validation_detail(exc)
    else:  # pragma: no cover - the model always raises
        raise AssertionError("expected a ValidationError")
    assert detail == [
        {
            "loc": ["max_len"],
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "type": "int_parsing",
        }
    ]
    assert "SECRET" not in repr(detail)
