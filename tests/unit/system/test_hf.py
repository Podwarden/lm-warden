import pytest

from app.system.hf import HfWhoAmI, validate_hf_token


async def test_validate_hf_token_ok(httpx_mock):
    httpx_mock.add_response(
        url="https://huggingface.co/api/whoami-v2",
        json={"name": "alice", "type": "user"},
        status_code=200,
    )
    info = await validate_hf_token("hf_xxx")
    assert info == HfWhoAmI(username="alice", account_type="user")


async def test_validate_hf_token_invalid(httpx_mock):
    httpx_mock.add_response(
        url="https://huggingface.co/api/whoami-v2",
        status_code=401,
    )
    with pytest.raises(ValueError):
        await validate_hf_token("hf_bad")
