from tests.conftest import csrf_header, jwt_login, seed_admin_user


def test_current_user_id_resolves_username_to_integer(tmp_data_dir, client) -> None:
    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    headers = {**jwt_login(client), **csrf_header(client)}
    r = client.get("/api/chat2/_whoami", headers=headers)
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["user_id"], int)


def test_current_user_id_rejects_unknown_subject(tmp_data_dir, client) -> None:
    from app.auth.jwt import mint_access

    client.get("/healthz")
    seed_admin_user(tmp_data_dir / "vllm-warden.db")
    tok = mint_access("ghost", client.app.state.jwt_secret, 5)
    r = client.get("/api/chat2/_whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401
