from app.auth.csrf import generate_csrf_token, verify_csrf_token


def test_csrf_token_round_trip():
    secret = "x" * 32
    t = generate_csrf_token("session-id-1", secret=secret)
    assert verify_csrf_token(t, "session-id-1", secret=secret) is True


def test_csrf_token_rejects_wrong_session():
    secret = "x" * 32
    t = generate_csrf_token("session-id-1", secret=secret)
    assert verify_csrf_token(t, "session-id-2", secret=secret) is False
