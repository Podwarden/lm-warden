import threading

from app.proxy.tokenizers import OFFLOOP_TOKENIZE_CHARS, TokenizerCache


class _Tok:
    def __init__(self):
        self.threads = []

    def encode(self, text):
        self.threads.append(threading.get_ident())
        return list(range(len(text) // 4))


async def test_large_text_encodes_off_loop(monkeypatch):
    cache = TokenizerCache()
    tok = _Tok()

    async def get(repo):
        return tok

    monkeypatch.setattr(cache, "get", get)
    loop_thread = threading.get_ident()
    n = await cache.count("r", "x" * (OFFLOOP_TOKENIZE_CHARS + 1))
    assert n > 0 and tok.threads[-1] != loop_thread
    await cache.count("r", "short")
    assert tok.threads[-1] == loop_thread


async def test_get_warms_up_tokenizer_once():
    from unittest.mock import patch

    calls = []

    class Tok:
        def encode(self, text):
            calls.append(text)
            return [1]

    with patch("app.proxy.tokenizers.AutoTokenizer.from_pretrained", return_value=Tok()):
        cache = TokenizerCache()
        await cache.get("r")
        await cache.get("r")
    assert calls == ["warmup"]


async def test_already_borrowed_is_retried_and_not_estimating(monkeypatch):
    cache = TokenizerCache()

    class Tok:
        n = 0

        def encode(self, text):
            Tok.n += 1
            if Tok.n == 1:
                raise RuntimeError("Already borrowed")
            return [1, 2, 3]

    async def get(repo):
        return Tok()

    monkeypatch.setattr(cache, "get", get)
    assert await cache.count("r", "hello") == 3
    assert "r" not in cache.estimating()
    # Persistent contention: estimate, but still not sticky.
    Tok.n = -(10**6)

    def always(self, text):
        raise RuntimeError("Already borrowed")

    Tok.encode = always
    assert await cache.count("r", "x" * 400) == 100
    assert "r" not in cache.estimating()
