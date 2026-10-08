from app.proxy.session_turns import SessionTurns


class _Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_counts_per_session_and_none_passes_through():
    st = SessionTurns()
    assert [st.next("a"), st.next("a"), st.next("b"), st.next("a")] == [0, 1, 0, 2]
    assert st.next(None) is None


def test_lru_eviction_and_ttl():
    clock = _Clock()
    st = SessionTurns(max_entries=2, ttl_s=10.0, clock=clock)
    st.next("a")
    st.next("b")
    st.next("c")  # "a" evicted
    assert st.next("a") == 0
    clock.t = 11.0
    assert st.next("c") == 0  # idle past ttl → restarts
