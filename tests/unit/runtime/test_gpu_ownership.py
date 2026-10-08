import pytest

from app.runtime.gpu_ownership import GpuConflict, GpuOwnership


def test_claim_grants_exclusive_ownership():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    assert g.owner_of(0) == "m1"
    assert g.owner_of(1) == "m1"
    assert g.owner_of(2) is None


def test_claim_conflict_raises():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    with pytest.raises(GpuConflict) as ei:
        g.claim("m2", [1, 2])
    assert "1" in str(ei.value)
    assert g.owner_of(2) is None


def test_conflict_names_the_occupant_and_a_remedy():
    """The old message was `GPUs [1] already claimed` -- it named neither the
    model in the way nor anything to do about it, and it goes straight into
    the row's ``last_error`` where an operator reads it. Worse, it reads as a
    capacity problem, so people go and tune ``gpu_memory_utilization``, which
    cannot possibly help: the claim is refused before the engine is spawned.
    """
    g = GpuOwnership()
    g.claim("m1", [0], label="qwen2.5-1.5b")
    with pytest.raises(GpuConflict) as ei:
        g.claim("m2", [0], label="qwen2.5-0.5b")
    msg = str(ei.value)
    assert "GPU 0" in msg
    assert "qwen2.5-1.5b" in msg  # who is in the way
    assert "unload" in msg.lower()  # what to do about it
    assert "one loaded model per GPU" in msg  # why, so it isn't read as a bug


def test_conflict_falls_back_to_the_model_id_when_unlabelled():
    """``label`` is optional -- an unlabelled claim must still name something."""
    g = GpuOwnership()
    g.claim("m1", [0])
    with pytest.raises(GpuConflict) as ei:
        g.claim("m2", [0])
    assert "m1" in str(ei.value)


def test_conflict_across_several_gpus_names_each_occupant():
    g = GpuOwnership()
    g.claim("m1", [0], label="alpha")
    g.claim("m2", [1], label="beta")
    with pytest.raises(GpuConflict) as ei:
        g.claim("m3", [0, 1], label="gamma")
    msg = str(ei.value)
    assert "alpha" in msg
    assert "beta" in msg


def test_release_forgets_the_label_too():
    """A stale label would name a model that is gone -- worse than an id."""
    g = GpuOwnership()
    g.claim("m1", [0], label="gone")
    g.release("m1")
    g.claim("m2", [0])
    with pytest.raises(GpuConflict) as ei:
        g.claim("m3", [0])
    assert "gone" not in str(ei.value)


def test_release_frees_gpus():
    g = GpuOwnership()
    g.claim("m1", [0, 1])
    g.release("m1")
    assert g.owner_of(0) is None
    g.claim("m2", [0, 1])
    assert g.owner_of(0) == "m2"


def test_release_unknown_is_noop():
    g = GpuOwnership()
    g.release("does-not-exist")
