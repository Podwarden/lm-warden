"""Shell status service — pure ``build_shell_status()`` contract.

The function is the backend half of the loopback-only ``GET /api/shell/status``
feed (planner iter 44): loaded models with resident bytes, used/usable memory
and pressure, and in-flight request groups. It is pure apart from calling the
injected ``footprint`` probe, and the output must carry nothing a caller
should not see: no client IPs, token ids, request ids, paths, prompt or
completion text, token counts, timings, argv, env, host or port.

The supervisor is duck-typed and stood in for here by a small fake; the
``footprint`` callable is injected, so the whole contract runs on Linux with no
live engine and no ctypes.
"""

from types import SimpleNamespace

import pytest

from app.proxy.request_registry import LiveRequest


class _FakeSupervisor:
    """Duck-typed stand-in: the four members the service reads.

    ``variants`` maps id -> SimpleNamespace(served_model_name, descriptor) or
    None (not launched by this supervisor). ``states`` maps id -> an
    enum-like with ``.value`` or None. ``pids`` maps id -> int or None.
    """

    def __init__(self, engine_ids, variants=None, states=None, pids=None):
        self._ids = set(engine_ids)
        self._variants = variants or {}
        self._states = states or {}
        self._pids = pids or {}

    def live_engine_ids(self):
        return set(self._ids)

    def get_variant(self, model_id):
        return self._variants.get(model_id)

    def get_state(self, model_id):
        return self._states.get(model_id)

    def get_pid(self, model_id):
        return self._pids.get(model_id)


def _req(who, model, **extra):
    return LiveRequest(
        id=extra.pop("id", "req-0"),
        token_id=extra.pop("token_id", None),
        token_name=who,
        client_ip=extra.pop("client_ip", None),
        model=model,
        model_row_id=model,
        path=extra.pop("path", "/v1/chat/completions"),
        prompt_tokens=extra.pop("prompt_tokens", 0),
        max_model_len=None,
        started_monotonic=0.0,
        started_iso="2026-01-01T00:00:00Z",
        **extra,
    )


def test_red_first_two_engines_sorted_and_summed():
    from app.system.shell_status import build_shell_status

    sup = _FakeSupervisor(
        ["b-model", "a-model"],
        variants={
            "b-model": SimpleNamespace(served_model_name="b-model", descriptor={"backend": "mlx"}),
            "a-model": SimpleNamespace(
                served_model_name="a-model", descriptor={"backend": "llamacpp"}
            ),
        },
        states={
            "b-model": SimpleNamespace(value="ready"),
            "a-model": SimpleNamespace(value="warming"),
        },
        pids={"b-model": 101, "a-model": 102},
    )
    footprint = {101: 2_000_000_000, 102: 3_000_000_000}

    status = build_shell_status(
        sup, [], footprint=lambda pid: footprint[pid], usable_bytes=None, pressure_level=None
    )

    # sorted by name: a-model before b-model, with backend / state / bytes
    assert [m.name for m in status.models] == ["a-model", "b-model"]
    assert [m.backend for m in status.models] == ["llamacpp", "mlx"]
    assert [m.state for m in status.models] == ["warming", "ready"]
    assert [m.bytes for m in status.models] == [3_000_000_000, 2_000_000_000]
    assert status.memory.used_bytes == 5_000_000_000


def test_no_engines_no_requests_empty():
    from app.system.shell_status import build_shell_status

    sup = _FakeSupervisor([])
    status = build_shell_status(
        sup, [], footprint=lambda pid: 0, usable_bytes=None, pressure_level=None
    )

    assert status.models == []
    assert status.memory.used_bytes == 0
    assert status.in_flight.count == 0
    assert status.in_flight.groups == []


def test_name_tie_sorted_by_id():
    from app.system.shell_status import build_shell_status

    # two engines serving the same name; tie on name broken by id
    sup = _FakeSupervisor(
        ["z-id", "a-id"],
        variants={
            "z-id": SimpleNamespace(served_model_name="same-name", descriptor={"backend": "mlx"}),
            "a-id": SimpleNamespace(served_model_name="same-name", descriptor={"backend": "mlx"}),
        },
        states={"z-id": SimpleNamespace(value="ready"), "a-id": SimpleNamespace(value="ready")},
        pids={"z-id": 1, "a-id": 2},
    )
    status = build_shell_status(
        sup, [], footprint=lambda pid: 10, usable_bytes=None, pressure_level=None
    )
    assert [m.id for m in status.models] == ["a-id", "z-id"]


def test_get_variant_none_and_state_none_and_descriptor_backend_variants():
    from app.system.shell_status import build_shell_status

    sup = _FakeSupervisor(
        ["n-variant", "no-backend-key", "empty-backend"],
        variants={
            "no-backend-key": SimpleNamespace(
                served_model_name="nbk", descriptor={"engine_image": "x"}
            ),
            "empty-backend": SimpleNamespace(served_model_name="eb", descriptor={"backend": ""}),
            # "n-variant" has no entry -> get_variant returns None
        },
        states={"no-backend-key": SimpleNamespace(value="ready")},
        pids={"n-variant": 7, "no-backend-key": 8, "empty-backend": 9},
    )
    status = build_shell_status(
        sup, [], footprint=lambda pid: 5, usable_bytes=None, pressure_level=None
    )
    by_id = {m.id: m for m in status.models}
    # get_variant None -> name is the id, backend 'unknown'
    assert by_id["n-variant"].name == "n-variant"
    assert by_id["n-variant"].backend == "unknown"
    # descriptor without 'backend' -> 'unknown'
    assert by_id["no-backend-key"].backend == "unknown"
    # descriptor with '' -> 'unknown'
    assert by_id["empty-backend"].backend == "unknown"
    # get_state None -> 'unknown'
    assert by_id["n-variant"].state == "unknown"
    assert by_id["empty-backend"].state == "unknown"


def test_bytes_none_legs_and_used_bytes_counts_only_known():
    from app.system.shell_status import build_shell_status

    def footprint(pid):
        if pid == 0:
            return None
        if pid == 1:
            raise OSError("no such process")
        if pid == 3:
            return -5
        if pid == 2:
            return 1_000
        return None  # pid 4

    sup = _FakeSupervisor(
        ["p-none", "p-zero", "p-raise", "p-neg", "p-ok"],
        variants={
            mid: SimpleNamespace(served_model_name=mid, descriptor={"backend": "mlx"})
            for mid in ["p-none", "p-zero", "p-raise", "p-neg", "p-ok"]
        },
        states={
            mid: SimpleNamespace(value="ready")
            for mid in ["p-none", "p-zero", "p-raise", "p-neg", "p-ok"]
        },
        pids={"p-none": None, "p-zero": 0, "p-raise": 1, "p-neg": 3, "p-ok": 2},
    )
    status = build_shell_status(
        sup, [], footprint=footprint, usable_bytes=None, pressure_level=None
    )
    by_id = {m.id: m for m in status.models}
    assert by_id["p-none"].bytes is None
    assert by_id["p-zero"].bytes is None
    assert by_id["p-raise"].bytes is None
    assert by_id["p-neg"].bytes is None
    assert by_id["p-ok"].bytes == 1_000
    assert status.memory.used_bytes == 1_000


def test_usable_bytes_and_pressure_level_clamping():
    from app.system.shell_status import build_shell_status

    sup = _FakeSupervisor([])

    zero = build_shell_status(sup, [], footprint=lambda p: 0, usable_bytes=0, pressure_level=0)
    assert zero.memory.usable_bytes is None
    assert zero.memory.pressure_level == 0

    one = build_shell_status(sup, [], footprint=lambda p: 0, usable_bytes=1, pressure_level=-1)
    assert one.memory.usable_bytes == 1
    assert one.memory.pressure_level is None

    none = build_shell_status(
        sup, [], footprint=lambda p: 0, usable_bytes=None, pressure_level=None
    )
    assert none.memory.usable_bytes is None
    assert none.memory.pressure_level is None


def test_in_flight_groups_and_ordering():
    from app.system.shell_status import build_shell_status

    requests = (
        _req("alice", "m1"),
        _req("alice", "m1"),
        _req("alice", "m1"),
        _req("bob", "m1"),
        _req(None, "m2"),
    )
    sup = _FakeSupervisor([])
    status = build_shell_status(
        sup, requests, footprint=lambda p: 0, usable_bytes=None, pressure_level=None
    )

    assert status.in_flight.count == 5
    groups = status.in_flight.groups
    assert [(g.who, g.model, g.requests) for g in groups] == [
        ("alice", "m1", 3),
        ("bob", "m1", 1),
        ("unknown", "m2", 1),
    ]


def test_in_flight_tie_broken_by_who_then_model():
    from app.system.shell_status import build_shell_status

    requests = (
        _req("zeta", "m2"),
        _req("alpha", "m2"),
        _req("alpha", "m1"),
        _req("zeta", "m1"),
    )
    sup = _FakeSupervisor([])
    status = build_shell_status(
        sup, requests, footprint=lambda p: 0, usable_bytes=None, pressure_level=None
    )
    # all ties at 1 -> sort by who, then model
    assert [(g.who, g.model, g.requests) for g in status.in_flight.groups] == [
        ("alpha", "m1", 1),
        ("alpha", "m2", 1),
        ("zeta", "m1", 1),
        ("zeta", "m2", 1),
    ]


def test_privacy_no_secrets_or_content_in_output():
    from app.system.shell_status import build_shell_status

    requests = (
        _req(
            "alice",
            "m1",
            id="req-XYZ",
            client_ip="203.0.113.9",
            token_id="tok-SECRET-ID",
            path="/v1/chat/completions",
            prompt_tokens=42,
        ),
    )
    sup = _FakeSupervisor(
        ["eng"],
        variants={"eng": SimpleNamespace(served_model_name="eng", descriptor={"backend": "mlx"})},
    )
    status = build_shell_status(
        sup, requests, footprint=lambda p: 0, usable_bytes=None, pressure_level=None
    )

    dumped = status.model_dump_json()
    for secret in ("203.0.113.9", "tok-SECRET-ID", "req-XYZ", "/v1/chat/completions"):
        assert secret not in dumped


def test_field_sets_exactly_as_contracted():
    from app.system.shell_status import (
        ShellInFlight,
        ShellInFlightGroup,
        ShellMemory,
        ShellModel,
        ShellStatus,
    )

    assert set(ShellStatus.model_fields) == {"models", "memory", "in_flight"}
    assert set(ShellModel.model_fields) == {"id", "name", "backend", "state", "bytes"}
    assert set(ShellMemory.model_fields) == {"used_bytes", "usable_bytes", "pressure_level"}
    assert set(ShellInFlight.model_fields) == {"count", "groups"}
    assert set(ShellInFlightGroup.model_fields) == {"who", "model", "requests"}


def test_extra_forbid_raises():
    from pydantic import ValidationError

    from app.system.shell_status import ShellModel

    with pytest.raises(ValidationError):
        ShellModel(id="i", name="n", backend="b", state="s", bytes=1, client_ip="1.2.3.4")


def test_sort_by_name_not_by_id():
    from app.system.shell_status import build_shell_status

    # id order and name order disagree: the served name wins
    sup = _FakeSupervisor(
        ["a-id", "z-id"],
        variants={
            "a-id": SimpleNamespace(served_model_name="zz-name", descriptor={"backend": "mlx"}),
            "z-id": SimpleNamespace(served_model_name="aa-name", descriptor={"backend": "mlx"}),
        },
    )
    status = build_shell_status(
        sup, [], footprint=lambda pid: 1, usable_bytes=None, pressure_level=None
    )
    assert [m.id for m in status.models] == ["z-id", "a-id"]


def test_non_positive_pid_never_probed_and_footprint_none_is_none():
    from app.system.shell_status import build_shell_status

    probed = []

    def footprint(pid):
        probed.append(pid)
        return None if pid == 4 else 7

    ids = ["p-zero", "p-minus-one", "p-fp-none", "p-one"]
    sup = _FakeSupervisor(
        ids,
        variants={
            mid: SimpleNamespace(served_model_name=mid, descriptor={"backend": "mlx"})
            for mid in ids
        },
        pids={"p-zero": 0, "p-minus-one": -1, "p-fp-none": 4, "p-one": 1},
    )
    status = build_shell_status(
        sup, [], footprint=footprint, usable_bytes=None, pressure_level=None
    )
    by_id = {m.id: m for m in status.models}
    # pid 0 and pid -1: no known process, the probe is never called
    assert by_id["p-zero"].bytes is None
    assert by_id["p-minus-one"].bytes is None
    assert sorted(probed) == [1, 4]
    # a positive pid whose footprint is None -> None; pid 1 is the at-bound leg
    assert by_id["p-fp-none"].bytes is None
    assert by_id["p-one"].bytes == 7
    assert status.memory.used_bytes == 7
