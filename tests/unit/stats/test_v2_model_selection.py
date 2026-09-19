"""``GET /api/stats/v2/overview?models=`` — the model selection.

The page listed the loaded models and then showed one combined set of numbers
with no way to ask about one of them. This module pins what a selection means,
per series, because the three data sources carry different dimensions:

  tokens / tps   ``model_samples`` has a model_id, so a selection is an exact
                 partition -- every model selected sums to the unfiltered total.
  vram / util /  no model column anywhere. The selection resolves to the union
  power          of the models' ``gpu_indices``, and the cards are filtered by
                 that: "this model's VRAM" is "the VRAM of the cards it holds".

The tests below are written against a fixture with two models on two separate
cards, which is the operator's actual deployment: llama-3.1-8b on vLLM/GPU 1
and qwen3.8-27b on llama.cpp/GPU 0.
"""

import sqlite3
import time

from tests.conftest import jwt_login, seed_admin_user


def _seed_two_models(db_path):
    """Two loaded models, one card each, with disjoint token traffic.

    The numbers are chosen so a reviewer can check any assertion by hand and so
    that no two of them collide: GPU 0 is the busy card (util 90, 12 GiB used),
    GPU 1 the quiet one (util 10, 2 GiB used); qwen owns GPU 0 and llama owns
    GPU 1.
    """
    seed_admin_user(db_path)
    now_min = int(time.time() // 60)
    with sqlite3.connect(db_path) as db:
        for mid, name, gpus in (
            ("m-qwen", "qwen3.8-27b", "[0]"),
            ("m-llama", "llama-3.1-8b", "[1]"),
        ):
            db.execute(
                "INSERT INTO models(id, served_model_name, hf_repo, hf_revision, "
                "gpu_indices, tensor_parallel_size, dtype, max_model_len, "
                "gpu_memory_utilization, trust_remote_code, extra_args, status) "
                f"VALUES ('{mid}', '{name}', 'o/r', 'main', '{gpus}', 1, "
                "NULL, NULL, 0.9, 0, '[]', 'loaded')"
            )
            db.execute(
                "INSERT INTO model_runtime(model_id, pid, port) "
                f"VALUES ('{mid}', 1, 10000)"
            )
        db.executemany(
            "INSERT INTO gpu_samples(gpu_index, minute, utilization_pct, "
            "memory_used_mib, memory_total_mib, name) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (0, now_min, 90, 12000, 16000, "Quadro RTX 5000"),
                (1, now_min, 10, 2000, 16000, "NVIDIA RTX A4000"),
            ],
        )
        db.executemany(
            "INSERT INTO power_samples(gpu_idx, minute, watts_sum, samples) "
            "VALUES (?, ?, ?, ?)",
            [(0, now_min, 200.0, 1), (1, now_min, 50.0, 1)],
        )
        # qwen: 600 tokens this minute. llama: 60. Total 660 -> tps 11.0.
        db.executemany(
            "INSERT INTO model_samples(model_id, minute, requests, "
            "prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?)",
            [
                ("m-qwen", now_min, 3, 400, 200),
                ("m-llama", now_min, 1, 40, 20),
            ],
        )
        db.commit()
    return now_min


def _ready(tmp_data_dir, client):
    client.get("/healthz")
    _seed_two_models(tmp_data_dir / "vllm-warden.db")
    return jwt_login(client)


def _get(client, auth, query=""):
    r = client.get(f"/api/stats/v2/overview?range=1h{query}", headers=auth)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Absent vs empty vs named.
# ---------------------------------------------------------------------------


def test_no_selection_still_covers_the_whole_deployment(tmp_data_dir, client):
    """The pre-selection behaviour, unchanged. Callers that never learn about
    ?models must keep getting the answer they got before."""
    auth = _ready(tmp_data_dir, client)
    body = _get(client, auth)
    assert body["selected_model_ids"] is None
    assert body["selected_gpu_indices"] is None
    assert body["current"]["vram_used_mib"] == 14000
    assert body["current"]["gpu_util_pct"] == 90
    assert body["current"]["power_w"] == 250.0
    assert body["current"]["tps"] == 11.0


def test_an_empty_selection_is_a_400_not_a_silent_everything(
    tmp_data_dir, client
):
    """``?models=`` could mean "all" or "none" and guessing turns a client bug
    into a wrong number. The UI's own rule is that one model always stays
    selected, so an empty string arriving here is a defect worth reporting."""
    auth = _ready(tmp_data_dir, client)
    r = client.get("/api/stats/v2/overview?range=1h&models=", headers=auth)
    assert r.status_code == 400
    assert "empty" in r.json()["detail"]


def test_an_unknown_model_id_is_rejected_not_dropped(tmp_data_dir, client):
    """A silently-ignored id gives a chart that answers a NARROWER question
    than its checkboxes claim -- the worst outcome for a page whose whole job
    is to say which models a number covers."""
    auth = _ready(tmp_data_dir, client)
    r = client.get(
        "/api/stats/v2/overview?range=1h&models=m-qwen,m-ghost", headers=auth
    )
    assert r.status_code == 400
    assert "m-ghost" in r.json()["detail"]


def test_selection_is_order_and_duplicate_insensitive(tmp_data_dir, client):
    """Two spellings of one selection must not be two different answers."""
    auth = _ready(tmp_data_dir, client)
    a = _get(client, auth, "&models=m-qwen,m-llama")
    b = _get(client, auth, "&models=m-llama,m-qwen,m-qwen")
    assert a["current"] == b["current"]
    assert a["selected_model_ids"] == b["selected_model_ids"]


# ---------------------------------------------------------------------------
# One model.
# ---------------------------------------------------------------------------


def test_one_model_reports_only_its_own_tokens(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    body = _get(client, auth, "&models=m-qwen")
    # 600 tokens / 60s.
    assert body["current"]["tps"] == 10.0
    assert body["series"]["tokens"][-1]["prompt"] == 400
    assert body["series"]["tokens"][-1]["completion"] == 200


def test_one_model_reports_only_the_cards_it_occupies(tmp_data_dir, client):
    """qwen holds GPU 0 alone, so its VRAM is GPU 0's VRAM -- not the box's.

    This is the number an operator asking "will another model fit beside it"
    actually needs, and it is the only per-model reading available: gpu_samples
    has no model column and never will.
    """
    auth = _ready(tmp_data_dir, client)
    body = _get(client, auth, "&models=m-qwen")
    assert body["selected_gpu_indices"] == [0]
    assert body["current"]["vram_used_mib"] == 12000
    assert body["current"]["vram_total_mib"] == 16000
    assert body["current"]["gpu_util_pct"] == 90
    assert body["current"]["power_w"] == 200.0


def test_the_quiet_model_does_not_inherit_the_busy_card(tmp_data_dir, client):
    """The inverse selection, which is what catches a filter that was written
    but never bound: llama holds GPU 1, the 10%-utilised card."""
    auth = _ready(tmp_data_dir, client)
    body = _get(client, auth, "&models=m-llama")
    assert body["selected_gpu_indices"] == [1]
    assert body["current"]["gpu_util_pct"] == 10
    assert body["current"]["vram_used_mib"] == 2000
    assert body["current"]["power_w"] == 50.0
    assert body["current"]["tps"] == 1.0


# ---------------------------------------------------------------------------
# The partition property.
# ---------------------------------------------------------------------------


def test_selecting_every_model_sums_to_the_unfiltered_total(
    tmp_data_dir, client
):
    """The property that makes the selector trustworthy.

    If "all selected" disagreed with "no filter", the operator would have no
    way to tell whether a number moved because of their checkboxes or because
    of the fleet. This is also why the token series comes from model_samples
    rather than token_usage_minute: the latter has no model column, so a
    filtered and an unfiltered query would have had to read different tables.
    """
    auth = _ready(tmp_data_dir, client)
    whole = _get(client, auth)
    both = _get(client, auth, "&models=m-qwen,m-llama")
    assert both["current"]["tps"] == whole["current"]["tps"]
    assert both["current"]["vram_used_mib"] == whole["current"]["vram_used_mib"]
    assert both["current"]["power_w"] == whole["current"]["power_w"]


def test_the_parts_add_up_to_the_whole(tmp_data_dir, client):
    auth = _ready(tmp_data_dir, client)
    q = _get(client, auth, "&models=m-qwen")["current"]
    lam = _get(client, auth, "&models=m-llama")["current"]
    whole = _get(client, auth)["current"]
    assert q["tps"] + lam["tps"] == whole["tps"]
    assert q["vram_used_mib"] + lam["vram_used_mib"] == whole["vram_used_mib"]
    assert q["power_w"] + lam["power_w"] == whole["power_w"]
    # Utilisation is a MAX, not a sum -- the busiest of the selected cards.
    assert whole["gpu_util_pct"] == max(q["gpu_util_pct"], lam["gpu_util_pct"])


# ---------------------------------------------------------------------------
# The selector's own input.
# ---------------------------------------------------------------------------


def test_active_models_is_never_narrowed_by_the_selection(tmp_data_dir, client):
    """The list the checkboxes are BUILT from must not shrink when a box is
    unticked -- a deselected model would become impossible to select again,
    which is the same dead end as a control with no capability behind it."""
    auth = _ready(tmp_data_dir, client)
    body = _get(client, auth, "&models=m-qwen")
    assert [m["id"] for m in body["active_models"]] == ["m-llama", "m-qwen"]


def test_active_models_carry_their_cards(tmp_data_dir, client):
    """So the selector can answer "why did VRAM drop when I unticked that
    model" without the operator having to open the model page."""
    auth = _ready(tmp_data_dir, client)
    body = _get(client, auth)
    by_id = {m["id"]: m for m in body["active_models"]}
    assert by_id["m-qwen"]["gpu_indices"] == [0]
    assert by_id["m-llama"]["gpu_indices"] == [1]


def test_a_selection_holding_no_card_reports_no_gpu_data(tmp_data_dir, client):
    """NOT a silent widening to the whole box.

    ``IN ()`` is a syntax error in SQLite, so the empty-union case needs its own
    branch, and the tempting `1=1` there would answer with the box's numbers
    under a selection that asked about none of it -- exactly the substitution
    this endpoint exists to prevent.
    """
    client.get("/healthz")
    db_path = tmp_data_dir / "vllm-warden.db"
    _seed_two_models(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE models SET gpu_indices = '[]' WHERE id = 'm-qwen'")
        db.commit()
    auth = jwt_login(client)
    body = _get(client, auth, "&models=m-qwen")
    assert body["selected_gpu_indices"] == []
    assert body["current"]["vram_used_mib"] == 0
    assert body["current"]["vram_total_mib"] == 0
    assert body["current"]["gpu_util_pct"] == 0
    assert body["current"]["power_w"] is None
    assert body["series"]["vram"] == []
    assert body["series"]["power"] == []
    # Its TOKENS are still its own -- the model dimension is unaffected by the
    # card dimension being empty.
    assert body["current"]["tps"] == 10.0


# ---------------------------------------------------------------------------
# GPU telemetry lost (#255).
# ---------------------------------------------------------------------------


def test_a_failing_gpu_probe_nulls_the_stale_current_numbers(tmp_data_dir, client):
    """With the probe failing, the newest gpu_samples row is from before it
    broke; presenting it as "current" hides that the warden lost its GPUs.
    The numbers go null and the reason rides along."""
    from app.system import gpu as gpu_mod

    auth = _ready(tmp_data_dir, client)
    gpu_mod.reset_gpu_probe_health()
    gpu_mod._record_primary_probe("failing", "nvidia-smi exit 255: Failed to initialize NVML: Unknown Error")
    try:
        cur = _get(client, auth)["current"]
        assert cur["vram_used_mib"] is None
        assert cur["vram_total_mib"] is None
        assert cur["vram_pct"] is None
        assert cur["gpu_util_pct"] is None
        assert cur["power_w"] is None
        assert cur["tps"] == 11.0  # not a GPU reading: unaffected
        assert cur["gpu_probe"] == {
            "state": "failing",
            "error": "nvidia-smi exit 255: Failed to initialize NVML: Unknown Error",
        }
    finally:
        gpu_mod.reset_gpu_probe_health()


def test_a_healthy_probe_reports_its_state_alongside_the_numbers(tmp_data_dir, client):
    from app.system import gpu as gpu_mod

    auth = _ready(tmp_data_dir, client)
    gpu_mod.reset_gpu_probe_health()
    gpu_mod._record_primary_probe("ok", None)
    try:
        cur = _get(client, auth)["current"]
        assert cur["vram_used_mib"] == 14000
        assert cur["gpu_probe"] == {"state": "ok", "error": None}
    finally:
        gpu_mod.reset_gpu_probe_health()
