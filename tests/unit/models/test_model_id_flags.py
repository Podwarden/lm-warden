"""``app.models.model_id_flags`` detector (planner iter 32, model-id override hole)."""

# --- the per-backend flag table ---------------------------------------------


def test_model_id_flags_constant():
    from app.models.model_id_flags import MODEL_ID_FLAGS

    assert MODEL_ID_FLAGS == {
        "vllm": {"--served-model-name"},
        "llamacpp": {"--alias"},
        "mlx": {"--model"},
    }
    assert all(isinstance(v, frozenset) for v in MODEL_ID_FLAGS.values())


# --- vllm: --served-model-name (underscore, = form, abbreviations) -----------


def test_vllm_canonical_space_and_equals():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("vllm", ["--served-model-name", "x"]) == "--served-model-name"
    assert model_id_flag_in("vllm", ["--served-model-name=x"]) == "--served-model-name=x"
    assert model_id_flag_in("vllm", ["--enforce-eager", "--served-model-name", "x"]) == (
        "--served-model-name"
    )


def test_vllm_underscore_spelling_is_recognised():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("vllm", ["--served_model_name", "x"]) == "--served_model_name"
    assert model_id_flag_in("vllm", ["--served_model_name=x"]) == "--served_model_name=x"


def test_vllm_abbreviation_down_to_floor():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("vllm", ["--served-model", "x"]) == "--served-model"
    assert model_id_flag_in("vllm", ["--served", "x"]) == "--served"
    assert model_id_flag_in("vllm", ["--served=x"]) == "--served=x"
    # vLLM v0.20.0 has no other --ser* option (only --seed), so argparse
    # resolves --serve / --ser to --served-model-name too: --ser is the floor.
    assert model_id_flag_in("vllm", ["--serve", "x"]) == "--serve"
    assert model_id_flag_in("vllm", ["--ser", "x"]) == "--ser"
    assert model_id_flag_in("vllm", ["--served_model", "x"]) == "--served_model"


def test_vllm_below_floor_and_non_prefixes_are_not_matched():
    from app.models.model_id_flags import model_id_flag_in

    # --se is ambiguous with --seed (argparse errors), so it is not the flag.
    assert model_id_flag_in("vllm", ["--se", "x"]) is None
    assert model_id_flag_in("vllm", ["--seed", "1"]) is None
    assert model_id_flag_in("vllm", ["--served-model-names-extra", "x"]) is None
    assert model_id_flag_in("vllm", ["--se", "--served-model-names-extra", "x"]) is None


def test_vllm_dotted_key_sets_the_served_name():
    from app.models.model_id_flags import model_id_flag_in

    # FlexibleArgumentParser turns --served-model-name.k=v into
    # --served-model-name '{"k": "v"}', which sets the served name.
    assert model_id_flag_in("vllm", ["--served-model-name.k=v"]) == "--served-model-name.k=v"
    assert model_id_flag_in("vllm", ["--served-model-name.k", "v"]) == "--served-model-name.k"
    assert model_id_flag_in("vllm", ["--served_model_name.k+", "a,b"]) == "--served_model_name.k+"
    assert model_id_flag_in("vllm", ["--ser.k=v"]) == "--ser.k=v"
    # A dot only in the value is not a dotted key.
    assert model_id_flag_in("vllm", ["--seed=1.5"]) is None
    assert model_id_flag_in("vllm", ["--se.k=v"]) is None
    # Dotted keys are a vLLM rewrite; plain argparse (MLX) has no such form.
    assert model_id_flag_in("mlx", ["--model.k=v"]) is None


# --- llamacpp: --alias / -a (no abbreviation, no underscore normalisation) ---


def test_llamacpp_canonical_short_and_equals():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("llamacpp", ["--alias", "x"]) == "--alias"
    assert model_id_flag_in("llamacpp", ["--alias=x"]) == "--alias=x"
    assert model_id_flag_in("llamacpp", ["-a", "x"]) == "-a"


def test_llamacpp_no_underscore_normalisation_and_no_abbreviation():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("llamacpp", ["--al_ias", "x"]) is None
    assert model_id_flag_in("llamacpp", ["--alia", "x"]) is None
    assert model_id_flag_in("llamacpp", ["--al", "x"]) is None


# --- mlx: --model (argparse abbreviations down to --mo) ----------------------


def test_mlx_canonical_and_abbreviation_down_to_floor():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("mlx", ["--model", "x"]) == "--model"
    assert model_id_flag_in("mlx", ["--model=x"]) == "--model=x"
    assert model_id_flag_in("mlx", ["--mod", "x"]) == "--mod"
    assert model_id_flag_in("mlx", ["--mo", "x"]) == "--mo"


def test_mlx_below_floor_and_non_prefixes_are_not_matched():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("mlx", ["--m", "x"]) is None
    assert model_id_flag_in("mlx", ["--draft-model", "x"]) is None
    assert model_id_flag_in("mlx", ["--model-extra", "x"]) is None
    # Plain argparse does not rewrite underscores: no underscore spelling.
    assert model_id_flag_in("mlx", ["--mo_del", "x"]) is None


# --- none arms ---------------------------------------------------------------


def test_empty_extra_args_is_none():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("vllm", []) is None
    assert model_id_flag_in("llamacpp", []) is None
    assert model_id_flag_in("mlx", []) is None


def test_args_without_the_flag_are_none():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("vllm", ["--enforce-eager", "--max-num-seqs", "32"]) is None
    assert model_id_flag_in("llamacpp", ["-ngl", "99", "--ctx-size", "4096"]) is None
    assert model_id_flag_in("mlx", ["--host", "127.0.0.1", "--port", "8080"]) is None


def test_flags_of_another_backend_are_none():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("vllm", ["--alias", "x"]) is None
    assert model_id_flag_in("vllm", ["--model", "x"]) is None
    assert model_id_flag_in("llamacpp", ["--served-model-name", "x"]) is None
    assert model_id_flag_in("mlx", ["--served-model-name", "x"]) is None
    assert model_id_flag_in("mlx", ["--alias", "x"]) is None


def test_first_offending_token_wins_when_two_are_present():
    from app.models.model_id_flags import model_id_flag_in

    assert model_id_flag_in("vllm", ["--served-model-name", "a", "--served", "b"]) == (
        "--served-model-name"
    )
    assert model_id_flag_in("mlx", ["--draft-model", "a", "--model", "b"]) == "--model"
    assert model_id_flag_in("mlx", ["--model", "a", "--mo", "b"]) == "--model"


def test_value_that_looks_like_a_flag_does_not_match():
    from app.models.model_id_flags import model_id_flag_in

    # "--served-model-name x" -- the value "x" is not a flag.
    assert model_id_flag_in("vllm", ["--enforce-eager", "x"]) is None
    # A later positional that merely resembles the flag is not scanned.
    assert model_id_flag_in("vllm", ["--served-model-name"]) == "--served-model-name"


# --- unknown backend ---------------------------------------------------------


def test_unknown_backend_raises_value_error_naming_the_accepted_names():
    import pytest

    from app.models.model_id_flags import model_id_flag_in

    with pytest.raises(ValueError) as exc:
        model_id_flag_in("openai", ["--served-model-name", "x"])
    msg = str(exc.value)
    assert "vllm" in msg
    assert "llamacpp" in msg
    assert "mlx" in msg


# --- refusal message ---------------------------------------------------------


def test_model_id_refusal_names_token_and_column():
    from app.models.model_id_flags import model_id_refusal

    msg = model_id_refusal("vllm", "--served-model-name")
    assert "--served-model-name" in msg
    assert "served_model_name" in msg

    mlx_msg = model_id_refusal("mlx", "--model")
    assert "--model" in mlx_msg
    assert "served_model_name" in mlx_msg
    assert "snapshot" in mlx_msg.lower() or "path" in mlx_msg.lower()


def test_model_id_refusal_for_llamacpp_alias():
    from app.models.model_id_flags import model_id_refusal

    msg = model_id_refusal("llamacpp", "--alias")
    assert "--alias" in msg
    assert "served_model_name" in msg
