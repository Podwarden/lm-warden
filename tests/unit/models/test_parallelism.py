"""``app.models.parallelism`` helpers (#286)."""

import pytest

from app.models.parallelism import (
    PARALLEL_FLAGS,
    parallel_flag_values,
    spill_threshold_for,
    strip_parallel_flags,
)


def test_parallel_flags_constant():
    assert PARALLEL_FLAGS == {
        "--tensor-parallel-size",
        "--data-parallel-size",
        "--pipeline-parallel-size",
    }


def test_parallel_flag_values_both_spellings():
    args = ["--tensor-parallel-size", "1", "--data-parallel-size=7", "--max-num-seqs", "32"]
    assert parallel_flag_values(args) == {"--tensor-parallel-size": 1, "--data-parallel-size": 7}


def test_parallel_flag_values_unparsable_is_none():
    assert parallel_flag_values(["--data-parallel-size", "x"]) == {"--data-parallel-size": None}
    assert parallel_flag_values(["--data-parallel-size=0"]) == {"--data-parallel-size": None}
    assert parallel_flag_values(["--data-parallel-size"]) == {"--data-parallel-size": None}


def test_parallel_flag_values_absent_is_empty():
    assert parallel_flag_values(["--max-num-seqs", "32"]) == {}


def test_strip_parallel_flags_removes_pairs_keeps_order():
    args = [
        "--enforce-eager",
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        "32",
        "--data-parallel-size=7",
        "--pipeline-parallel-size",
        "2",
        "--seed",
        "1",
    ]
    assert strip_parallel_flags(args) == ["--enforce-eager", "--max-num-seqs", "32", "--seed", "1"]


def test_spill_threshold_for():
    assert spill_threshold_for([], None) == (64, "auto")
    assert spill_threshold_for(["--max-num-seqs", "32"], None) == (8, "auto")
    assert spill_threshold_for(["--max-num-seqs", "2"], None) == (1, "auto")
    assert spill_threshold_for(["--max-num-seqs=32"], None) == (8, "auto")
    assert spill_threshold_for([], 5) == (5, "setting")
    assert spill_threshold_for(["--max-num-seqs", "32"], 5) == (5, "setting")


# --- #286 review #3: vLLM's FlexibleArgumentParser maps `_` to `-` in long names and
# registers -tp / -dp / -pp as aliases (verified against vllm 0.30 arg_utils.py /
# argparse_utils.py).


def test_short_and_underscore_spellings_are_recognised():
    assert parallel_flag_values(["-tp", "1", "-dp", "7"]) == {
        "--tensor-parallel-size": 1,
        "--data-parallel-size": 7,
    }
    assert parallel_flag_values(["--tensor_parallel_size", "2", "-pp=3"]) == {
        "--tensor-parallel-size": 2,
        "--pipeline-parallel-size": 3,
    }
    assert parallel_flag_values(["--data_parallel_size=4"]) == {"--data-parallel-size": 4}


def test_strip_removes_every_spelling():
    args = [
        "-tp",
        "1",
        "--keep",
        "-dp=7",
        "--data_parallel_size",
        "7",
        "--pipeline_parallel_size=1",
    ]
    assert strip_parallel_flags(args) == ["--keep"]


def test_unrelated_short_flags_are_not_parallel():
    assert parallel_flag_values(["-O3", "-q", "awq", "--tp-plan", "x"]) == {}
    assert strip_parallel_flags(["-O3", "--tp-plan", "x"]) == ["-O3", "--tp-plan", "x"]


# --- #286 re-review N2: argparse allow_abbrev=True accepts unique prefixes.


def test_unambiguous_abbreviations_are_recognised():
    assert parallel_flag_values(["--tensor-parallel", "2"]) == {"--tensor-parallel-size": 2}
    assert parallel_flag_values(["--tensor-parallel-s=2"]) == {"--tensor-parallel-size": 2}
    assert parallel_flag_values(["--data-parallel-s", "7"]) == {"--data-parallel-size": 7}
    assert parallel_flag_values(["--data_parallel_siz=7"]) == {"--data-parallel-size": 7}
    assert parallel_flag_values(["--pipeline-parallel", "3"]) == {"--pipeline-parallel-size": 3}
    assert strip_parallel_flags(["--tensor-parallel", "2", "--keep", "--data-parallel-s=7"]) == [
        "--keep"
    ]


def test_short_prefixes_and_sibling_options_are_not_parallel_flags():
    args = [
        "--tensor",
        "--data",
        "--data-parallel",
        "--data-parallel-rank",
        "--data-parallel-start-rank",
        "--data-parallel-size-local",
        "--data-parallel-backend",
        "--tensor-parallel-sizes",
        "--tokenizer",
    ]
    assert parallel_flag_values([x for a in args for x in (a, "2")]) == {}
    assert strip_parallel_flags(args) == args


def test_spec_refuses_abbreviated_parallel_flag():
    from pydantic import ValidationError

    from app.models.schemas import ModelSpec

    for flag in (["--tensor-parallel", "2"], ["--data-parallel-s=7"]):
        with pytest.raises(ValidationError):
            ModelSpec(
                served_model_name="m1",
                hf_repo="org/repo",
                hf_revision="main",
                gpu_indices=list(range(7)),
                gpu_memory_utilization=0.9,
                extra_args=flag,
            )
