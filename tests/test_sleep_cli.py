"""Tests for the shared CLI plumbing in scripts/sleep/common.py."""

import argparse
import json

from rlm.sleep.config import SleepConfig
from scripts.sleep.common import add_common_args, apply_judge_overrides, dump_judge_usage
from tests.mock_lm import MockLM


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    return parser.parse_args(argv)


def test_n_samples_overrides_judge_config():
    config = SleepConfig()
    apply_judge_overrides(config, parse(["--n-samples", "3"]))
    assert config.judge.n_samples == 3


def test_n_samples_default_keeps_config_default():
    config = SleepConfig()
    default = config.judge.n_samples
    apply_judge_overrides(config, parse([]))
    assert config.judge.n_samples == default


def test_dump_judge_usage_writes_totals(tmp_path):
    lm = MockLM()
    lm.completion("one")
    lm.completion("two")
    out = tmp_path / "judge_usage.json"
    data = dump_judge_usage(lm, out)
    assert data == json.loads(out.read_text())
    assert data["mock-model"]["total_calls"] == 2
    assert data["mock-model"]["total_input_tokens"] == 20
