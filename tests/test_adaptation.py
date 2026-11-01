"""Deterministic pytest suite for `adapting-gemma-for-your-app-in-jax`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _load_exp(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_01_when_prompting_hits_a_wall() -> None:
    exp01 = _load_exp("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    res = exp01.benchmark_prompting_wall(num_iters=10)
    assert res["token_overhead_ratio"] == 13.0
    assert res["prefill_speedup"] > 1.2
    assert res["kv_cache_reduction_pct"] > 92.0
    assert res["unadapted_valid_json"] is False


def test_02_preparing_domain_dataset_masking() -> None:
    exp02 = _load_exp("exp02_masking", "02_preparing_domain_dataset_masking.py")
    res = exp02.compare_prompt_vs_completion_masking(steps=12)
    assert res["batch_shape"] == (12, 32)
    assert res["wasted_prompt_token_share_pct"] > 35.0
    assert res["masked_completion_loss"] <= res["unmasked_completion_loss"]
    assert res["masked_completion_acc_pct"] >= 99.0


def test_03_lora_adapters_in_flax_nnx() -> None:
    exp03 = _load_exp("exp03_lora", "03_lora_adapters_in_flax_nnx.py")
    _, res = exp03.train_domain_lora_adapter(lora_rank=4, steps=45)
    assert res["reduction_pct"] > 97.0
    assert res["max_base_weight_diff"] == 0.0
    assert res["base_weights_frozen_bitwise"] is True
    assert res["lora_weights_updated"] is True
    assert res["loss_after"] < 0.25


def test_04_structured_output_validation_eval() -> None:
    exp04 = _load_exp("exp04_eval", "04_structured_output_validation_eval.py")
    res = exp04.run_before_after_evaluation()
    assert res["before_lora"]["exact_match_rate_pct"] == 0.0
    assert res["after_lora"]["schema_valid_rate_pct"] == 100.0
    assert res["after_lora"]["exact_match_rate_pct"] == 100.0


def test_05_merge_weights_and_serve_endpoint() -> None:
    exp05 = _load_exp("exp05_merge", "05_merge_weights_and_serve_endpoint.py")
    res = exp05.verify_weight_merge_and_serving(num_iters=15)
    assert res["lora_params_after_merge"] == 0
    assert res["max_logits_abs_diff"] < 1e-5
    assert res["merged_exact_match_rate_pct"] == 100.0
    assert res["http_status"] == 200
    assert res["http_response"]["schema_valid"] is True
    assert res["http_response"]["triage"] == {"sev": "P1", "svc": "pagos", "act": "rollback"}
