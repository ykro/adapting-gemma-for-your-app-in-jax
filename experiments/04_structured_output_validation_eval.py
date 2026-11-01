"""Experiment 04: Deterministic Structured Output Evaluation (Schema Validation + Exact Match).

Treats foundational model evaluation as a deterministic Software Engineering integration
test suite over 12 Spanish operational incident reports (6 primary templates + 6 reordered
paraphrases), measuring before and after LoRA adaptation:
  1. Valid JSON syntax rate (`json.loads`).
  2. Strict schema & enum compliance (`{"sev", "svc", "act"}`).
  3. Field-level extraction accuracy and full-record Exact Match.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

ALLOWED_SEVERITIES = frozenset({"P1", "P2", "P3"})
ALLOWED_SERVICES = frozenset({"pagos", "auth", "db"})
ALLOWED_ACTIONS = frozenset({"rollback", "escalar", "cache"})
REQUIRED_KEYS = frozenset({"sev", "svc", "act"})


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name,
        Path(__file__).resolve().parent / filename,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_incident_json(raw_text: str) -> tuple[bool, bool, dict[str, str] | None]:
    """Validates raw decoded text against JSON syntax and strict domain schema enums.

    Returns:
        `(json_parse_ok, schema_valid_ok, parsed_dict_or_none)`
    """
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return False, False, None

    if not isinstance(parsed, dict):
        return True, False, None

    if set(parsed.keys()) != REQUIRED_KEYS:
        return True, False, None

    schema_ok = (
        parsed.get("sev") in ALLOWED_SEVERITIES
        and parsed.get("svc") in ALLOWED_SERVICES
        and parsed.get("act") in ALLOWED_ACTIONS
    )
    return True, schema_ok, (parsed if schema_ok else None)


@nnx.jit
def _decode_step_jit(mdl: Any, buf: jax.Array, pos: jax.Array) -> jax.Array:
    logits = mdl(buf)
    next_token = jnp.argmax(logits[:, pos - 1, :], axis=-1).astype(jnp.int32)
    return buf.at[:, pos].set(next_token)


def greedy_generate_batch(
    model: Any,
    prompt_batch: jax.Array,
    max_new_tokens: int = 14,
    pad_id: int = 0,
) -> jax.Array:
    """Autoregressively decodes `max_new_tokens` in a static buffer `[B, P + max_new_tokens]`."""
    bsz, prompt_len = prompt_batch.shape
    total_len = prompt_len + max_new_tokens
    pad_tail = jnp.full((bsz, max_new_tokens), pad_id, dtype=jnp.int32)
    buffer = jnp.concatenate([prompt_batch, pad_tail], axis=1)

    for offset in range(max_new_tokens):
        pos_idx = jnp.array(prompt_len + offset, dtype=jnp.int32)
        buffer = _decode_step_jit(model, buffer, pos_idx)

    return buffer[:, prompt_len : total_len]


def evaluate_model_on_suite(model: Any) -> dict[str, Any]:
    """Runs the deterministic software test suite across all 12 Spanish incident examples."""
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    dataset = exp01.DOMAIN_INCIDENT_DATASET
    prompt_rows = [exp01.encode_prompt_tokens(ex.incident_text, prompt_len=16) for ex in dataset]
    prompt_batch = jnp.array(prompt_rows, dtype=jnp.int32)

    generated_batch = greedy_generate_batch(
        model,
        prompt_batch,
        max_new_tokens=14,
        pad_id=exp01.PAD_ID,
    )
    gen_list = generated_batch.tolist()

    total = len(dataset)
    json_parse_count = 0
    schema_valid_count = 0
    field_matches = 0
    exact_matches = 0
    records: list[dict[str, Any]] = []

    for ex, gen_ids in zip(dataset, gen_list):
        decoded = exp01.decode_tokens(gen_ids)
        parse_ok, schema_ok, parsed_dict = validate_incident_json(decoded)
        if parse_ok:
            json_parse_count += 1
        if schema_ok:
            schema_valid_count += 1

        matched_fields = 0
        if parsed_dict is not None:
            for k in ("sev", "svc", "act"):
                if parsed_dict.get(k) == ex.expected_dict[k]:
                    matched_fields += 1
        field_matches += matched_fields
        is_exact = matched_fields == 3 and schema_ok
        if is_exact:
            exact_matches += 1

        records.append(
            {
                "incident_text": ex.incident_text,
                "expected_json": ex.expected_json,
                "decoded_output": decoded,
                "schema_valid": schema_ok,
                "exact_match": is_exact,
            }
        )

    return {
        "num_cases": total,
        "json_parse_rate_pct": (json_parse_count / total) * 100.0,
        "schema_valid_rate_pct": (schema_valid_count / total) * 100.0,
        "field_accuracy_pct": (field_matches / (total * 3)) * 100.0,
        "exact_match_rate_pct": (exact_matches / total) * 100.0,
        "records": records,
    }


def run_before_after_evaluation() -> dict[str, Any]:
    """Evaluates the base Gemma model before LoRA and after surgical LoRA adaptation."""
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    exp03 = _load_module("exp03_lora", "03_lora_adapters_in_flax_nnx.py")

    base_model = exp01.GemmaForCausalLM(exp01.GemmaAppConfig(), rngs=nnx.Rngs(7))
    before_eval = evaluate_model_on_suite(base_model)

    adapted_model, lora_train_metrics = exp03.train_domain_lora_adapter(lora_rank=4, steps=45)
    after_eval = evaluate_model_on_suite(adapted_model)

    return {
        "before_lora": before_eval,
        "after_lora": after_eval,
        "lora_train_metrics": lora_train_metrics,
        "adapted_model": adapted_model,
    }


def main() -> None:
    res = run_before_after_evaluation()
    b = res["before_lora"]
    a = res["after_lora"]

    print("=" * 78)
    print(" Experiment 04: Deterministic Structured Output Evaluation Suite (12 Cases)")
    print("=" * 78)
    print(" Metric                      | Before LoRA (Base) | After LoRA (rank=4)")
    print("-" * 78)
    print(
        f" Valid JSON Syntax Rate      | {b['json_parse_rate_pct']:16.1f}%  | "
        f"{a['json_parse_rate_pct']:17.1f}%"
    )
    print(
        f" Strict Schema & Enum Rate   | {b['schema_valid_rate_pct']:16.1f}%  | "
        f"{a['schema_valid_rate_pct']:17.1f}%"
    )
    print(
        f" Field Extraction Accuracy   | {b['field_accuracy_pct']:16.1f}%  | "
        f"{a['field_accuracy_pct']:17.1f}%"
    )
    print(
        f" Full-Record Exact Match     | {b['exact_match_rate_pct']:16.1f}%  | "
        f"{a['exact_match_rate_pct']:17.1f}%"
    )
    print("-" * 78)
    for idx in range(3):
        rec_b = b["records"][idx]
        rec_a = a["records"][idx]
        print(f" Sample #{idx + 1} Input    : {rec_a['incident_text']}")
        print(f"   Before LoRA Out  : {rec_b['decoded_output']!r}")
        print(
            f"   After LoRA Out   : {rec_a['decoded_output']} "
            f"(schema_valid={rec_a['schema_valid']}, exact_match={rec_a['exact_match']})"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
