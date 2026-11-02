"""Experiment 04: Deterministic Structured Output Evaluation (Schema Validation + Exact Match).

Treats foundational model evaluation as a deterministic Software Engineering integration
test suite, measuring before and after LoRA adaptation:
  1. Valid JSON syntax rate (`json.loads`).
  2. Strict schema & enum compliance (`{"sev", "svc", "act"}`).
  3. Field-level extraction accuracy and full-record Exact Match.

Two honest caveats are printed with the results:
  - The 12-case suite IS the training set (6 reports + 6 reorderings of the same words), so
    12/12 shows the contract was learned and the pipeline works, not that it generalizes.
    `HELDOUT_INCIDENTS` adds 6 reports that are not in training (same 64-word vocabulary,
    new word order, some words dropped) as a small, separate check.
  - "Before LoRA" is the same toy model with random weights and no examples in the prompt,
    a deliberately weak baseline. Decoding is greedy with no grammar mask, so a valid JSON
    comes from the weights alone.
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

# Reports that are NOT in the training set: same vocabulary, new word order, words dropped.
HELDOUT_INCIDENTS: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "timeout cobros tarjeta pasarela pagos alerta critica",
        {"sev": "P1", "svc": "pagos", "act": "rollback"},
    ),
    (
        "latencia moderada cache pagos error sincronizacion",
        {"sev": "P2", "svc": "pagos", "act": "cache"},
    ),
    (
        "tokens jwt rechazados fallo critico auth produccion",
        {"sev": "P1", "svc": "auth", "act": "rollback"},
    ),
    (
        "sesion expiracion aviso login reintento auth",
        {"sev": "P3", "svc": "auth", "act": "escalar"},
    ),
    (
        "corrupcion indice db caida nodo primario urgente",
        {"sev": "P1", "svc": "db", "act": "rollback"},
    ),
    (
        "bloqueo conexiones replica lectura lenta db disco",
        {"sev": "P2", "svc": "db", "act": "escalar"},
    ),
)


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


def evaluate_model_on_suite(model: Any, heldout: bool = False) -> dict[str, Any]:
    """Runs the deterministic test suite: the 12 training examples, or the 6 held-out reports."""
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    if heldout:
        dataset = tuple(
            exp01.IncidentExample(category_id=-1, incident_text=t, expected_dict=d)
            for t, d in HELDOUT_INCIDENTS
        )
    else:
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
        "split": "heldout" if heldout else "train",
        "num_cases": total,
        "json_parse_count": json_parse_count,
        "schema_valid_count": schema_valid_count,
        "exact_match_count": exact_matches,
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
    heldout_before = evaluate_model_on_suite(base_model, heldout=True)
    heldout_after = evaluate_model_on_suite(adapted_model, heldout=True)

    return {
        "before_lora": before_eval,
        "after_lora": after_eval,
        "heldout_before_lora": heldout_before,
        "heldout_after_lora": heldout_after,
        "lora_train_metrics": lora_train_metrics,
        "adapted_model": adapted_model,
    }


def _fields(raw: str) -> str:
    """`{"sev":"P1","svc":"pagos","act":"rollback"}` -> `P1 pagos rollback` (or the raw text)."""
    try:
        d = json.loads(raw)
        return f"{d['sev']} {d['svc']} {d['act']}"
    except Exception:
        return raw[:24]


def main() -> None:
    res = run_before_after_evaluation()
    b = res["before_lora"]
    a = res["after_lora"]
    hb = res["heldout_before_lora"]
    ha = res["heldout_after_lora"]

    print("=" * 78)
    print(" Experiment 04: Deterministic Structured Output Evaluation (greedy, no grammar mask)")
    print("=" * 78)
    print(" 12 cases = the 12 TRAINING examples (6 reports + 6 reorderings)")
    print(" Before = same toy model, random weights, no examples in the prompt (weak baseline)")
    print("-" * 78)
    print(" Metric (12 train cases)     | Before LoRA (Base) | After LoRA (rank=4)")
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
    rec_b = b["records"][0]
    rec_a = a["records"][0]
    print(f" Train #1 Input   : {rec_a['incident_text']}")
    print(f"   Before LoRA    : {rec_b['decoded_output'][:52]}...")
    print(f"   After LoRA     : {rec_a['decoded_output']} (exact={rec_a['exact_match']})")
    print("-" * 78)
    print(
        f" Held-out (not in training, {ha['num_cases']} reports, same vocabulary, new order)"
    )
    print(
        f" Held-out Schema Valid       | {hb['schema_valid_count']:>14}/{hb['num_cases']}  | "
        f"{ha['schema_valid_count']:>15}/{ha['num_cases']}"
    )
    print(
        f" Held-out Exact Match        | {hb['exact_match_count']:>14}/{hb['num_cases']}  | "
        f"{ha['exact_match_count']:>15}/{ha['num_cases']}"
    )
    for rec in ha["records"]:
        mark = "OK  " if rec["exact_match"] else "MISS"
        print(
            f"   {mark} {rec['incident_text'][:34]:<34} | got {_fields(rec['decoded_output'])}"
            f" | want {_fields(rec['expected_json'])}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
