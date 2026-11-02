"""The Outside Judge (`verify_adaptation.py`) for `adapting-gemma-for-your-app-in-jax`.

Executes all 5 domain adaptation experiments, enforces deterministic assertions (13x more
prompt tokens and a faster prefill, completion masking scored on the same JSON tokens,
~98% of the parameters frozen with 0.0 base drift, 12/12 JSON schema & exact match on the
training cases after LoRA, <1e-5 logits parity after folding `W_merged = W_base + A @ B`,
the same compiled `dot` count as the base model, and a local HTTP 200), and writes
`benchmarks/latest_results.json`, `benchmarks/terminal_output.txt` and the results block of
`README.md`, so the README numbers are always the ones in the JSON.

The model is a small Gemma-style decoder with random base weights, not a Gemma checkpoint.
The checks are deterministic. The timings are medians per call after a warmup and vary run
to run; see "Cómo leer estas cifras" in the README.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import flax
import jax
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
README_START = "<!-- judge:start -->"
README_END = "<!-- judge:end -->"


def _load_exp(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cpu_model() -> str:
    """Human-readable CPU model (platform.processor() only says 'arm' on macOS)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _load_average() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def _strip_records(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "records"}


def run_verification() -> dict[str, Any]:
    load_before = _load_average()
    exp01 = _load_exp("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    exp02 = _load_exp("exp02_masking", "02_preparing_domain_dataset_masking.py")
    exp03 = _load_exp("exp03_lora", "03_lora_adapters_in_flax_nnx.py")
    exp04 = _load_exp("exp04_eval", "04_structured_output_validation_eval.py")
    exp05 = _load_exp("exp05_merge", "05_merge_weights_and_serve_endpoint.py")

    res01 = exp01.benchmark_prompting_wall()
    assert res01["token_overhead_ratio"] == 13.0
    assert res01["prefill_speedup"] > 1.5
    assert res01["kv_cache_reduction_pct"] > 90.0
    assert res01["unadapted_valid_json"] is False

    res02 = exp02.compare_prompt_vs_completion_masking(steps=12)
    assert res02["batch_shape"] == (12, 32)
    assert res02["jit_trace_count"] == 2, "one trace per model, no retrace"
    assert res02["wasted_prompt_token_share_pct"] > 35.0
    assert res02["masked_completion_loss"] <= res02["unmasked_completion_loss"]
    assert res02["masked_completion_acc_pct"] >= 99.0

    _, res03 = exp03.train_domain_lora_adapter(lora_rank=4, steps=45)
    assert res03["reduction_pct"] > 97.0
    assert res03["max_base_weight_diff"] == 0.0
    assert res03["base_weights_frozen_bitwise"] is True
    assert res03["lora_weights_updated"] is True
    assert res03["loss_after"] < 0.25

    res04_raw = exp04.run_before_after_evaluation()
    res04_before = res04_raw["before_lora"]
    res04_after = res04_raw["after_lora"]
    res04_held = res04_raw["heldout_after_lora"]
    assert res04_before["exact_match_rate_pct"] == 0.0
    assert res04_after["schema_valid_rate_pct"] == 100.0
    assert res04_after["exact_match_rate_pct"] == 100.0
    assert res04_held["num_cases"] == len(exp04.HELDOUT_INCIDENTS)

    res05 = exp05.verify_weight_merge_and_serving()
    assert res05["lora_params_after_merge"] == 0
    assert res05["merged_total_params"] == res05["base_total_params"]
    assert res05["max_logits_abs_diff"] < 1e-5
    assert res05["hlo_dot_ops"]["merged"] == res05["hlo_dot_ops"]["base"]
    assert res05["hlo_dot_ops"]["two_branch_lora"] > res05["hlo_dot_ops"]["merged"]
    assert res05["merged_exact_match_rate_pct"] == 100.0
    assert res05["http_status"] == 200
    assert res05["http_response"]["schema_valid"] is True

    return {
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "timing": "median ms per call after a warmup (A/B interleaved); varies run to run",
            "model": "toy Gemma-style decoder (4 blocks, d_model 256), random base weights; "
            "not a Gemma checkpoint",
        },
        "experiments": {
            "01_when_prompting_hits_a_wall": res01,
            "02_preparing_domain_dataset_masking": res02,
            "03_lora_adapters_in_flax_nnx": res03,
            "04_structured_output_validation_eval": {
                "train_set_note": "the 12 cases are the 12 training examples",
                "before_lora": _strip_records(res04_before),
                "after_lora": _strip_records(res04_after),
                "heldout_before_lora": _strip_records(res04_raw["heldout_before_lora"]),
                "heldout_after_lora": _strip_records(res04_held),
                "sample_records": res04_after["records"][:3],
                "heldout_records": res04_held["records"],
            },
            "05_merge_weights_and_serve_endpoint": res05,
        },
        "all_checks_passed": True,
    }


def fmt_ms(v: float) -> str:
    if v >= 100:
        return f"{v:.1f} ms"
    if v >= 1:
        return f"{v:.2f} ms"
    return f"{v:.3f} ms"


def table_rows(summary: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    ex = summary["experiments"]
    e1 = ex["01_when_prompting_hits_a_wall"]
    e2 = ex["02_preparing_domain_dataset_masking"]
    e3 = ex["03_lora_adapters_in_flax_nnx"]
    e4 = ex["04_structured_output_validation_eval"]
    e5 = ex["05_merge_weights_and_serve_endpoint"]
    b4, a4, h4 = e4["before_lora"], e4["after_lora"], e4["heldout_after_lora"]
    d5 = e5["hlo_dot_ops"]
    return [
        (
            "01. Prefill 208t vs 16t (jit)",
            fmt_ms(e1["fewshot_prefill_ms"]),
            fmt_ms(e1["compact_prefill_ms"]),
            f"{e1['prefill_speedup']:.2f}x",
            f"same weights, no LoRA; KV -{e1['kv_cache_reduction_pct']:.1f}% (formula)",
        ),
        (
            "02. JSON loss: no mask/mask",
            f"{e2['unmasked_completion_loss']:.4f} CE",
            f"{e2['masked_completion_loss']:.4f} CE",
            f"{e2['loss_improvement_ratio']:.2f}x",
            "both 100% JSON tokens; 1 seed, train",
        ),
        (
            "03. LoRA rank=4 trainable",
            f"{e3['total_params']:,}",
            f"{e3['lora_params']:,}",
            f"{e3['trainable_ratio_pct']:.2f}%",
            f"base diff = {e3['max_base_weight_diff']:.1f} (bitwise)",
        ),
        (
            "04. JSON suite, exact match",
            f"{b4['exact_match_count']}/{b4['num_cases']} train",
            f"{a4['exact_match_count']}/{a4['num_cases']} train",
            f"{a4['schema_valid_count']}/{a4['num_cases']} schema",
            f"held-out {h4['exact_match_count']}/{h4['num_cases']} exact, "
            f"{h4['schema_valid_count']}/{h4['num_cases']} schema",
        ),
        (
            "05. Merged vs two-branch LoRA",
            fmt_ms(e5["unmerged_forward_ms"]),
            fmt_ms(e5["merged_forward_ms"]),
            f"{e5['merge_inference_speedup']:.2f}x",
            f"dots {d5['merged']} = base {d5['base']}; diff {e5['max_logits_abs_diff']:.2e}",
        ),
    ]


def format_report(summary: dict[str, Any]) -> str:
    rt = summary["runtime"]
    load = rt["load_average_before"], rt["load_average_after"]
    e5 = summary["experiments"]["05_merge_weights_and_serve_endpoint"]
    width = 106
    lines = [
        "=" * width,
        "THE OUTSIDE JUDGE — Adapting Gemma for Your App in JAX Verification Report",
        "=" * width,
        (
            f"Runtime : Python {rt['python']} | JAX {rt['jax']} | Flax {rt['flax']} | "
            f"Backend: {rt['backend']}"
        ),
        (
            f"CPU     : {rt['cpu_model']} ({rt['cpu_count']} cores) | 1-min load average "
            f"{load[0][0] if load[0] else 'n/a'} before, {load[1][0] if load[1] else 'n/a'} after"
        ),
        "Model   : toy Gemma-style decoder, 3.67M params, random base weights (not a checkpoint)",
        "-" * width,
        (
            f"{'Experiment':<29} | {'Baseline':<15} | {'Adapted/JAX':<12} | "
            f"{'Result':<12} | Fair baseline / note"
        ),
        "-" * width,
    ]
    for name, base, jx, speed, fair in table_rows(summary):
        lines.append(f"{name:<29} | {base:<15} | {jx:<12} | {speed:<12} | {fair}")
    lines += [
        "-" * width,
        f"Local HTTP POST /v1/triage (127.0.0.1, nothing deployed): HTTP {e5['http_status']}",
        "Times: median per call after a warmup; they vary run to run and with machine load.",
        "Saved JSON report to: benchmarks/latest_results.json",
        "ALL DETERMINISTIC ADAPTATION CHECKS PASSED.",
    ]
    return "\n".join(lines)


def readme_block(summary: dict[str, Any], report_text: str) -> str:
    rt = summary["runtime"]
    load0 = rt["load_average_before"][0] if rt["load_average_before"] else "n/a"
    out = [
        README_START,
        "",
        (
            f"Corrida del juez en {rt['cpu_model']} ({rt['cpu_count']} núcleos), "
            f"Python {rt['python']}, JAX {rt['jax']}, Flax {rt['flax']}, backend "
            f"`{rt['backend']}`. Carga del sistema (load average de 1 min): {load0} al empezar. "
            "Tiempos = mediana por llamada después de una llamada de calentamiento, "
            "con las dos variantes intercaladas; speedup = mediana de los cocientes por par. "
            "Modelo de juguete estilo Gemma con pesos base aleatorios."
        ),
        "",
        "| Experimento | Baseline | Adaptado / JAX | Resultado | Baseline justo / nota |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for name, base, jx, speed, fair in table_rows(summary):
        out.append(f"| **{name}** | `{base}` | `{jx}` | **{speed}** | {fair} |")
    out += [
        "",
        "Salida íntegra de terminal generada por `verify_adaptation.py`:",
        "",
        "```text",
        report_text,
        "```",
        "",
        README_END,
    ]
    return "\n".join(out)


def update_readme(summary: dict[str, Any], report_text: str) -> None:
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(head + readme_block(summary, report_text) + tail, encoding="utf-8")


def main() -> None:
    summary = run_verification()
    report_text = format_report(summary)
    print(report_text)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    (BENCHMARKS_DIR / "latest_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (BENCHMARKS_DIR / "terminal_output.txt").write_text(report_text + "\n", encoding="utf-8")
    update_readme(summary, report_text)


if __name__ == "__main__":
    main()
