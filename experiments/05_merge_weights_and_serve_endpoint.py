"""Experiment 05: Mathematical LoRA Weight Merging (`W_merged = W_base + A @ B`) & Serving.

Proves that after training a `nnx.LoRA` adapter we can fold the low-rank matrices directly
into the frozen base weights (`W_merged = W_base + lora_a @ lora_b`) and unwrap `nnx.LoRA`
back into standard `nnx.Linear` layers:
  1. Numerical equivalence up to float32 rounding (`max_logits_abs_diff < 1e-5`, the sum is
     reassociated) and the same greedy JSON on the 12 training cases.
  2. Zero remaining `nnx.LoRAParam` variables (`lora_params_after_merge == 0`) and the same
     parameter count as the base model.
  3. The same compiled program shape as the base model: the optimized HLO of the merged model
     has as many `dot` ops as the base model (checked structurally, not with a stopwatch),
     while the two-branch LoRA model has more. The merged vs two-branch forward is timed as
     the median per call after a warmup call (compilation timed apart).
  4. A local HTTP `POST /v1/triage` server (127.0.0.1, inside this process) that answers a
     real request. Nothing is deployed anywhere.
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


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


def merge_lora_weights_into_base(lora_model: Any) -> Any:
    """Clones `lora_model` and folds `lora_a @ lora_b` into each wrapped `nnx.Linear` kernel."""
    merged_model = nnx.clone(lora_model)

    for block in merged_model.blocks:
        attn = block.attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            wrapper = getattr(attn, proj_name)
            if isinstance(wrapper, nnx.LoRA):
                base_linear: nnx.Linear = wrapper.base_module
                delta_w = wrapper.lora_a[...] @ wrapper.lora_b[...]
                base_linear.kernel[...] = base_linear.kernel[...] + delta_w
                setattr(attn, proj_name, base_linear)

        mlp = block.mlp
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            wrapper = getattr(mlp, proj_name)
            if isinstance(wrapper, nnx.LoRA):
                base_linear = wrapper.base_module
                delta_w = wrapper.lora_a[...] @ wrapper.lora_b[...]
                base_linear.kernel[...] = base_linear.kernel[...] + delta_w
                setattr(mlp, proj_name, base_linear)

    return merged_model


def _paired_median_ms(fn_a: Any, fn_b: Any, n_iters: int) -> tuple[float, float, float]:
    """Times A and B interleaved (A,B then B,A) so load spikes on a shared machine hit both.

    Returns (median ms of A, median ms of B, median of the per-pair ratios A/B).
    """
    a_ms: list[float] = []
    b_ms: list[float] = []
    for i in range(n_iters):
        pair = {}
        for name, fn in (("a", fn_a), ("b", fn_b)) if i % 2 == 0 else (("b", fn_b), ("a", fn_a)):
            t0 = time.perf_counter()
            fn()
            pair[name] = (time.perf_counter() - t0) * 1000.0
        a_ms.append(pair["a"])
        b_ms.append(pair["b"])
    ratios = [a / b for a, b in zip(a_ms, b_ms)]
    return statistics.median(a_ms), statistics.median(b_ms), statistics.median(ratios)


def count_hlo_dots(model: Any, tokens: jax.Array) -> int:
    """Number of `dot` ops in the optimized HLO that XLA compiles for `model(tokens)`."""
    graphdef, state = nnx.split(model)
    fwd = jax.jit(lambda st, t: nnx.merge(graphdef, st)(t))
    hlo = fwd.lower(state, tokens).compile().as_text()
    return len(re.findall(r" dot\(", hlo))


def serve_and_verify_triage_endpoint(
    merged_model: Any,
    incident_text: str = "alerta critica pasarela pagos timeout cobros tarjeta produccion urgente",
) -> dict[str, Any]:
    """Starts a local HTTP server (127.0.0.1, `POST /v1/triage`) and sends it one request."""
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    exp04 = _load_module("exp04_eval", "04_structured_output_validation_eval.py")

    # Warmup single-request compilation before serving HTTP traffic
    warm_ids = jnp.array(
        [exp01.encode_prompt_tokens(incident_text, prompt_len=16)],
        dtype=jnp.int32,
    )
    _ = exp04.greedy_generate_batch(merged_model, warm_ids, max_new_tokens=14).block_until_ready()

    class TriageHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/triage":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw_body)
            text = str(payload.get("incident", ""))

            t0 = time.perf_counter()
            prompt_ids = jnp.array(
                [exp01.encode_prompt_tokens(text, prompt_len=16)],
                dtype=jnp.int32,
            )
            gen_ids = exp04.greedy_generate_batch(
                merged_model,
                prompt_ids,
                max_new_tokens=14,
                pad_id=exp01.PAD_ID,
            )
            gen_ids.block_until_ready()
            infer_ms = (time.perf_counter() - t0) * 1000.0

            decoded = exp01.decode_tokens(gen_ids[0].tolist())
            _, schema_valid, parsed_dict = exp04.validate_incident_json(decoded)

            resp_bytes = json.dumps(
                {
                    "status": "ok",
                    "triage": parsed_dict,
                    "raw_completion": decoded,
                    "schema_valid": schema_valid,
                    "inference_ms": round(infer_ms, 4),
                    "serving_mode": "merged_single_branch_xla",
                },
                ensure_ascii=False,
            ).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), TriageHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        req_body = json.dumps({"incident": incident_text}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/triage",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            http_status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    return {
        "http_status": int(http_status),
        "response": body,
    }


def verify_weight_merge_and_serving(num_iters: int = 35) -> dict[str, Any]:
    """Benchmarks unmerged LoRA vs merged `W_base + A @ B` model and runs the HTTP check."""
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    exp02 = _load_module("exp02_masking", "02_preparing_domain_dataset_masking.py")
    exp03 = _load_module("exp03_lora", "03_lora_adapters_in_flax_nnx.py")
    exp04 = _load_module("exp04_eval", "04_structured_output_validation_eval.py")

    lora_model, _ = exp03.train_domain_lora_adapter(lora_rank=4, steps=45)
    # Same seed as the base inside exp03, before LoRA was injected: the original model.
    base_model = exp01.GemmaForCausalLM(exp01.GemmaAppConfig(), rngs=nnx.Rngs(7))
    merged_model = merge_lora_weights_into_base(lora_model)

    counts_before = exp03.count_lora_vs_base_parameters(lora_model)
    counts_after = exp03.count_lora_vs_base_parameters(merged_model)

    batch = exp02.build_masked_domain_batch()
    input_ids = batch["input_ids"]

    @nnx.jit
    def forward_jit(mdl: Any, tokens: jax.Array) -> jax.Array:
        return mdl(tokens)

    t0 = time.perf_counter()
    logits_unmerged = forward_jit(lora_model, input_ids).block_until_ready()
    unmerged_first_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    logits_merged = forward_jit(merged_model, input_ids).block_until_ready()
    merged_first_ms = (time.perf_counter() - t0) * 1000.0
    max_logits_diff = float(jnp.max(jnp.abs(logits_unmerged - logits_merged)))

    unmerged_ms, merged_ms, merge_ratio = _paired_median_ms(
        lambda: forward_jit(lora_model, input_ids).block_until_ready(),
        lambda: forward_jit(merged_model, input_ids).block_until_ready(),
        num_iters,
    )

    hlo_dots = {
        "base": count_hlo_dots(base_model, input_ids),
        "merged": count_hlo_dots(merged_model, input_ids),
        "two_branch_lora": count_hlo_dots(lora_model, input_ids),
    }
    base_params = exp03.count_lora_vs_base_parameters(base_model)["total_params"]

    merged_eval = exp04.evaluate_model_on_suite(merged_model)
    http_check = serve_and_verify_triage_endpoint(merged_model)

    return {
        "lora_params_before_merge": int(counts_before["lora_params"]),
        "lora_params_after_merge": int(counts_after["lora_params"]),
        "merged_total_params": int(counts_after["total_params"]),
        "base_total_params": int(base_params),
        "max_logits_abs_diff": max_logits_diff,
        "hlo_dot_ops": hlo_dots,
        "timing": f"interleaved pairs, median of {num_iters} calls after one warmup call",
        "num_iters": num_iters,
        "unmerged_first_call_ms": float(unmerged_first_ms),
        "merged_first_call_ms": float(merged_first_ms),
        "unmerged_forward_ms": float(unmerged_ms),
        "merged_forward_ms": float(merged_ms),
        "merge_inference_speedup": float(merge_ratio),
        "merged_exact_match_rate_pct": float(merged_eval["exact_match_rate_pct"]),
        "http_server": "local, 127.0.0.1, in-process (nothing deployed)",
        "http_status": http_check["http_status"],
        "http_response": http_check["response"],
    }


def main() -> None:
    res = verify_weight_merge_and_serving()
    dots = res["hlo_dot_ops"]
    print("=" * 76)
    print(" Experiment 05: LoRA Weight Merging (W_base + A @ B) & Local HTTP Endpoint")
    print("=" * 76)
    print(
        f" LoRA Parameters in Graph     : {res['lora_params_before_merge']:,} (two-branch) -> "
        f"{res['lora_params_after_merge']:,} (merged)"
    )
    print(
        f" Merged Model Total Params    : {res['merged_total_params']:,} "
        f"(base model: {res['base_total_params']:,})"
    )
    print(f" Max |Logits_LoRA - Merged|   : {res['max_logits_abs_diff']:.2e} (judge: < 1e-5)")
    print(
        f" Compiled HLO dot ops         : base {dots['base']} | merged {dots['merged']} | "
        f"two-branch {dots['two_branch_lora']}"
    )
    print("-" * 76)
    print(f" First Call (trace+compile)   : {res['unmerged_first_call_ms']:.1f} ms two-branch, "
          f"{res['merged_first_call_ms']:.1f} ms merged")
    n = res["num_iters"]
    print(f" Median ms per call after warmup (batch 12x31, n={n}, interleaved pairs):")
    print(f" Two-Branch LoRA Forward      : {res['unmerged_forward_ms']:.4f} ms")
    print(f" Merged Single-Branch Forward : {res['merged_forward_ms']:.4f} ms")
    ratio = res["merge_inference_speedup"]
    print(f" Merged vs Two-Branch         : {ratio:.2f}x (median per-pair ratio)")
    print(f" Merged Exact Match (train)   : {res['merged_exact_match_rate_pct']:.1f}% of 12 cases")
    print("-" * 76)
    status = res["http_status"]
    print(f" Local POST /v1/triage        : HTTP {status} (127.0.0.1, nothing deployed)")
    r = res["http_response"]
    print(f" Response triage              : {json.dumps(r['triage'], ensure_ascii=False)}")
    print(f" Response schema_valid        : {r['schema_valid']} ({r['serving_mode']})")
    print("=" * 76)


if __name__ == "__main__":
    main()
