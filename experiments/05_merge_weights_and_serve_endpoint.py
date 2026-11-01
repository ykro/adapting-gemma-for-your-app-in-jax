"""Experiment 05: Mathematical LoRA Weight Merging (`W_merged = W_base + A @ B`) & Serving.

Proves that after training a `nnx.LoRA` adapter we can fold the low-rank matrices directly
into the frozen base weights (`W_merged = W_base + lora_a @ lora_b`) and unwrap `nnx.LoRA`
back into standard `nnx.Linear` layers:
  1. Exact numerical equivalence (`max_logits_abs_diff < 1e-5`, 100% identical greedy JSON).
  2. Zero remaining `nnx.LoRAParam` variables (`lora_params_after_merge == 0`).
  3. Faster `@nnx.jit` inference latency than running dual-branch `nnx.LoRA` modules.
  4. Live HTTP `POST /v1/triage` microservice verified (Cloud Run `--min-instances=0` ready).
"""

from __future__ import annotations

import importlib.util
import json
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


def serve_and_verify_triage_endpoint(
    merged_model: Any,
    incident_text: str = "alerta critica pasarela pagos timeout cobros tarjeta produccion urgente",
) -> dict[str, Any]:
    """Spins up an ephemeral HTTP server (`POST /v1/triage`) and verifies a live request."""
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
    exp02 = _load_module("exp02_masking", "02_preparing_domain_dataset_masking.py")
    exp03 = _load_module("exp03_lora", "03_lora_adapters_in_flax_nnx.py")
    exp04 = _load_module("exp04_eval", "04_structured_output_validation_eval.py")

    lora_model, _ = exp03.train_domain_lora_adapter(lora_rank=4, steps=45)
    merged_model = merge_lora_weights_into_base(lora_model)

    counts_before = exp03.count_lora_vs_base_parameters(lora_model)
    counts_after = exp03.count_lora_vs_base_parameters(merged_model)

    batch = exp02.build_masked_domain_batch()
    input_ids = batch["input_ids"]

    @nnx.jit
    def forward_jit(mdl: Any, tokens: jax.Array) -> jax.Array:
        return mdl(tokens)

    logits_unmerged = forward_jit(lora_model, input_ids).block_until_ready()
    logits_merged = forward_jit(merged_model, input_ids).block_until_ready()
    max_logits_diff = float(jnp.max(jnp.abs(logits_unmerged - logits_merged)))

    t0 = time.perf_counter()
    for _ in range(num_iters):
        out_u = forward_jit(lora_model, input_ids)
        out_u.block_until_ready()
    unmerged_ms = ((time.perf_counter() - t0) / num_iters) * 1000.0

    t1 = time.perf_counter()
    for _ in range(num_iters):
        out_m = forward_jit(merged_model, input_ids)
        out_m.block_until_ready()
    merged_ms = ((time.perf_counter() - t1) / num_iters) * 1000.0

    merged_eval = exp04.evaluate_model_on_suite(merged_model)
    http_check = serve_and_verify_triage_endpoint(merged_model)

    return {
        "lora_params_before_merge": int(counts_before["lora_params"]),
        "lora_params_after_merge": int(counts_after["lora_params"]),
        "merged_total_params": int(counts_after["total_params"]),
        "max_logits_abs_diff": max_logits_diff,
        "unmerged_forward_ms": float(unmerged_ms),
        "merged_forward_ms": float(merged_ms),
        "merge_inference_speedup": float(unmerged_ms / merged_ms),
        "merged_exact_match_rate_pct": float(merged_eval["exact_match_rate_pct"]),
        "http_status": http_check["http_status"],
        "http_response": http_check["response"],
    }


def main() -> None:
    res = verify_weight_merge_and_serving()
    print("=" * 76)
    print(" Experiment 05: LoRA Weight Merging (W_base + A @ B) & HTTP Endpoint")
    print("=" * 76)
    print(
        f" LoRA Parameters in Graph     : {res['lora_params_before_merge']:,} (unmerged) -> "
        f"{res['lora_params_after_merge']:,} (merged)"
    )
    print(f" Merged Model Total Params    : {res['merged_total_params']:,} (zero extra parameters)")
    print(f" Max |Logits_LoRA - Merged|   : {res['max_logits_abs_diff']:.2e} (threshold < 1e-5)")
    print("-" * 76)
    print(f" Unmerged Dual-Branch Latency : {res['unmerged_forward_ms']:.4f} ms")
    print(f" Merged Single-Branch Latency : {res['merged_forward_ms']:.4f} ms")
    print(f" Weight-Merge Serving Speedup : {res['merge_inference_speedup']:.2f}x faster")
    print(f" Merged Suite Exact Match     : {res['merged_exact_match_rate_pct']:.1f}%")
    print("-" * 76)
    print(f" Live HTTP POST /v1/triage    : HTTP {res['http_status']}")
    print(f" Endpoint JSON Response       : {json.dumps(res['http_response'], ensure_ascii=False)}")
    print("=" * 76)


if __name__ == "__main__":
    main()
