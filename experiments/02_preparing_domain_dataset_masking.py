"""Experiment 02: Domain Dataset Engineering & Completion-Only Loss Masking.

Demonstrates how to pack variable-length `(instruction, JSON completion)` pairs into
static-shape batches (`batch_size=12, seq_len=32`) so `@nnx.jit` traces once per model and
never retraces, and compares a `loss_mask` that is `0.0` on instruction/padding tokens and
`1.0` strictly on the target JSON completion tokens against no prompt mask.

Both runs fine-tune the full toy model (no LoRA here), from the same seed, for 12 steps on
the same 12 training examples, and are scored with the same yardstick: cross-entropy on the
JSON tokens only. One seed, training set, so it is a modest effect, not a benchmark.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
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


def build_masked_domain_batch(
    prompt_len: int = 16,
    total_seq_len: int = 32,
) -> dict[str, jax.Array]:
    """Packs all 12 Spanish incident examples into static-shape arrays with completion masks.

    Sequence layout (length 32):
      - Positions [0 .. 15]  : Prompt tokens (<bos>, left-padding, incident words, `=>`)
      - Positions [16 .. 29] : 14 JSON completion tokens (`{`, `"sev"`, ..., `}`, `<eos>`)
      - Positions [30 .. 31] : 2 trailing `<pad>` tokens for static shape alignment (32)

    For next-token prediction (`inputs = seq[:, :-1]`, `targets = seq[:, 1:]`, length 31):
      - `completion_mask` is `1.0` strictly where `targets` correspond to the 14 JSON completion
        tokens (indices 15..28 in `targets`), and `0.0` on all prompt and `<pad>` positions.
      - `unmasked_nonpad_mask` is `1.0` on all non-padding target tokens (both prompt words and
        JSON completion tokens) to illustrate what happens when you forget to mask the prompt.
    """
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    sequences: list[list[int]] = []
    completion_masks: list[list[float]] = []
    unmasked_masks: list[list[float]] = []

    for ex in exp01.DOMAIN_INCIDENT_DATASET:
        p_ids = exp01.encode_prompt_tokens(ex.incident_text, prompt_len=prompt_len)
        c_ids = exp01.encode_json_completion_tokens(ex.expected_dict)
        pad_tail = [exp01.PAD_ID] * (total_seq_len - len(p_ids) - len(c_ids))
        full_seq = p_ids + c_ids + pad_tail
        assert len(full_seq) == total_seq_len

        # Target at index t is full_seq[t + 1]
        comp_start = prompt_len - 1  # target index 15 predicts full_seq[16] (`{`)
        comp_end = comp_start + len(c_ids)  # target index 28 predicts full_seq[29] (`<eos>`)

        c_mask = [
            1.0 if (comp_start <= t < comp_end) else 0.0 for t in range(total_seq_len - 1)
        ]
        u_mask = [
            1.0 if full_seq[t + 1] != exp01.PAD_ID else 0.0 for t in range(total_seq_len - 1)
        ]
        sequences.append(full_seq)
        completion_masks.append(c_mask)
        unmasked_masks.append(u_mask)

    seq_arr = jnp.array(sequences, dtype=jnp.int32)
    return {
        "full_sequences": seq_arr,
        "input_ids": seq_arr[:, :-1],
        "target_ids": seq_arr[:, 1:],
        "completion_mask": jnp.array(completion_masks, dtype=jnp.float32),
        "unmasked_mask": jnp.array(unmasked_masks, dtype=jnp.float32),
    }


def masked_cross_entropy_loss(
    logits: jax.Array,
    targets: jax.Array,
    loss_mask: jax.Array,
) -> jax.Array:
    """Computes average token cross-entropy strictly over positions where `loss_mask > 0`."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_lp = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    masked_nll = -token_lp * loss_mask
    denom = jnp.maximum(jnp.sum(loss_mask), 1.0)
    return jnp.sum(masked_nll) / denom


def compare_prompt_vs_completion_masking(steps: int = 12) -> dict[str, Any]:
    """Compares training with full-sequence loss vs completion-only masked loss."""
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    batch = build_masked_domain_batch()

    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    comp_mask = batch["completion_mask"]
    unmasked_mask = batch["unmasked_mask"]

    total_active_unmasked = float(jnp.sum(unmasked_mask))
    total_active_completion = float(jnp.sum(comp_mask))
    wasted_prompt_token_share_pct = (
        (total_active_unmasked - total_active_completion) / total_active_unmasked
    ) * 100.0

    # Initialize two identical models with the exact same PRNG seed
    config = exp01.GemmaAppConfig()
    model_unmasked = exp01.GemmaForCausalLM(config, rngs=nnx.Rngs(101))
    model_masked = exp01.GemmaForCausalLM(config, rngs=nnx.Rngs(101))

    opt_unmasked = nnx.Optimizer(model_unmasked, optax.adamw(learning_rate=1.5e-3), wrt=nnx.Param)
    opt_masked = nnx.Optimizer(model_masked, optax.adamw(learning_rate=1.5e-3), wrt=nnx.Param)

    trace_counter = {"count": 0}

    @nnx.jit
    def train_step(
        mdl: Any,
        opt: nnx.Optimizer,
        inp: jax.Array,
        tgt: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        trace_counter["count"] += 1

        def loss_fn(m: Any) -> jax.Array:
            logits = m(inp)
            return masked_cross_entropy_loss(logits, tgt, mask)

        loss, grads = nnx.value_and_grad(loss_fn)(mdl)
        opt.update(mdl, grads)
        return loss

    @nnx.jit
    def eval_completion_metrics(
        mdl: Any,
        inp: jax.Array,
        tgt: jax.Array,
        c_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        logits = mdl(inp)
        c_loss = masked_cross_entropy_loss(logits, tgt, c_mask)
        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == tgt).astype(jnp.float32) * c_mask
        acc = jnp.sum(correct) / jnp.maximum(jnp.sum(c_mask), 1.0)
        return c_loss, acc

    init_comp_loss, init_comp_acc = eval_completion_metrics(
        model_masked, input_ids, target_ids, comp_mask
    )

    for _ in range(steps):
        _ = train_step(model_unmasked, opt_unmasked, input_ids, target_ids, unmasked_mask)
        _ = train_step(model_masked, opt_masked, input_ids, target_ids, comp_mask)

    unmasked_comp_loss, unmasked_comp_acc = eval_completion_metrics(
        model_unmasked, input_ids, target_ids, comp_mask
    )
    masked_comp_loss, masked_comp_acc = eval_completion_metrics(
        model_masked, input_ids, target_ids, comp_mask
    )

    return {
        "batch_shape": tuple(int(d) for d in batch["full_sequences"].shape),
        "steps_per_model": steps,
        "jit_trace_count": int(trace_counter["count"]),
        "supervised_tokens_unmasked": int(total_active_unmasked),
        "supervised_tokens_completion": int(total_active_completion),
        "wasted_prompt_token_share_pct": float(wasted_prompt_token_share_pct),
        "initial_completion_loss": float(init_comp_loss),
        "initial_completion_acc_pct": float(init_comp_acc * 100.0),
        "unmasked_completion_loss": float(unmasked_comp_loss),
        "unmasked_completion_acc_pct": float(unmasked_comp_acc * 100.0),
        "masked_completion_loss": float(masked_comp_loss),
        "masked_completion_acc_pct": float(masked_comp_acc * 100.0),
        "loss_improvement_ratio": float(unmasked_comp_loss / jnp.maximum(masked_comp_loss, 1e-6)),
    }


def main() -> None:
    res = compare_prompt_vs_completion_masking(steps=12)
    n_prompt = res["supervised_tokens_unmasked"] - res["supervised_tokens_completion"]
    print("=" * 76)
    print(" Experiment 02: Domain Dataset Engineering & Completion-Only Loss Masking")
    print("=" * 76)
    shape = res["batch_shape"]
    print(f" Static Padded Batch Shape    : {shape} int32 (16 prompt + 14 JSON + 2 pad)")
    print(
        f" @nnx.jit Traces (2x{res['steps_per_model']} steps) : {res['jit_trace_count']} "
        "(one per model, no retrace)"
    )
    print(
        f" Prompt Share Without Mask    : {res['wasted_prompt_token_share_pct']:.2f}% "
        f"({n_prompt} of {res['supervised_tokens_unmasked']} supervised tokens)"
    )
    print("-" * 76)
    print(" Full fine-tuning, same seed, 12 steps each; both scored on the JSON tokens")
    print(" of the same 12 training examples (one seed, training set)")
    print(
        f" Initial JSON Loss            : {res['initial_completion_loss']:.4f} "
        f"(JSON token acc {res['initial_completion_acc_pct']:.1f}%)"
    )
    print(
        f" No Prompt Mask (12 steps)    : {res['unmasked_completion_loss']:.4f} "
        f"(JSON token acc {res['unmasked_completion_acc_pct']:.1f}%)"
    )
    print(
        f" Completion-Masked (12 steps) : {res['masked_completion_loss']:.4f} "
        f"(JSON token acc {res['masked_completion_acc_pct']:.1f}%)"
    )
    ratio = res["loss_improvement_ratio"]
    print(f" JSON Loss, No Mask / Masked  : {ratio:.2f}x lower with the mask")
    print("=" * 76)


if __name__ == "__main__":
    main()
