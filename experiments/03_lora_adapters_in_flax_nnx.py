"""Experiment 03: Freezing 98% of the Model & Training a Surgical LoRA Adapter in Flax NNX.

Demonstrates how a low-rank update matrix `Delta_W = A @ B` (`rank=4`) wrapped via
`nnx.LoRA` and filtered via `nnx.LoRAParam` vs `nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))`
allows adapting a Gemma model to a Spanish structured JSON task while:
  1. Reducing trainable parameters by ~97.90% (training only ~2.10% of weights).
  2. Preserving 100% of base model weights bit-for-bit (`max_base_weight_diff == 0.0`).
  3. Driving completion-masked cross-entropy loss down with `optax.adamw` in seconds.
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


def inject_lora_adapters(
    model: Any,
    lora_rank: int = 4,
    *,
    include_mlp: bool = True,
    rngs: nnx.Rngs,
) -> Any:
    """Surgically wraps target Gemma Linear projections with `nnx.LoRA` adapters in-place."""
    for block in model.blocks:
        attn = block.attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base_linear: nnx.Linear = getattr(attn, proj_name)
            lora_wrapped = nnx.LoRA(
                in_features=base_linear.in_features,
                lora_rank=lora_rank,
                out_features=base_linear.out_features,
                base_module=base_linear,
                rngs=rngs,
            )
            setattr(attn, proj_name, lora_wrapped)

        if include_mlp:
            mlp = block.mlp
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                base_linear = getattr(mlp, proj_name)
                lora_wrapped = nnx.LoRA(
                    in_features=base_linear.in_features,
                    lora_rank=lora_rank,
                    out_features=base_linear.out_features,
                    base_module=base_linear,
                    rngs=rngs,
                )
                setattr(mlp, proj_name, lora_wrapped)

    return model


def count_lora_vs_base_parameters(model: Any) -> dict[str, float | int]:
    """Splits model state into trainable `nnx.LoRAParam` vs frozen base `nnx.Param`."""
    base_filter = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    _, lora_state, base_state = nnx.split(model, nnx.LoRAParam, base_filter)

    lora_params = sum(x.size for x in jax.tree.leaves(lora_state))
    base_params = sum(x.size for x in jax.tree.leaves(base_state))
    total_params = lora_params + base_params
    trainable_ratio_pct = (lora_params / total_params) * 100.0
    reduction_pct = (1.0 - (lora_params / total_params)) * 100.0

    return {
        "base_params": int(base_params),
        "lora_params": int(lora_params),
        "total_params": int(total_params),
        "trainable_ratio_pct": float(trainable_ratio_pct),
        "reduction_pct": float(reduction_pct),
    }


def train_domain_lora_adapter(
    lora_rank: int = 4,
    steps: int = 45,
    learning_rate: float = 1.0e-2,
) -> tuple[Any, dict[str, Any]]:
    """Trains a surgical LoRA adapter on the Spanish incident dataset with completion masking."""
    exp01 = _load_module("exp01_prompting", "01_when_prompting_hits_a_wall.py")
    exp02 = _load_module("exp02_masking", "02_preparing_domain_dataset_masking.py")

    config = exp01.GemmaAppConfig()
    model = exp01.GemmaForCausalLM(config, rngs=nnx.Rngs(7))
    model = inject_lora_adapters(model, lora_rank=lora_rank, include_mlp=True, rngs=nnx.Rngs(13))

    counts = count_lora_vs_base_parameters(model)

    base_filter = nnx.All(nnx.Param, nnx.Not(nnx.LoRAParam))
    _, lora_before, base_before = nnx.split(model, nnx.LoRAParam, base_filter)
    base_leaves_before = [jnp.copy(x) for x in jax.tree.leaves(base_before)]
    lora_leaves_before = [jnp.copy(x) for x in jax.tree.leaves(lora_before)]

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),
        wrt=nnx.LoRAParam,
    )

    batch = exp02.build_masked_domain_batch()
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    comp_mask = batch["completion_mask"]

    @nnx.jit
    def lora_train_step(
        mdl: Any,
        opt: nnx.Optimizer,
        inp: jax.Array,
        tgt: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        def loss_fn(m: Any) -> jax.Array:
            logits = m(inp)
            return exp02.masked_cross_entropy_loss(logits, tgt, mask)

        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.LoRAParam))
        loss, grads = grad_fn(mdl)
        opt.update(mdl, grads)
        return loss

    @nnx.jit
    def eval_loss(mdl: Any, inp: jax.Array, tgt: jax.Array, mask: jax.Array) -> jax.Array:
        logits = mdl(inp)
        return exp02.masked_cross_entropy_loss(logits, tgt, mask)

    loss_before = float(eval_loss(model, input_ids, target_ids, comp_mask))
    loss_history: list[float] = [loss_before]

    for _ in range(steps):
        step_loss = lora_train_step(model, optimizer, input_ids, target_ids, comp_mask)
        loss_history.append(float(step_loss))

    loss_after = float(eval_loss(model, input_ids, target_ids, comp_mask))

    _, lora_after, base_after = nnx.split(model, nnx.LoRAParam, base_filter)
    base_leaves_after = jax.tree.leaves(base_after)
    lora_leaves_after = jax.tree.leaves(lora_after)

    max_base_diff = max(
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(base_leaves_before, base_leaves_after)
    )
    max_lora_diff = max(
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(lora_leaves_before, lora_leaves_after)
    )

    metrics = {
        **counts,
        "lora_rank": lora_rank,
        "steps": steps,
        "loss_before": loss_before,
        "loss_step_20": loss_history[min(20, len(loss_history) - 1)],
        "loss_after": loss_after,
        "max_base_weight_diff": max_base_diff,
        "max_lora_weight_diff": max_lora_diff,
        "base_weights_frozen_bitwise": max_base_diff == 0.0,
        "lora_weights_updated": max_lora_diff > 0.0,
    }
    return model, metrics


def main() -> None:
    _, res = train_domain_lora_adapter(lora_rank=4, steps=45)
    print("=" * 74)
    print(" Experiment 03: Surgical LoRA Injection & Adaptation in Flax NNX")
    print("=" * 74)
    print(f" Frozen Base Parameters    : {res['base_params']:,}")
    print(f" Trainable LoRA Parameters : {res['lora_params']:,} (rank={res['lora_rank']})")
    print(f" Total Parameters          : {res['total_params']:,}")
    print(
        f" Trainable Parameter Share : {res['trainable_ratio_pct']:.2f}% "
        f"({res['reduction_pct']:.2f}% parameter reduction)"
    )
    print("-" * 74)
    print(
        f" Completion-Masked Loss    : {res['loss_before']:.4f} (step 0) -> "
        f"{res['loss_step_20']:.4f} (step 20) -> {res['loss_after']:.4f} (step {res['steps']})"
    )
    print(
        f" Base Weight Max |Diff|    : {res['max_base_weight_diff']:.8f} "
        f"(frozen={res['base_weights_frozen_bitwise']})"
    )
    print(
        f" LoRA Adapter Max |Diff|   : {res['max_lora_weight_diff']:.6f} "
        f"(updated={res['lora_weights_updated']})"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
