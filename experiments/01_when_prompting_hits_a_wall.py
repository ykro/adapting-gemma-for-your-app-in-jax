"""Experiment 01: When Few-Shot Prompting Hits a Wall (Latency, Token Cost & Schema Accuracy).

Demonstrates why software engineers building high-volume domain applications in Spanish
(such as operational incident triage into strict JSON commands) hit three walls with
long few-shot prompts:
  1. Prefill compute latency grows with prompt length (208 few-shot tokens vs 16 compact tokens).
  2. Input tokens and the KV-cache footprint are 13.0x larger on every single request.
  3. A model that was never adapted does not produce the JSON contract from a 16-token prompt.

The model is a small Gemma-style decoder (4 blocks, d_model 256, ~3.7M parameters) with
random weights and a hand-made 64-word vocabulary. It is NOT a Gemma checkpoint: the
mechanics carry over to a real Gemma, the quality numbers do not. Both prefill timings use
the same model and weights, compiled with @nnx.jit, and report the median per call after a
warmup call (compilation is reported apart). LoRA plays no part in this comparison.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class GemmaAppConfig:
    """Compact Gemma-style config for fast, deterministic local & CI execution."""

    vocab_size: int = 1024
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 32
    d_ff: int = 768
    num_layers: int = 4
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6


# Deterministic vocabulary for Spanish incident reports and structured JSON tokens
VOCAB_TOKENS: tuple[str, ...] = (
    "<pad>",       # 0
    "<bos>",       # 1
    "<eos>",       # 2
    "=>",          # 3
    "{",           # 4
    '"sev"',       # 5
    ":",           # 6
    '"P1"',        # 7
    '"P2"',        # 8
    '"P3"',        # 9
    ",",           # 10
    '"svc"',       # 11
    '"pagos"',     # 12
    '"auth"',      # 13
    '"db"',        # 14
    '"act"',       # 15
    '"rollback"',  # 16
    '"escalar"',   # 17
    '"cache"',     # 18
    "}",           # 19
    "incidente",   # 20
    "alerta",      # 21
    "critica",     # 22
    "pasarela",    # 23
    "pagos",       # 24
    "timeout",     # 25
    "cobros",      # 26
    "tarjeta",     # 27
    "error",       # 28
    "sincronizacion",  # 29
    "cache",       # 30
    "latencia",    # 31
    "moderada",    # 32
    "fallo",       # 33
    "critico",     # 34
    "servicio",    # 35
    "auth",        # 36
    "tokens",      # 37
    "jwt",         # 38
    "rechazados",  # 39
    "aviso",       # 40
    "expiracion",  # 41
    "sesion",      # 42
    "reintento",   # 43
    "login",       # 44
    "caida",       # 45
    "nodo",        # 46
    "primario",    # 47
    "db",          # 48
    "corrupcion",  # 49
    "indice",      # 50
    "transacciones",  # 51
    "uso",         # 52
    "disco",       # 53
    "replica",     # 54
    "lectura",     # 55
    "lenta",       # 56
    "produccion",  # 57
    "urgente",     # 58
    "clientes",    # 59
    "api",         # 60
    "gateway",     # 61
    "bloqueo",     # 62
    "conexiones",  # 63
)

TOKEN_TO_ID: dict[str, int] = {tok: idx for idx, tok in enumerate(VOCAB_TOKENS)}
ID_TO_TOKEN: dict[int, str] = {idx: tok for idx, tok in enumerate(VOCAB_TOKENS)}

PAD_ID = TOKEN_TO_ID["<pad>"]
BOS_ID = TOKEN_TO_ID["<bos>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
ARROW_ID = TOKEN_TO_ID["=>"]


@dataclass(frozen=True)
class IncidentExample:
    """Domain training/evaluation pair mapping a Spanish incident report to strict JSON."""

    category_id: int
    incident_text: str
    expected_dict: dict[str, str]

    @property
    def expected_json(self) -> str:
        return json.dumps(self.expected_dict, separators=(",", ":"), ensure_ascii=False)


DOMAIN_INCIDENT_DATASET: tuple[IncidentExample, ...] = (
    IncidentExample(
        category_id=0,
        incident_text="alerta critica pasarela pagos timeout cobros tarjeta produccion urgente",
        expected_dict={"sev": "P1", "svc": "pagos", "act": "rollback"},
    ),
    IncidentExample(
        category_id=1,
        incident_text="error sincronizacion cache pagos latencia moderada api gateway",
        expected_dict={"sev": "P2", "svc": "pagos", "act": "cache"},
    ),
    IncidentExample(
        category_id=2,
        incident_text="fallo critico servicio auth tokens jwt rechazados clientes produccion",
        expected_dict={"sev": "P1", "svc": "auth", "act": "rollback"},
    ),
    IncidentExample(
        category_id=3,
        incident_text="aviso expiracion sesion auth reintento login latencia moderada",
        expected_dict={"sev": "P3", "svc": "auth", "act": "escalar"},
    ),
    IncidentExample(
        category_id=4,
        incident_text="caida nodo primario db corrupcion indice transacciones urgente",
        expected_dict={"sev": "P1", "svc": "db", "act": "rollback"},
    ),
    IncidentExample(
        category_id=5,
        incident_text="alerta uso disco db replica lectura lenta bloqueo conexiones",
        expected_dict={"sev": "P2", "svc": "db", "act": "escalar"},
    ),
    # Paraphrased / reordered operational variants for evaluation generalization
    IncidentExample(
        category_id=0,
        incident_text="urgente produccion pasarela pagos alerta critica timeout cobros tarjeta",
        expected_dict={"sev": "P1", "svc": "pagos", "act": "rollback"},
    ),
    IncidentExample(
        category_id=1,
        incident_text="api gateway pagos error sincronizacion cache latencia moderada",
        expected_dict={"sev": "P2", "svc": "pagos", "act": "cache"},
    ),
    IncidentExample(
        category_id=2,
        incident_text="produccion servicio auth fallo critico tokens jwt rechazados urgente",
        expected_dict={"sev": "P1", "svc": "auth", "act": "rollback"},
    ),
    IncidentExample(
        category_id=3,
        incident_text="reintento login servicio auth aviso expiracion sesion moderada",
        expected_dict={"sev": "P3", "svc": "auth", "act": "escalar"},
    ),
    IncidentExample(
        category_id=4,
        incident_text="urgente db caida nodo primario corrupcion indice transacciones",
        expected_dict={"sev": "P1", "svc": "db", "act": "rollback"},
    ),
    IncidentExample(
        category_id=5,
        incident_text="replica db lectura lenta alerta uso disco bloqueo conexiones",
        expected_dict={"sev": "P2", "svc": "db", "act": "escalar"},
    ),
)


def encode_prompt_tokens(incident_text: str, prompt_len: int = 16) -> list[int]:
    """Encodes a Spanish incident text into a fixed-length prompt ending with `=>`."""
    words = incident_text.strip().split()
    word_ids = [TOKEN_TO_ID.get(w, TOKEN_TO_ID["incidente"]) for w in words]
    # Reserve first token for <bos> and last token of the prompt window for `=>`
    max_words = prompt_len - 2
    trimmed = word_ids[:max_words]
    pad_count = max_words - len(trimmed)
    return [BOS_ID] + ([PAD_ID] * pad_count) + trimmed + [ARROW_ID]


def encode_json_completion_tokens(expected_dict: dict[str, str]) -> list[int]:
    """Encodes `{"sev":..., "svc":..., "act":...}<eos>` into 12 deterministic token IDs."""
    sev_tok = f'"{expected_dict["sev"]}"'
    svc_tok = f'"{expected_dict["svc"]}"'
    act_tok = f'"{expected_dict["act"]}"'
    tokens = [
        "{",
        '"sev"',
        ":",
        sev_tok,
        ",",
        '"svc"',
        ":",
        svc_tok,
        ",",
        '"act"',
        ":",
        act_tok,
        "}",
        "<eos>",
    ]
    return [TOKEN_TO_ID[t] for t in tokens]


def decode_tokens(token_ids: list[int]) -> str:
    """Converts a sequence of token IDs back into a compact JSON string."""
    pieces: list[str] = []
    for tid in token_ids:
        if tid == EOS_ID:
            break
        if tid in (PAD_ID, BOS_ID, ARROW_ID):
            continue
        pieces.append(ID_TO_TOKEN.get(int(tid), f"<unk:{tid}>"))
    return "".join(pieces)


class GemmaRMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization with Gemma's (1 + scale) parameterization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.eps = eps
        self.scale = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return normed * (1.0 + self.scale[...])


def apply_rope(x: jax.Array, theta: float = 10_000.0) -> jax.Array:
    """Applies Rotary Position Embeddings (RoPE) to query/key tensors [B, T, H, D]."""
    _, seq_len, _, head_dim = x.shape
    half_dim = head_dim // 2
    freq_seq = jnp.arange(0, half_dim, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (freq_seq / half_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.einsum("t,d->td", positions, inv_freq)
    sin = jnp.sin(angles)[None, :, None, :]
    cos = jnp.cos(angles)[None, :, None, :]
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class GemmaAttention(nnx.Module):
    """Grouped-Query Attention (GQA) with RoPE for Gemma causal decoding."""

    def __init__(self, config: GemmaAppConfig, *, rngs: nnx.Rngs) -> None:
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.n_heads // config.n_kv_heads
        self.rope_theta = config.rope_theta

        q_out = config.n_heads * config.head_dim
        kv_out = config.n_kv_heads * config.head_dim

        self.q_proj = nnx.Linear(config.d_model, q_out, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(config.d_model, kv_out, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(config.d_model, kv_out, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(q_out, config.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, theta=self.rope_theta)
        k = apply_rope(k, theta=self.rope_theta)

        if self.num_queries_per_kv > 1:
            k = jnp.repeat(k, self.num_queries_per_kv, axis=2)
            v = jnp.repeat(v, self.num_queries_per_kv, axis=2)

        scale = 1.0 / jnp.sqrt(float(self.head_dim))
        scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=-1)

        context = jnp.einsum("bhts,bshd->bthd", attn_weights, v)
        context = context.reshape(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(context)


class GemmaGeGLUMLP(nnx.Module):
    """Gated GELU MLP (gate_proj, up_proj, down_proj) used in Gemma blocks."""

    def __init__(self, config: GemmaAppConfig, *, rngs: nnx.Rngs) -> None:
        self.gate_proj = nnx.Linear(config.d_model, config.d_ff, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(config.d_model, config.d_ff, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(config.d_ff, config.d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        gated = jax.nn.gelu(self.gate_proj(x), approximate=True) * self.up_proj(x)
        return self.down_proj(gated)


class GemmaDecoderBlock(nnx.Module):
    """Pre-norm residual decoder block with RMSNorm, GQA, and GeGLU."""

    def __init__(self, config: GemmaAppConfig, *, rngs: nnx.Rngs) -> None:
        self.in_norm = GemmaRMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = GemmaAttention(config, rngs=rngs)
        self.post_attn_norm = GemmaRMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = GemmaGeGLUMLP(config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.in_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class GemmaForCausalLM(nnx.Module):
    """Gemma-style Causal Language Model authored in Flax NNX."""

    def __init__(self, config: GemmaAppConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        self.embed = nnx.Embed(num_embeddings=config.vocab_size, features=config.d_model, rngs=rngs)
        self.blocks = nnx.List(
            [GemmaDecoderBlock(config, rngs=rngs) for _ in range(config.num_layers)]
        )
        self.final_norm = GemmaRMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            kernel_init=nnx.initializers.normal(0.035),
            rngs=rngs,
        )

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        h = self.embed(token_ids)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.lm_head(h)


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def count_params(model: Any) -> int:
    """Number of scalars in every `nnx.Param` of the model."""
    return int(sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))))


def benchmark_prompting_wall(
    batch_size: int = 8,
    fewshot_seq_len: int = 208,
    compact_seq_len: int = 16,
    num_iters: int = 25,
) -> dict[str, Any]:
    """Compares long few-shot prompting (208 tokens) vs compact adapted prompt (16 tokens)."""
    config = GemmaAppConfig()
    model = GemmaForCausalLM(config, rngs=nnx.Rngs(42))

    @nnx.jit
    def forward_prefill(mdl: GemmaForCausalLM, tokens: jax.Array) -> jax.Array:
        return mdl(tokens)

    # Build deterministic token batches for 6-example few-shot prompt (208 tokens) vs compact (16)
    compact_rows = [
        encode_prompt_tokens(ex.incident_text, prompt_len=compact_seq_len)
        for ex in DOMAIN_INCIDENT_DATASET[:batch_size]
    ]
    compact_batch = jnp.array(compact_rows, dtype=jnp.int32)

    # Few-shot prompt prepends 6 full (prompt + JSON completion) demonstrations = 192 prefix tokens
    prefix_tokens: list[int] = []
    for demo in DOMAIN_INCIDENT_DATASET[:6]:
        p_ids = encode_prompt_tokens(demo.incident_text, prompt_len=18)
        c_ids = encode_json_completion_tokens(demo.expected_dict)
        prefix_tokens.extend(p_ids + c_ids)  # 18 + 14 = 32 tokens * 6 = 192 tokens
    assert len(prefix_tokens) == fewshot_seq_len - compact_seq_len

    fewshot_rows = [prefix_tokens + row for row in compact_rows]
    fewshot_batch = jnp.array(fewshot_rows, dtype=jnp.int32)

    # First call per shape traces + compiles; it is timed apart and excluded from the median.
    t0 = time.perf_counter()
    forward_prefill(model, fewshot_batch).block_until_ready()
    fewshot_compile_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    forward_prefill(model, compact_batch).block_until_ready()
    compact_compile_ms = (time.perf_counter() - t0) * 1000.0

    fewshot_ms = _median_ms(
        lambda: forward_prefill(model, fewshot_batch).block_until_ready(), num_iters
    )
    compact_ms = _median_ms(
        lambda: forward_prefill(model, compact_batch).block_until_ready(), num_iters
    )

    # KV cache memory in float32, computed from the shapes (not measured):
    # 2 (K,V) * num_layers * seq_len * n_kv_heads * head_dim * 4 bytes
    bytes_per_token_kv = 2 * config.num_layers * config.n_kv_heads * config.head_dim * 4
    fewshot_kv_kb = (fewshot_seq_len * bytes_per_token_kv * batch_size) / 1024.0
    compact_kv_kb = (compact_seq_len * bytes_per_token_kv * batch_size) / 1024.0
    kv_reduction_pct = (1.0 - (compact_kv_kb / fewshot_kv_kb)) * 100.0

    # Greedy output of the same never-adapted model (random weights) for the first report
    greedy_ids: list[int] = list(compact_rows[0])
    for _ in range(14):
        inp = jnp.array([greedy_ids], dtype=jnp.int32)
        logits = model(inp)
        next_id = int(jnp.argmax(logits[0, -1, :]))
        greedy_ids.append(next_id)
    unadapted_decoded = decode_tokens(greedy_ids[compact_seq_len:])
    try:
        parsed = json.loads(unadapted_decoded)
        unadapted_valid_json = isinstance(parsed, dict) and set(parsed.keys()) == {
            "sev",
            "svc",
            "act",
        }
    except Exception:
        unadapted_valid_json = False

    return {
        "backend": jax.default_backend(),
        "model": "toy Gemma-style decoder, random weights (not a Gemma checkpoint)",
        "model_params": count_params(model),
        "num_layers": config.num_layers,
        "d_model": config.d_model,
        "timing": f"median of {num_iters} calls after one warmup call",
        "num_iters": num_iters,
        "batch_size": batch_size,
        "fewshot_seq_len": fewshot_seq_len,
        "compact_seq_len": compact_seq_len,
        "token_overhead_ratio": float(fewshot_seq_len / compact_seq_len),
        "fewshot_first_call_ms": float(fewshot_compile_ms),
        "compact_first_call_ms": float(compact_compile_ms),
        "fewshot_prefill_ms": float(fewshot_ms),
        "compact_prefill_ms": float(compact_ms),
        "prefill_speedup": float(fewshot_ms / compact_ms),
        "fewshot_kv_cache_kb": float(fewshot_kv_kb),
        "compact_kv_cache_kb": float(compact_kv_kb),
        "kv_cache_reduction_pct": float(kv_reduction_pct),
        "unadapted_sample_output": unadapted_decoded,
        "unadapted_valid_json": unadapted_valid_json,
    }


def main() -> None:
    res = benchmark_prompting_wall()
    print("=" * 76)
    print(" Experiment 01: When Few-Shot Prompting Hits a Wall (Gemma-style, Flax NNX)")
    print("=" * 76)
    print(
        f" Model                       : {res['num_layers']} blocks, d_model {res['d_model']}, "
        f"{res['model_params']:,} params, random weights"
    )
    print(" Not a Gemma checkpoint      : same mechanics, quality numbers do not carry over")
    print(f" Backend Device              : {res['backend']}")
    print(
        f" Few-Shot Context Length     : {res['fewshot_seq_len']} tokens "
        "(6 examples x 32 + report)"
    )
    print(
        f" Compact Prompt Length       : {res['compact_seq_len']} tokens "
        "(report only, no examples)"
    )
    print(
        f" Input Token Multiplier      : {res['token_overhead_ratio']:.1f}x "
        "more tokens per request"
    )
    print("-" * 76)
    print(f" Same model and weights, both @nnx.jit, batch={res['batch_size']}, LoRA not involved")
    print(f" First Call 208t (compile)   : {res['fewshot_first_call_ms']:.1f} ms (timed apart)")
    print(f" First Call 16t (compile)    : {res['compact_first_call_ms']:.1f} ms (timed apart)")
    print(f" Median ms per call after warmup (n={res['num_iters']}):")
    print(f" Few-Shot Prefill (208t)     : {res['fewshot_prefill_ms']:.4f} ms")
    print(f" Compact Prefill (16t)       : {res['compact_prefill_ms']:.4f} ms")
    print(
        f" Prefill Speedup (time)      : {res['prefill_speedup']:.2f}x "
        f"(tokens: {res['token_overhead_ratio']:.1f}x)"
    )
    print(
        f" KV-Cache, batch (computed)  : {res['fewshot_kv_cache_kb']:.1f} KB -> "
        f"{res['compact_kv_cache_kb']:.1f} KB ({res['kv_cache_reduction_pct']:.2f}% less)"
    )
    print("-" * 76)
    print(f" Untrained Model, 16t Prompt : {res['unadapted_sample_output'][:44]}...")
    print(f" Valid JSON Schema           : {res['unadapted_valid_json']}")
    print("=" * 76)


if __name__ == "__main__":
    main()
