# Cómo Adaptar un Foundational Model (`Gemma`) a tu Aplicación con `Flax NNX` y `LoRA`

- **Por qué es relevante:** Cuando una aplicación en producción exige respuestas estructuradas estrictas (como convertir reportes de incidentes operativos en español hacia comandos JSON tipados), el *few-shot prompting* repetitivo multiplica por `13.0x` el costo en tokens de entrada y el tamaño del `KV Cache` en cada request. Adaptar los pesos con `Flax NNX` y `LoRA` permite eliminar los ejemplos del prompt, congelar el `97.92%` del modelo base (`Δbase = 0.0`) y fusionar matemáticamente `W_merged = W_base + A @ B` para servir en `Cloud Run` con cero latencia extra.
- **Para quién es:** Ingenieros de software, desarrolladores backend y arquitectos de sistemas que buscan integrar y adaptar modelos fundacionales abiertos (`Gemma`) dentro de microservicios reales usando `JAX` y `Flax NNX`.
- **Quién debería verlo:** Equipos de ingeniería que operan clasificadores de tickets, extracción de entidades tipadas, ruteo de comandos operativos o validación de esquemas JSON y desean reemplazar prompts largos y frágiles por adaptadores deterministas verificados con `pytest`.
- **Serie completa:** Quinto repositorio práctico (**B5**) de la guía [Field Guide: De Software Dev a Machine Learning y Foundational Models con JAX](https://github.com/ykro/jax-ml-for-software-devs-field-guide) (y puente directo hacia la [Serie A — JAX & OpenXLA Field Guide](https://github.com/ykro/jax-openxla-field-guide)).

---

## Arquitectura y Modelo Mental

### 1. Cuándo el *Few-Shot Prompting* Toca Techo (`208` tokens vs `16` tokens)

En un microservicio que recibe miles de reportes en español por hora, adjuntar 6 ejemplos JSON en cada llamada obliga a procesar `208` tokens de prefill por un reporte de apenas `16` tokens (`13.0x` más tokens facturados y `6656.0 KB` de `KV Cache` por `batch`). Al trasladar las reglas del dominio a los pesos mediante `LoRA`, el prompt baja a `16` tokens, ahorra el `92.31%` de memoria de `KV Cache` y acelera el prefill **4.80x**.

![Comparación entre few-shot prompting y modelo Gemma adaptado con LoRA](diagrams/01-fewshot-prompt-vs-lora.svg)

### 2. Ingeniería de Datos y `Completion-Only Loss Masking`

Al empaquetar pares `(instrucción en español, respuesta JSON)` en un `batch` estático de forma `(12, 32)` evitamos recompilaciones dinámicas en `@nnx.jit` (`0` recompilaciones tras el primer trace). Aplicando una máscara `loss_mask = 0.0` sobre los tokens del prompt y del padding (`40.43%` de tokens ignorados) y `loss_mask = 1.0` exclusivamente sobre los `14` tokens de la respuesta JSON (`{"sev":"P1","svc":"pagos","act":"rollback"}<eos>`), el optimizador concentra el `100%` del gradiente en aprender el contrato de salida, logrando **1.48x menor pérdida** sobre el JSON objetivo en el mismo número de pasos.

![Flujo de empaquetado estático y Completion-Only Loss Masking](diagrams/02-completion-only-loss-masking.svg)

### 3. Adaptadores Quirúrgicos `nnx.LoRA` (`rank=4`) y Fusión Matemática `W_merged = W_base + A @ B`

Durante el entrenamiento con `Flax NNX`, envolvemos las proyecciones de atención (`q_proj`, `k_proj`, `v_proj`, `o_proj`) y del bloque `GeGLU` (`gate_proj`, `up_proj`, `down_proj`) con `nnx.LoRA`. Filtrando el estado con `nnx.LoRAParam`, congelamos `3,672,320` parámetros base (`97.92%` intactos, `max_base_diff == 0.0`) y actualizamos únicamente `77,824` escalares (`2.08%`). Al terminar el entrenamiento, plegamos algebraicamente `W_merged = W_base + (lora_a @ lora_b)` de vuelta en cada capa `nnx.Linear` para servir una sola rama en `Cloud Run` (`--min-instances=0`) con paridad numérica (`max_diff = 9.95e-06 < 1e-5`) y **1.31x** menor latencia de inferencia frente al grafo con ramas separadas.

![Entrenamiento con nnx.LoRA y fusión matemática de pesos para producción](diagrams/03-lora-train-and-weight-merge.svg)

---

## Resultados Reales de Verificación (`verify_adaptation.py`)

Benchmarks deterministas ejecutados con Python `3.12.11`, JAX `0.12.0.1` y Flax `0.12.12`:

| Experimento | Archivo | Baseline / Antes de LoRA | Gemma Adaptado (`Flax NNX`) | Verificación / Ganancia |
| :--- | :--- | :--- | :--- | :--- |
| **01. Few-Shot vs Compact Prefill** | `experiments/01_when_prompting_hits_a_wall.py` | `34.09 ms` (`208` tokens, `6656 KB` KV) | `7.11 ms` (`16` tokens, `512 KB` KV) | **4.80x speedup** (**-92.31% KV Cache**, `13.0x` menos tokens) |
| **02. Prompt vs Completion Masking** | `experiments/02_preparing_domain_dataset_masking.py` | `0.0677` CE loss (sin máscara de prompt) | `0.0456` CE loss (`loss_mask=1` en JSON) | **1.48x menor loss** (`100%` exactitud tokens JSON, `0` recompiles) |
| **03. Surgical LoRA (`rank=4`, NNX)** | `experiments/03_lora_adapters_in_flax_nnx.py` | `3,750,144` parámetros totales | `77,824` entrenables (`2.08%`) | **97.92% reducción** (`max_base_diff = 0.00000000`) |
| **04. Deterministic JSON Test Suite** | `experiments/04_structured_output_validation_eval.py` | `0.0%` JSON válido / `0.0%` Exact Match | `100.0%` JSON válido / `100.0%` Exact Match | **12/12 casos PASS** (sintaxis, esquema, enums y campos) |
| **05. Merge `W_base + A@B` & HTTP API** | `experiments/05_merge_weights_and_serve_endpoint.py` | `11.66 ms` (rama doble `nnx.LoRA`) | `9.83 ms` (`W_merged` en `nnx.Linear`) | **`Δ = 9.95e-06 < 1e-5`**, `0` `LoRAParam`, `POST /v1/triage` **HTTP 200** |

Salida íntegra de terminal generada por `verify_adaptation.py`:

```text
==============================================================================
THE OUTSIDE JUDGE — Adapting Gemma for Your App in JAX Verification Report
==============================================================================
Runtime : Python 3.12.11 | JAX 0.12.0.1 | Flax 0.12.12 | Backend: cpu
------------------------------------------------------------------------------
Experiment                         | Baseline / Before  | Adapted / JAX      | Verification
------------------------------------------------------------------------------
01. Few-Shot vs Compact Prefill    |  34.09 ms (208t)   |   7.11 ms (16t)    | 4.80x (-92.3% KV)
02. Prompt vs Completion Masking   | 0.0677 CE loss     | 0.0456 CE loss     | 1.48x (100% JSON)
03. Surgical LoRA (rank=4, NNX)    | 3,750,144 params   | 77,824 trainable   | -97.92% (Δbase=0.0)
04. Deterministic JSON Test Suite  |   0.0% exact match | 100.0% exact match | 12/12 Schema PASS
05. Merge W_base + A@B & HTTP API  |  11.66 ms (LoRA)   |   9.83 ms (Merged) | Δ=9.95e-06 (HTTP 200)
------------------------------------------------------------------------------
Saved JSON report to: benchmarks/latest_results.json
ALL DETERMINISTIC ADAPTATION CHECKS PASSED.
```

---

## Estructura del Repositorio

```text
.
├── benchmarks/
│   ├── latest_results.json                             # Métricas verificadas por verify_adaptation.py
│   └── terminal_output.txt                             # Salida íntegra del Outside Judge
├── diagrams/
│   ├── 01-fewshot-prompt-vs-lora.{html,svg,png}
│   ├── 02-completion-only-loss-masking.{html,svg,png}
│   └── 03-lora-train-and-weight-merge.{html,svg,png}
├── experiments/
│   ├── 01_when_prompting_hits_a_wall.py                # Latencia de prefill, KV cache y límites del few-shot prompting
│   ├── 02_preparing_domain_dataset_masking.py          # Batches estáticos (12, 32) y Completion-Only Loss Masking
│   ├── 03_lora_adapters_in_flax_nnx.py                 # Inyección quirúrgica nnx.LoRA (rank=4) y congelamiento del 97.92%
│   ├── 04_structured_output_validation_eval.py         # Suite de evaluación determinista (JSON Schema + Exact Match)
│   └── 05_merge_weights_and_serve_endpoint.py          # Fusión W_merged = W_base + A @ B y microservicio HTTP /v1/triage
├── gcp/
│   └── run_on_demand_adaptation.sh                     # Runner on-demand con auto-teardown (TPU v5e, GPU L4 y Cloud Run)
├── tests/
│   └── test_adaptation.py                              # Suite pytest determinista (5 tests)
├── Makefile
├── pyproject.toml
└── verify_adaptation.py                                # El Outside Judge (aserciones de paridad, LoRA y endpoint HTTP)
```

---

## Onboarding Paso a Paso

### 1. Instalación desde cero (macOS / Linux)

```bash
git clone https://github.com/ykro/adapting-gemma-for-your-app-in-jax.git
cd adapting-gemma-for-your-app-in-jax
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Recorrido guiado por los 5 experimentos (Orden recomendado)

Cada script es autocontenido, corre en pocos segundos en CPU, GPU o TPU y cubre una etapa del ciclo de adaptación de `Gemma` para una aplicación en español:

```bash
# Paso 1 — Medir el costo en tokens (13.0x), KV cache y latencia del few-shot prompting frente a un prompt compacto
python experiments/01_when_prompting_hits_a_wall.py

# Paso 2 — Empaquetar pares (instrucción, JSON) con padding estático y aplicar Completion-Only Loss Masking
python experiments/02_preparing_domain_dataset_masking.py

# Paso 3 — Congelar el 97.92% del modelo e inyectar adaptadores nnx.LoRA (rank=4) entrenados con optax.adamw
python experiments/03_lora_adapters_in_flax_nnx.py

# Paso 4 — Evaluar el modelo antes y después de LoRA con una suite determinista de validación de esquemas JSON
python experiments/04_structured_output_validation_eval.py

# Paso 5 — Fusionar matemáticamente W_merged = W_base + A @ B y levantar el endpoint HTTP POST /v1/triage
python experiments/05_merge_weights_and_serve_endpoint.py
```

### 3. Verificación completa (Linter + Tests + Outside Judge)

```bash
# Ejecutar únicamente el Outside Judge y regenerar benchmarks/latest_results.json
python verify_adaptation.py

# Ejecutar ruff, pytest (5 tests) y verify_adaptation.py en un solo comando
make all
```

### 4. Ejecución on-demand con auto-teardown en Google Cloud (`Cloud TPU v5e`, `Vertex AI L4` y `Cloud Run`)

El script `gcp/run_on_demand_adaptation.sh` opera sobre el proyecto `ai-experiments-487722` y garantiza la destrucción automática de instancias de entrenamiento mediante `trap cleanup EXIT INT TERM`:

```bash
# Ejecutar adaptación y verificación en una instancia on-demand de Cloud TPU v5e-1 (us-central1-a)
make gcp-tpu

# Ejecutar adaptación en un Vertex AI Custom Job on-demand con GPU NVIDIA L4
make gcp-gpu

# Desplegar el microservicio HTTP con pesos fusionados en Cloud Run (--min-instances=0)
make gcp-cloudrun
```
