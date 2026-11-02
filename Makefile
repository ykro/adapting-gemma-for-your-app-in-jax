PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
OUT := benchmarks/experiment_outputs

.PHONY: all experiments verify test lint security gcp-tpu gcp-gpu clean

all: lint test verify

# Each run is also saved to benchmarks/experiment_outputs/ so every number is traceable.
experiments:
	mkdir -p $(OUT)
	$(PYTHON) experiments/01_when_prompting_hits_a_wall.py | tee $(OUT)/01_when_prompting_hits_a_wall.txt
	$(PYTHON) experiments/02_preparing_domain_dataset_masking.py | tee $(OUT)/02_preparing_domain_dataset_masking.txt
	$(PYTHON) experiments/03_lora_adapters_in_flax_nnx.py | tee $(OUT)/03_lora_adapters_in_flax_nnx.txt
	$(PYTHON) experiments/04_structured_output_validation_eval.py | tee $(OUT)/04_structured_output_validation_eval.txt
	$(PYTHON) experiments/05_merge_weights_and_serve_endpoint.py | tee $(OUT)/05_merge_weights_and_serve_endpoint.txt

verify:
	$(PYTHON) verify_adaptation.py

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check experiments/ tests/ verify_adaptation.py

security:
	gitleaks detect --source . --no-git

gcp-tpu:
	bash gcp/run_on_demand_adaptation.sh tpu

gcp-gpu:
	bash gcp/run_on_demand_adaptation.sh gpu

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache experiments/__pycache__ tests/__pycache__
