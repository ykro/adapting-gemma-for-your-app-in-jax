PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all experiments verify test lint security gcp-tpu gcp-gpu gcp-cloudrun clean

all: lint test verify

experiments:
	$(PYTHON) experiments/01_when_prompting_hits_a_wall.py
	$(PYTHON) experiments/02_preparing_domain_dataset_masking.py
	$(PYTHON) experiments/03_lora_adapters_in_flax_nnx.py
	$(PYTHON) experiments/04_structured_output_validation_eval.py
	$(PYTHON) experiments/05_merge_weights_and_serve_endpoint.py

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

gcp-cloudrun:
	bash gcp/run_on_demand_adaptation.sh cloudrun

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache experiments/__pycache__ tests/__pycache__
