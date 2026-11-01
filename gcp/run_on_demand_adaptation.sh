#!/usr/bin/env bash
# On-demand Cloud TPU (v5litepod-1), Vertex AI GPU (NVIDIA L4), and Cloud Run (--min-instances=0)
# runner with auto-teardown for adapting-gemma-for-your-app-in-jax.
# Project: ai-experiments-487722
# Guarantees zero orphaned cloud resources via EXIT trap.

set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"
PROJECT_ID="${GCP_PROJECT_ID:-ai-experiments-487722}"
REGION="${GCP_REGION:-us-central1}"
ZONE="${GCP_ZONE:-us-central1-a}"
MODE="${1:-tpu}"
TPU_NAME="gemma-lora-app-$(date +%s)"
GPU_JOB_ID=""
SERVICE_NAME="gemma-incident-triage-jax"

cleanup() {
  if [[ "${MODE}" == "tpu" ]]; then
    echo "[auto-teardown] Deleting on-demand TPU VM ${TPU_NAME} (if present)..."
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --quiet >/dev/null 2>&1 || true
  elif [[ "${MODE}" == "gpu" && -n "${GPU_JOB_ID}" ]]; then
    echo "[auto-teardown] Ensuring Vertex AI Custom Job ${GPU_JOB_ID} is cancelled if still active..."
    gcloud ai custom-jobs cancel "${GPU_JOB_ID}" \
      --project="${PROJECT_ID}" \
      --region="${REGION}" \
      --quiet >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo " Adapting Gemma for Your App in JAX — On-Demand GCP Runner"
echo " Project : ${PROJECT_ID}"
echo " Mode    : ${MODE}"
echo "============================================================"

if [[ "${MODE}" == "tpu" ]]; then
  echo "[1/3] Provisioning on-demand Cloud TPU v5e-1 (v5litepod-1) in ${ZONE}..."
  gcloud compute tpus tpu-vm create "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --accelerator-type="v5litepod-1" \
    --version="v2-alpha-tpuv5-lite" \
    --spot

  echo "[2/3] Copying adaptation suite and installing dependencies..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="mkdir -p ~/gemma-app-bench"
  gcloud compute tpus tpu-vm scp --recurse \
    experiments verify_adaptation.py pyproject.toml \
    "${TPU_NAME}:~/gemma-app-bench/" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}"

  echo "[3/3] Executing verify_adaptation.py on Cloud TPU v5e..."
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="pip install -q 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax optax numpy && cd ~/gemma-app-bench && python3 verify_adaptation.py"

elif [[ "${MODE}" == "gpu" ]]; then
  echo "[1/2] Submitting on-demand Vertex AI Custom Job on NVIDIA L4 GPU (g2-standard-4)..."
  BUNDLE_B64="$(tar czf - experiments verify_adaptation.py pyproject.toml | base64 | tr -d '\n')"
  GPU_JOB_ID="$(gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="adapting-gemma-app-l4-$(date +%s)" \
    --worker-pool-spec="machine-type=g2-standard-4,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest" \
    --command="bash,-c,mkdir -p /tmp/bench && cd /tmp/bench && echo ${BUNDLE_B64} | base64 -d | tar xzf - && pip install -q 'jax[cuda12]' flax optax numpy && python3 verify_adaptation.py" \
    --format="value(name)")"
  echo "[2/2] Streaming logs for ${GPU_JOB_ID}..."
  gcloud ai custom-jobs stream-logs "${GPU_JOB_ID}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}"

elif [[ "${MODE}" == "cloudrun" ]]; then
  echo "[1/1] Deploying merged Gemma JSON triage endpoint to Cloud Run with --min-instances=0..."
  gcloud run deploy "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --source="." \
    --min-instances=0 \
    --max-instances=2 \
    --cpu=2 \
    --memory=2Gi \
    --no-allow-unauthenticated
else
  echo "Usage: $0 [tpu|gpu|cloudrun]"
  exit 1
fi
