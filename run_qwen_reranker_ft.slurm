#!/usr/bin/env bash
#SBATCH --job-name=inlp-qwen-ft
#SBATCH --account=ACD115049
#SBATCH --qos=contest_v100
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p outputs/slurm outputs

if command -v module >/dev/null 2>&1; then
  module load miniconda3/conda24.5.0_py3.9
fi

TRAIN_PATH="${TRAIN_PATH:-train.jsonl}"
DATA_ROOT="${DATA_ROOT:-.}"
CACHE_DIR="${CACHE_DIR:-.cache/hw3}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen_reranker_ft_mixed_probe}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Reranker-8B}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FIT_ON_FULL_DATA="${FIT_ON_FULL_DATA:-0}"
SKIP_DEV_EVAL="${SKIP_DEV_EVAL:-0}"
HARD_NEGATIVES_PER_SAMPLE="${HARD_NEGATIVES_PER_SAMPLE:-4}"
RANDOM_NEGATIVES_PER_SAMPLE="${RANDOM_NEGATIVES_PER_SAMPLE:-0}"
EPOCHS="${EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
EVAL_SPLIT="${EVAL_SPLIT:-mixed}"
HOLDOUT_DOMAIN="${HOLDOUT_DOMAIN:-}"
DOMAIN_BALANCE="${DOMAIN_BALANCE:-none}"
MAX_LENGTH="${MAX_LENGTH:-384}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-120}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
INSTRUCTION="${INSTRUCTION:-Given a document question, retrieve evidence snippets or image descriptions that support the answer.}"

if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ "${FIT_ON_FULL_DATA}" == "1" ]]; then
  EXTRA_ARGS+=" --fit-on-full-data"
fi
if [[ "${SKIP_DEV_EVAL}" == "1" ]]; then
  EXTRA_ARGS+=" --skip-dev-eval"
fi
EXTRA_ARGS+=" --hard-negatives-per-sample ${HARD_NEGATIVES_PER_SAMPLE}"
EXTRA_ARGS+=" --random-negatives-per-sample ${RANDOM_NEGATIVES_PER_SAMPLE}"
EXTRA_ARGS+=" --epochs ${EPOCHS}"
EXTRA_ARGS+=" --train-batch-size ${TRAIN_BATCH_SIZE}"
EXTRA_ARGS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
EXTRA_ARGS+=" --grad-accum-steps ${GRAD_ACCUM_STEPS}"
EXTRA_ARGS+=" --eval-split ${EVAL_SPLIT}"
EXTRA_ARGS+=" --domain-balance ${DOMAIN_BALANCE}"
EXTRA_ARGS+=" --max-length ${MAX_LENGTH}"
EXTRA_ARGS+=" --max-train-steps ${MAX_TRAIN_STEPS}"
EXTRA_ARGS+=" --learning-rate ${LEARNING_RATE}"
EXTRA_ARGS+=" --lora-r ${LORA_R}"
EXTRA_ARGS+=" --lora-alpha ${LORA_ALPHA}"
EXTRA_ARGS+=" --lora-dropout ${LORA_DROPOUT}"
if [[ -n "${HOLDOUT_DOMAIN}" ]]; then
  EXTRA_ARGS+=" --holdout-domain ${HOLDOUT_DOMAIN}"
fi

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Python: $(${PYTHON_BIN} --version 2>&1)"
echo "Base model: ${BASE_MODEL}"
echo "Output dir: ${OUTPUT_DIR}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi

"${PYTHON_BIN}" train_qwen_reranker.py \
  --train-path "${TRAIN_PATH}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-model "${BASE_MODEL}" \
  --device "${DEVICE}" \
  --instruction "${INSTRUCTION}" \
  ${EXTRA_ARGS}

echo "Job finished at $(date)"
