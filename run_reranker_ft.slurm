#!/usr/bin/env bash
#SBATCH --job-name=inlp-reranker-ft
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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reranker_ft_m3_mixed}"
BASE_MODEL="${BASE_MODEL:-BAAI/bge-reranker-v2-m3}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
FIT_ON_FULL_DATA="${FIT_ON_FULL_DATA:-0}"
SKIP_DEV_EVAL="${SKIP_DEV_EVAL:-0}"
TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"
HARD_NEGATIVES_PER_SAMPLE="${HARD_NEGATIVES_PER_SAMPLE:-6}"
RANDOM_NEGATIVES_PER_SAMPLE="${RANDOM_NEGATIVES_PER_SAMPLE:-2}"
EPOCHS="${EPOCHS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
EVAL_SPLIT="${EVAL_SPLIT:-}"
HOLDOUT_DOMAIN="${HOLDOUT_DOMAIN:-}"
DOMAIN_BALANCE="${DOMAIN_BALANCE:-}"
MAX_LENGTH="${MAX_LENGTH:-}"

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
EXTRA_ARGS+=" --train-dtype ${TRAIN_DTYPE}"
EXTRA_ARGS+=" --hard-negatives-per-sample ${HARD_NEGATIVES_PER_SAMPLE}"
EXTRA_ARGS+=" --random-negatives-per-sample ${RANDOM_NEGATIVES_PER_SAMPLE}"
if [[ -n "${EPOCHS}" ]]; then
  EXTRA_ARGS+=" --epochs ${EPOCHS}"
fi
if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  EXTRA_ARGS+=" --train-batch-size ${TRAIN_BATCH_SIZE}"
fi
if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
  EXTRA_ARGS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
fi
if [[ -n "${GRAD_ACCUM_STEPS}" ]]; then
  EXTRA_ARGS+=" --grad-accum-steps ${GRAD_ACCUM_STEPS}"
fi
if [[ -n "${EVAL_SPLIT}" ]]; then
  EXTRA_ARGS+=" --eval-split ${EVAL_SPLIT}"
fi
if [[ -n "${HOLDOUT_DOMAIN}" ]]; then
  EXTRA_ARGS+=" --holdout-domain ${HOLDOUT_DOMAIN}"
fi
if [[ -n "${DOMAIN_BALANCE}" ]]; then
  EXTRA_ARGS+=" --domain-balance ${DOMAIN_BALANCE}"
fi
if [[ -n "${MAX_LENGTH}" ]]; then
  EXTRA_ARGS+=" --max-length ${MAX_LENGTH}"
fi

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Python: $(${PYTHON_BIN} --version 2>&1)"
echo "Base model: ${BASE_MODEL}"
echo "Output dir: ${OUTPUT_DIR}"

"${PYTHON_BIN}" train_reranker.py \
  --train-path "${TRAIN_PATH}" \
  --data-root "${DATA_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-model "${BASE_MODEL}" \
  --device "${DEVICE}" \
  ${EXTRA_ARGS}

echo "Job finished at $(date)"
