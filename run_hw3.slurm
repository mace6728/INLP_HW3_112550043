#!/usr/bin/env bash
#SBATCH --job-name=inlp-hw3
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

MODE="${MODE:-submit}"
METHOD="${METHOD:-hybrid}"
TRAIN_PATH="${TRAIN_PATH:-train.jsonl}"
TEST_PATH="${TEST_PATH:-test.jsonl}"
DATA_ROOT="${DATA_ROOT:-.}"
OUTPUT="${OUTPUT:-outputs/submission.csv}"
CACHE_DIR="${CACHE_DIR:-.cache/hw3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Account: acd115049"
echo "QOS: contest_v100"
echo "Partition: nycugpu_queue"
echo "Mode: ${MODE}"
echo "Method: ${METHOD}"
echo "Python: $(${PYTHON_BIN} --version 2>&1)"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 7):
    raise SystemExit("HW3_112550043.py requires Python 3.7 or newer")
PY

"${PYTHON_BIN}" HW3_112550043.py \
  --mode "${MODE}" \
  --method "${METHOD}" \
  --train-path "${TRAIN_PATH}" \
  --test-path "${TEST_PATH}" \
  --data-root "${DATA_ROOT}" \
  --output "${OUTPUT}" \
  --cache-dir "${CACHE_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  ${EXTRA_ARGS}

echo "Job finished at $(date)"
