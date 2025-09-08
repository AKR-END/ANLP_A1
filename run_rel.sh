#!/bin/bash
#SBATCH -A akr
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode087
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-code}"
DATA_DIR="${DATA_DIR:-data}"
SRC_FILE="${SRC_FILE:-EUbookshop.fi}"
TGT_FILE="${TGT_FILE:-EUbookshop.en}"

EXP_NAME="${EXP_NAME:-eub_relbias_$(date +%Y%m%d_%H%M%S)}"
EXP_DIR="${EXP_DIR:-/scratch/akr/${EXP_NAME}}"
CKPT_DIR="${CKPT_DIR:-${EXP_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

POSENC="${POSENC:-relbias}"       # relbias
DMODEL="${DMODEL:-512}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-8}"
DFF="${DFF:-2048}"
DROPOUT="${DROPOUT:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
EPOCHS="${EPOCHS:-20}"
VAL_RATIO="${VAL_RATIO:-0.05}"
TEST_RATIO="${TEST_RATIO:-0.05}"
PATIENCE="${PATIENCE:-5}"
MIN_DELTA="${MIN_DELTA:-0.0005}"
SEED="${SEED:-42}"

LOG_CSV="${LOG_CSV:-loss_log.csv}"
PLOT_PNG="${PLOT_PNG:-loss_curve.png}"

MAX_LEN="${MAX_LEN:-96}"
BEAM_SIZE="${BEAM_SIZE:-5}"
ALPHA="${ALPHA:-0.7}"
TOPK="${TOPK:-50}"
TEMP="${TEMP:-0.9}"
TEST_SENT="${TEST_SENT:-miten voit}"

source .venv/bin/activate
echo "===> Training (${POSENC}) -> ${CKPT_DIR}"
python "${CODE_DIR}/train.py" \
  --data-src "${DATA_DIR}/${SRC_FILE}" \
  --data-tgt "${DATA_DIR}/${TGT_FILE}" \
  --val-ratio "${VAL_RATIO}" --test-ratio "${TEST_RATIO}" \
  --posenc "${POSENC}" \
  --d-model "${DMODEL}" --layers "${LAYERS}" --heads "${HEADS}" --d-ff "${DFF}" \
  --dropout "${DROPOUT}" --batch-size "${BATCH_SIZE}" --lr "${LR}" --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" --min-delta "${MIN_DELTA}" --seed "${SEED}" \
  --save-dir "${CKPT_DIR}" --log-csv "${LOG_CSV}" --plot-png "${PLOT_PNG}" \
  --tokenizer spm --max-len "${MAX_LEN}" \
  --spm-size-src 8000 --spm-size-tgt 8000 --spm-model-type bpe --spm-character-coverage 1.0

cp -f "${CKPT_DIR}/${LOG_CSV}" "${LOG_DIR}/${LOG_CSV}"
cp -f "${CKPT_DIR}/${PLOT_PNG}" "${LOG_DIR}/${PLOT_PNG}"

BEST="${CKPT_DIR}/best.pt"
test -f "${BEST}" || { echo "best.pt not found in ${CKPT_DIR}"; exit 2; }

# Evaluate BLEU for greedy/beam/topk and write CSV
echo "===> Evaluating BLEU (all strategies)"
python "${CODE_DIR}/eval_all.py" \
  --ckpt-dir "${CKPT_DIR}" \
  --data-src "${DATA_DIR}/${SRC_FILE}" \
  --data-tgt "${DATA_DIR}/${TGT_FILE}"

echo "Done. Checkpoints in: ${CKPT_DIR}"
echo "Loss CSV & plot in:  ${LOG_DIR}"