#!/bin/bash
#SBATCH -A akr
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode071
set -euo pipefail

# ---------------------------
# Paths (override via env/CLI)
# ---------------------------
CODE_DIR="${CODE_DIR:-code}"
DATA_DIR="${DATA_DIR:-data}"
SRC_FILE="${SRC_FILE:-EUbookshop.fi}"
TGT_FILE="${TGT_FILE:-EUbookshop.en}"

# Where your training saved checkpoints
CKPT_DIR="${CKPT_DIR:-/scratch/akr/eub_relbias_20250907_095928/checkpoints}"

# Output CSV (defaults to ckpt dir if not set)
OUT_CSV="${OUT_CSV:-bleu_all_strategies.csv}"

# Split reproduction: "perm" (random-permutation) or "head" (contiguous head)
SPLIT_MODE="${SPLIT_MODE:-perm}"

# ---------------------------
# Decoding knobs
# ---------------------------
BEAM_SIZE="${BEAM_SIZE:-5}"
ALPHA="${ALPHA:-0.7}"
TOPK="${TOPK:-50}"
TEMP="${TEMP:-0.9}"
MAX_LEN="${MAX_LEN:-96}"

# ---------------------------
# Optional: auto-activate venv
# ---------------------------

source ".venv/bin/activate"

mkdir -p "${CKPT_DIR}"

# Build command
cmd=( python -u "${CODE_DIR}/eval_all.py"
      --ckpt-dir "${CKPT_DIR}"
      --data-src "${DATA_DIR}/${SRC_FILE}"
      --data-tgt "${DATA_DIR}/${TGT_FILE}"
      --split-mode "${SPLIT_MODE}"
      --beam-size "${BEAM_SIZE}"
      --alpha "${ALPHA}"
      --topk "${TOPK}"
      --temperature "${TEMP}"
      --max-len "${MAX_LEN}" )

if [[ -n "${OUT_CSV}" ]]; then
  cmd+=( --out-csv "${OUT_CSV}" )
fi

echo "===> Evaluating all checkpoints in: ${CKPT_DIR}"
echo "===> Data: ${DATA_DIR}/${SRC_FILE}  |  ${DATA_DIR}/${TGT_FILE}"
echo "===> Strategies: greedy, beam(size=${BEAM_SIZE}, alpha=${ALPHA}), top-k(k=${TOPK}, temp=${TEMP})"
echo "===> Split mode: ${SPLIT_MODE}"
echo "===> Max decode len: ${MAX_LEN}"
echo "===> Out CSV: ${OUT_CSV:-<ckpt-dir>/bleu_all_strategies.csv}"
echo

"${cmd[@]}"

echo
echo "Done. CSV is at: ${OUT_CSV:-${CKPT_DIR}/bleu_all_strategies.csv}"