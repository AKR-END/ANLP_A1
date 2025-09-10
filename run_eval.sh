#!/bin/bash
#SBATCH -A akr
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END
#SBATCH -w gnode085
set -euo pipefail

# ---------------------------
# Paths (override via env/CLI)
# ---------------------------
CODE_DIR="${CODE_DIR:-code}"
DATA_DIR="${DATA_DIR:-data}"
SRC_FILE="${SRC_FILE:-EUbookshop.fi}"
TGT_FILE="${TGT_FILE:-EUbookshop.en}"

# Path to specific checkpoint file
CKPT_PATH="${CKPT_PATH:-/scratch/akr/rope_epoch010_train2.4517_val3.6886.pt}"

# Directory where CSV output will be saved
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/akr}"

# Output CSV (defaults to checkpoint-specific name if not set)
OUT_CSV="${OUT_CSV:-}"

# Split reproduction: "perm" (random-permutation) or "head" (contiguous head)
SPLIT_MODE="${SPLIT_MODE:-perm}"

# ---------------------------
# Decoding knobs
# ---------------------------
BEAM_SIZE="${BEAM_SIZE:-5}"
ALPHA="${ALPHA:-0.7}"
TOPK="${TOPK:-50}"
TEMP="${TEMP:-0.9}"
MAX_LEN="${MAX_LEN:-128}"

# Limit and progress UI (optional)
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
PROGRESS="${PROGRESS:-1}"

# SentencePiece models (optional). If both are provided, SPM will be used.
SRC_SPM_PROTO="${SRC_SPM_PROTO:-/scratch/akr/src_spm.model}"
TGT_SPM_PROTO="${TGT_SPM_PROTO:-/scratch/akr/tgt_spm.model}"

# ---------------------------
# Optional: auto-activate venv
# ---------------------------

source ".venv/bin/activate"

mkdir -p "${OUTPUT_DIR}"

# Build command
cmd=( python -u "${CODE_DIR}/test.py"
      --ckpt-path "${CKPT_PATH}"
      --data-src "${DATA_DIR}/${SRC_FILE}"
      --data-tgt "${DATA_DIR}/${TGT_FILE}"
      --split-mode "${SPLIT_MODE}"
      --beam-size "${BEAM_SIZE}"
      --alpha "${ALPHA}"
      --topk "${TOPK}"
      --temperature "${TEMP}"
      --max-len "${MAX_LEN}" 
      --samples 3
      --sample-mode random
      --sample-seed 42)

# Optional flags
# if [[ -n "${MAX_EXAMPLES}" ]]; then
#   cmd+=( --max-examples "${MAX_EXAMPLES}" )
# fi
if [[ -n "${PROGRESS}" ]]; then
  cmd+=( --progress )
fi
if [[ -n "${SRC_SPM_PROTO}" && -n "${TGT_SPM_PROTO}" ]]; then
  cmd+=( --src-spm-proto "${SRC_SPM_PROTO}" --tgt-spm-proto "${TGT_SPM_PROTO}" )
fi

if [[ -n "${OUT_CSV}" ]]; then
  cmd+=( --out-csv "${OUT_CSV}" )
else
  # Generate default CSV name based on checkpoint
  checkpoint_name=$(basename "${CKPT_PATH}" .pt)
  cmd+=( --out-csv "${OUTPUT_DIR}/${checkpoint_name}_bleu.csv" )
fi

echo "===> Evaluating checkpoint: ${CKPT_PATH}"
echo "===> Data: ${DATA_DIR}/${SRC_FILE}  |  ${DATA_DIR}/${TGT_FILE}"
echo "===> Strategies: greedy, beam(size=${BEAM_SIZE}, alpha=${ALPHA}), top-k(k=${TOPK}, temp=${TEMP})"
echo "===> Split mode: ${SPLIT_MODE}"
echo "===> Max decode len: ${MAX_LEN}"
# if [[ -n "${MAX_EXAMPLES}" ]]; then
#   echo "===> Max examples: ${MAX_EXAMPLES}"
# fi
if [[ -n "${PROGRESS}" ]]; then
  echo "===> Progress bar: enabled"
fi
if [[ -n "${SRC_SPM_PROTO}" && -n "${TGT_SPM_PROTO}" ]]; then
  echo "===> Tokenizer: SentencePiece"
  echo "===> SPM src: ${SRC_SPM_PROTO}"
  echo "===> SPM tgt: ${TGT_SPM_PROTO}"
fi
if [[ -n "${OUT_CSV}" ]]; then
  echo "===> Out CSV: ${OUT_CSV}"
else
  checkpoint_name=$(basename "${CKPT_PATH}" .pt)
  echo "===> Out CSV: ${OUTPUT_DIR}/${checkpoint_name}_bleu.csv"
fi
echo

"${cmd[@]}"

echo
if [[ -n "${OUT_CSV}" ]]; then
  echo "Done. CSV is at: ${OUT_CSV}"
else
  checkpoint_name=$(basename "${CKPT_PATH}" .pt)
  echo "Done. CSV is at: ${OUTPUT_DIR}/${checkpoint_name}_bleu.csv"
fi