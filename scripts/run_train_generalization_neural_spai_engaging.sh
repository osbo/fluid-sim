#!/bin/bash
# ORCD Engaging (MIT): train NeuralSPAI split-matched generalization models at N=4096.
#
# Fair-comparison training: one model per train split (same split protocol as LeafOnly).
# Array mapping:
#   [0] ID baseline train split
#   [1] topology holdout train split (no closed in train)
#   [2] parameter-range holdout train split (low contrast in train)
#   [3] compositional holdout train split (no closed + low contrast in train)
#
# Outputs:
#   - checkpoints in LearningSparsePreconditioner4GPU/mlruns/... with exp_name below
#   - task summary CSV:
#       results/lsp_generalization_train_task_<task>.csv
#
# Merge summaries:
#   python3 - <<'PY'
# from pathlib import Path
# import pandas as pd
# root = Path("/home/osbo/ondemand/fluid-sim/results")
# parts = sorted(root.glob("lsp_generalization_train_task_*.csv"))
# out = root / "lsp_generalization_train_summary.csv"
# if parts:
#     pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out, index=False)
#     print(out)
# else:
#     print("No task CSVs found.")
# PY
#
# Submit:
#   sbatch run_train_generalization_neural_spai_engaging.sh
#
#SBATCH -J lsp_train_gen4096
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o /home/osbo/ondemand/fluid-sim/slurm_logs/lsp_train_gen4096_%j.out
#SBATCH -e /home/osbo/ondemand/fluid-sim/slurm_logs/lsp_train_gen4096_%j.err

set -euo pipefail

MODEL_NAMES=(
  neural_spai_g4096_id
  neural_spai_g4096_topo_no_closed
  neural_spai_g4096_param_low
  neural_spai_g4096_combo_no_closed_low
)
TRAIN_SPLITS=(
  g4096_train_id
  g4096_train_topo_no_closed
  g4096_train_param_low
  g4096_train_combo_no_closed_low
)
EXP_NAMES=(
  generalization_4096_neural_spai_id
  generalization_4096_neural_spai_topo_no_closed
  generalization_4096_neural_spai_param_low
  generalization_4096_neural_spai_combo_no_closed_low
)

TASK_IDS_STR="${TASK_IDS:-0 1 2 3}"
read -r -a TASK_IDS <<< "${TASK_IDS_STR}"

FLUID_SIM_ROOT="/home/osbo/ondemand/fluid-sim"
LSP_REPO="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU"
RESULTS_DIR="${FLUID_SIM_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/lsp_generalization_train_logs"

mkdir -p "${LOG_DIR}" "${FLUID_SIM_ROOT}/slurm_logs"

module load miniforge
module load cuda
source activate fluid

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-NA} on $(hostname)"
echo "Started: $(date)"
echo "Sequential task ids: ${TASK_IDS[*]}"
echo "================================================================"

for TASK_ID in "${TASK_IDS[@]}"; do
  if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#MODEL_NAMES[@]}" ]; then
    echo "[skip] invalid TASK_ID=${TASK_ID} (expected 0..$((${#MODEL_NAMES[@]}-1)))" >&2
    continue
  fi

  MODEL_NAME="${MODEL_NAMES[$TASK_ID]}"
  TRAIN_SPLIT="${TRAIN_SPLITS[$TASK_ID]}"
  EXP_NAME="${EXP_NAMES[$TASK_ID]}"
  CSV_OUT="${RESULTS_DIR}/lsp_generalization_train_task_${TASK_ID}.csv"
  TRAIN_LOG="${LOG_DIR}/${MODEL_NAME}.log"
  SRC_TRAIN_DIR="${FLUID_SIM_ROOT}/data/generalization_4096/${TRAIN_SPLIT}/train"
  GEN_TRAIN_DIR="${FLUID_SIM_ROOT}/results/lsp_generalization_generated_train/${TRAIN_SPLIT}"

  echo
  echo "----------------------------------------------------------------"
  echo "TASK ${TASK_ID}: ${MODEL_NAME}"
  echo "Train split: ${TRAIN_SPLIT}"
  echo "Exp name: ${EXP_NAME}"
  echo "----------------------------------------------------------------"

  if [ ! -d "${SRC_TRAIN_DIR}" ]; then
    echo "[error] Missing source train folder: ${SRC_TRAIN_DIR}" >&2
    continue
  fi

  python3 -u "${FLUID_SIM_ROOT}/python/convert_generalization_to_lsp_generated.py" \
    --input-dir "${SRC_TRAIN_DIR}" \
    --out-dir "${GEN_TRAIN_DIR}"

  cd "${LSP_REPO}"

  T0="$(date +%s)"
  set +e
  python3.12 -u train.py \
    exp_name="${EXP_NAME}" \
    "data.prefix=${GEN_TRAIN_DIR}" \
    data.is_fixed_topology=true \
    data.has_shared_features=false \
    data.use_node_features=false \
    data.load_into_memory=true \
    check_converge=false \
    batch_size=8 \
    2>&1 | tee "${TRAIN_LOG}"
  TRAIN_RC=$?
  set -e
  T1="$(date +%s)"
  ELAPSED_SEC="$((T1 - T0))"

  python3.12 - <<'PY' "${CSV_OUT}" "${MODEL_NAME}" "${TRAIN_SPLIT}" "${EXP_NAME}" "${TRAIN_LOG}" "${ELAPSED_SEC}" "${TRAIN_RC}" "${LSP_REPO}"
import csv
import sys
from pathlib import Path

csv_out = Path(sys.argv[1])
model_name = sys.argv[2]
train_split = sys.argv[3]
exp_name = sys.argv[4]
train_log = sys.argv[5]
elapsed_sec = int(sys.argv[6])
train_rc = int(sys.argv[7])
repo = Path(sys.argv[8])

ckpts = sorted(
    set(repo.glob(f"**/{exp_name}-simple-epoch=*.ckpt")) | set(repo.glob(f"**/{exp_name}-scaled-epoch=*.ckpt")),
    key=lambda p: p.stat().st_mtime,
)
latest_ckpt = str(ckpts[-1]) if ckpts else ""

fieldnames = [
    "model_name",
    "train_split",
    "exp_name",
    "checkpoint",
    "elapsed_sec",
    "train_log",
    "status",
]
row = {
    "model_name": model_name,
    "train_split": train_split,
    "exp_name": exp_name,
    "checkpoint": latest_ckpt,
    "elapsed_sec": elapsed_sec,
    "train_log": train_log,
    "status": "ok" if train_rc == 0 else f"train_failed_rc_{train_rc}",
}

csv_out.parent.mkdir(parents=True, exist_ok=True)
with csv_out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(row)

print(f"[summary] wrote {csv_out}")
PY

  if [ "${TRAIN_RC}" -ne 0 ]; then
    echo "[warn] Training failed for ${MODEL_NAME} (rc=${TRAIN_RC}); continuing."
  fi
done

echo "================================================================"
echo "Finished all requested tasks: $(date)"
echo "================================================================"
