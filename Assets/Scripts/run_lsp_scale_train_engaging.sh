#!/bin/bash
# ORCD Engaging: train LearningSparsePreconditioner4GPU on multiphase_v2 scale sweep.
#
# One array task = one scale:
#   0 -> 1024
#   1 -> 2048
#   2 -> 4096
#   3 -> 8192
#   4 -> 16384
#
# Outputs:
# - Full train logs:
#     Assets/Scripts/results/lsp_scale_train_logs/multiphase_v2_<scale>.log
# - Per-task CSV summary row:
#     Assets/Scripts/results/lsp_scale_train_task_<task>.csv
#
# Merge task CSVs after all jobs finish:
#   python3 - <<'PY'
#   import csv
#   from pathlib import Path
#   root = Path("/home/osbo/ondemand/fluid-sim/Assets/Scripts/results")
#   parts = sorted(root.glob("lsp_scale_train_task_*.csv"))
#   out = root / "lsp_scale_train_summary.csv"
#   if not parts:
#       raise SystemExit("No task CSVs found.")
#   with out.open("w", newline="") as fout:
#       w = None
#       for p in parts:
#           with p.open(newline="") as fin:
#               r = csv.DictReader(fin)
#               if w is None:
#                   w = csv.DictWriter(fout, fieldnames=r.fieldnames)
#                   w.writeheader()
#               for row in r:
#                   w.writerow(row)
#   print(out)
#   PY
#
#SBATCH -J lsp_train_scales
#SBATCH --array=0-4
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lsp_train_scale_%A_%a.out
#SBATCH -e /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lsp_train_scale_%A_%a.err

set -euo pipefail

SCALES=(1024 2048 4096 8192 16384)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#SCALES[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID (expected 0..$((${#SCALES[@]}-1)))." >&2
  exit 2
fi

SCALE="${SCALES[$TASK_ID]}"
EXP_NAME="multiphase_v2_${SCALE}"

# Conservative per-scale batch sizes to avoid OOM during sweep.
case "$SCALE" in
  1024)  BATCH_SIZE=16 ;;
  2048)  BATCH_SIZE=12 ;;
  4096)  BATCH_SIZE=8  ;;
  8192)  BATCH_SIZE=8  ;;
  16384) BATCH_SIZE=4  ;;
  *)     BATCH_SIZE=8  ;;
esac

FLUID_SIM_ROOT="/home/osbo/ondemand/fluid-sim"
LSP_REPO="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU"
RESULTS_DIR="${FLUID_SIM_ROOT}/Assets/Scripts/results"
LOG_DIR="${RESULTS_DIR}/lsp_scale_train_logs"
CSV_OUT="${RESULTS_DIR}/lsp_scale_train_task_${TASK_ID}.csv"
TRAIN_LOG="${LOG_DIR}/${EXP_NAME}.log"

mkdir -p "${LOG_DIR}" "${FLUID_SIM_ROOT}/Assets/Scripts/slurm_logs"

module load miniforge
module load cuda
source activate fluid

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID:-NA}, task ${TASK_ID} on $(hostname)"
echo "Started: $(date)"
echo "Scale: ${SCALE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Repo: ${LSP_REPO}"
echo "================================================================"

cd "${LSP_REPO}"

T0="$(date +%s)"
python3.12 -u train.py \
  exp_name="${EXP_NAME}" \
  data.prefix="generated/${EXP_NAME}" \
  data.is_fixed_topology=true \
  data.has_shared_features=false \
  data.use_node_features=false \
  data.load_into_memory=true \
  batch_size="${BATCH_SIZE}" \
  2>&1 | tee "${TRAIN_LOG}"
T1="$(date +%s)"
ELAPSED_SEC="$((T1 - T0))"

python3.12 - <<'PY' "${TRAIN_LOG}" "${CSV_OUT}" "${SCALE}" "${EXP_NAME}" "${BATCH_SIZE}" "${ELAPSED_SEC}"
import csv
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
csv_out = Path(sys.argv[2])
scale = int(sys.argv[3])
exp_name = sys.argv[4]
batch_size = int(sys.argv[5])
elapsed_sec = int(sys.argv[6])

text = log_path.read_text(encoding="utf-8", errors="ignore")

def last_float(pattern: str):
    vals = re.findall(pattern, text)
    return float(vals[-1]) if vals else None

def last_int_pair(pattern: str):
    vals = re.findall(pattern, text)
    if not vals:
        return (None, None)
    a, b = vals[-1]
    return (int(a), int(b))

final_train_loss = last_float(r"Train/Loss:\s*([0-9.eE+\-]+)")
epoch_done, epoch_max = last_int_pair(r"Epoch\s+(\d+)\s*/\s*(\d+)")
loaded_samples = re.findall(r"Loaded\s+(\d+)\s+samples into memory\.", text)
loaded_samples = int(loaded_samples[-1]) if loaded_samples else None

shape = re.findall(r"matrix shape=\((\d+),\s*(\d+)\), nnz=(\d+)", text)
if shape:
    nrows, ncols, nnz = shape[-1]
    nrows, ncols, nnz = int(nrows), int(ncols), int(nnz)
else:
    nrows = ncols = nnz = None

fieldnames = [
    "scale",
    "exp_name",
    "batch_size",
    "loaded_samples",
    "matrix_rows",
    "matrix_cols",
    "matrix_nnz",
    "epoch_done",
    "epoch_max",
    "final_train_loss",
    "elapsed_sec",
    "train_log",
]
row = {
    "scale": scale,
    "exp_name": exp_name,
    "batch_size": batch_size,
    "loaded_samples": loaded_samples,
    "matrix_rows": nrows,
    "matrix_cols": ncols,
    "matrix_nnz": nnz,
    "epoch_done": epoch_done,
    "epoch_max": epoch_max,
    "final_train_loss": final_train_loss,
    "elapsed_sec": elapsed_sec,
    "train_log": str(log_path),
}

csv_out.parent.mkdir(parents=True, exist_ok=True)
with csv_out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(row)

print(f"[summary] wrote {csv_out}")
PY

echo "================================================================"
echo "Finished: $(date)"
echo "Elapsed: ${ELAPSED_SEC}s"
echo "Train log: ${TRAIN_LOG}"
echo "Summary CSV: ${CSV_OUT}"
echo "================================================================"
