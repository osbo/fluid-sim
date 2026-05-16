#!/bin/bash
# ORCD Engaging (MIT): evaluate NeuralSPAI checkpoint on generalization_4096 eval cells.
# One array task = one eval cell.
#
# Array mapping:
#   [0] g4096_eval_id
#   [1] g4096_eval_closed_only
#   [2] g4096_eval_param_high
#   [3] g4096_eval_combo_closed_high
#
# Outputs:
#   results/generalization_neural_spai_sweep_task_<task>.csv
#   results/lsp_generalization_infer_logs/*.log
#   results/lsp_generalization_infer_raw/*generalization*.csv
#
# Merge task CSVs after completion:
#   python3 - <<'PY'
# from pathlib import Path
# import pandas as pd
# root = Path("/home/osbo/ondemand/fluid-sim/results")
# parts = sorted(root.glob("generalization_neural_spai_sweep_task_*.csv"))
# if parts:
#     pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
#         root / "generalization_neural_spai_sweep.csv", index=False
#     )
#     print(root / "generalization_neural_spai_sweep.csv")
# else:
#     print("No task CSVs found.")
# PY
#
# Submit:
#   sbatch run_eval_generalization_neural_spai_engaging.sh
#
#SBATCH -J lsp_eval_gen4096
#SBATCH --array=0-3
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o /home/osbo/ondemand/fluid-sim/slurm_logs/lsp_eval_gen4096_%A_%a.out
#SBATCH -e /home/osbo/ondemand/fluid-sim/slurm_logs/lsp_eval_gen4096_%A_%a.err

set -euo pipefail

EVAL_CELLS=(
  g4096_eval_id
  g4096_eval_closed_only
  g4096_eval_param_high
  g4096_eval_combo_closed_high
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#EVAL_CELLS[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID (expected 0..$((${#EVAL_CELLS[@]}-1)))." >&2
  exit 2
fi

EVAL_CELL="${EVAL_CELLS[$TASK_ID]}"

FLUID_SIM_ROOT="/home/osbo/ondemand/fluid-sim"
LSP_REPO="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU"
RESULTS_DIR="${FLUID_SIM_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/lsp_generalization_infer_logs"
RAW_OUT_DIR="${RESULTS_DIR}/lsp_generalization_infer_raw"
TASK_CSV="${RESULTS_DIR}/generalization_neural_spai_sweep_task_${TASK_ID}.csv"
INFER_LOG="${LOG_DIR}/${EVAL_CELL}.log"

DATA_PREFIX="${FLUID_SIM_ROOT}/data/generalization_4096/${EVAL_CELL}/test"
CONVERTED_PREFIX="${FLUID_SIM_ROOT}/results/lsp_generalization_generated/${EVAL_CELL}"
EXP_NAME="generalization_4096_${EVAL_CELL}"
BASE_EXP_FOR_CKPT="multiphase_v2_4096"

mkdir -p "${LOG_DIR}" "${RAW_OUT_DIR}" "${FLUID_SIM_ROOT}/slurm_logs"

module load miniforge
module load cuda
source activate fluid

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID:-NA}, task ${TASK_ID} on $(hostname)"
echo "Started: $(date)"
echo "Eval cell: ${EVAL_CELL}"
echo "Data: ${DATA_PREFIX}"
echo "================================================================"

if [ ! -d "${DATA_PREFIX}" ]; then
  echo "Missing data folder: ${DATA_PREFIX}" >&2
  exit 2
fi

python3 -u "${FLUID_SIM_ROOT}/python/convert_generalization_to_lsp_generated.py" \
  --input-dir "${DATA_PREFIX}" \
  --out-dir "${CONVERTED_PREFIX}"

cd "${LSP_REPO}"

CKPT="$(
python3.12 - <<'PY' "${BASE_EXP_FOR_CKPT}" "${LSP_REPO}"
import sys
from pathlib import Path

exp = sys.argv[1]
repo = Path(sys.argv[2])
patterns = [
    f"**/{exp}-simple-epoch=*.ckpt",
    f"**/{exp}-scaled-epoch=*.ckpt",
    f"**/{exp}-*-epoch=*.ckpt",
]
cands = []
for pat in patterns:
    cands.extend(repo.glob(pat))
if not cands:
    raise SystemExit("")
cands = sorted(set(cands), key=lambda p: p.stat().st_mtime)
print(cands[-1])
PY
)"

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "Could not find checkpoint for ${BASE_EXP_FOR_CKPT} under ${LSP_REPO}" >&2
  exit 2
fi

PRETRAINED_ESC="${CKPT//=/\\=}"
echo "Checkpoint: ${CKPT}"

T0="$(date +%s)"
set +e
python3.12 -u infer.py \
  exp_name="${EXP_NAME}" \
  "pretrained=${PRETRAINED_ESC}" \
  "data.prefix=${CONVERTED_PREFIX}" \
  data.is_fixed_topology=true \
  data.has_shared_features=false \
  data.use_node_features=false \
  "+dataloader=infer" \
  "+out_dir=${RAW_OUT_DIR}" \
  "+infer_prefix=generalization_" \
  2>&1 | tee "${INFER_LOG}"
INFER_RC=$?
set -e
T1="$(date +%s)"
ELAPSED_SEC="$((T1 - T0))"

python3.12 - <<'PY' \
  "${RAW_OUT_DIR}" "${TASK_CSV}" "${EVAL_CELL}" "${CKPT}" "${INFER_LOG}" "${ELAPSED_SEC}" "${INFER_RC}" "${EXP_NAME}"
import csv
import math
import sys
from pathlib import Path
import pandas as pd

raw_dir = Path(sys.argv[1])
task_csv = Path(sys.argv[2])
eval_cell = sys.argv[3]
ckpt = sys.argv[4]
infer_log = sys.argv[5]
elapsed_sec = int(sys.argv[6])
infer_rc = int(sys.argv[7])
exp_name = sys.argv[8]

fieldnames = [
    "model_name",
    "train_split",
    "eval_cell",
    "method",
    "device",
    "setup_ms",
    "setup_ms_std",
    "inference_ms",
    "inference_ms_std",
    "solve_ms",
    "solve_ms_std",
    "iters",
    "iters_std",
    "total_ms",
    "total_ms_std",
    "n_frames",
    "checkpoint",
    "elapsed_sec",
    "infer_csv",
    "all_infer_csv",
    "infer_log",
    "status",
]

def write_row(row):
    task_csv.parent.mkdir(parents=True, exist_ok=True)
    with task_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(row)

if infer_rc != 0:
    write_row(
        {
            "model_name": "neural_spai_4096",
            "train_split": "multiphase_v2_4096",
            "eval_cell": eval_cell,
            "method": "neural_spai",
            "device": "gpu",
            "setup_ms": "",
            "setup_ms_std": "",
            "inference_ms": "",
            "inference_ms_std": "",
            "solve_ms": "",
            "solve_ms_std": "",
            "iters": "",
            "iters_std": "",
            "total_ms": "",
            "total_ms_std": "",
            "n_frames": "",
            "checkpoint": ckpt,
            "elapsed_sec": elapsed_sec,
            "infer_csv": "",
            "all_infer_csv": "",
            "infer_log": infer_log,
            "status": f"infer_failed_rc_{infer_rc}",
        }
    )
    print(f"[summary] wrote failed row to {task_csv}")
    raise SystemExit(0)

infer_files = sorted(raw_dir.glob(f"infer_generalization_{exp_name}_*.csv"), key=lambda p: p.stat().st_mtime)
all_files = sorted(raw_dir.glob(f"all_infer_generalization_{exp_name}_*.csv"), key=lambda p: p.stat().st_mtime)
if not infer_files or not all_files:
    write_row(
        {
            "model_name": "neural_spai_4096",
            "train_split": "multiphase_v2_4096",
            "eval_cell": eval_cell,
            "method": "neural_spai",
            "device": "gpu",
            "setup_ms": "",
            "setup_ms_std": "",
            "inference_ms": "",
            "inference_ms_std": "",
            "solve_ms": "",
            "solve_ms_std": "",
            "iters": "",
            "iters_std": "",
            "total_ms": "",
            "total_ms_std": "",
            "n_frames": "",
            "checkpoint": ckpt,
            "elapsed_sec": elapsed_sec,
            "infer_csv": "",
            "all_infer_csv": "",
            "infer_log": infer_log,
            "status": "missing_infer_csv",
        }
    )
    print(f"[summary] wrote missing-csv row to {task_csv}")
    raise SystemExit(0)

infer_csv = infer_files[-1]
all_csv = all_files[-1]
df = pd.read_csv(all_csv)

target_key = "Neural+CUDA" if (df["Key"] == "Neural+CUDA").any() else "Neural"
sub = df[df["Key"] == target_key].copy()
if sub.empty:
    write_row(
        {
            "model_name": "neural_spai_4096",
            "train_split": "multiphase_v2_4096",
            "eval_cell": eval_cell,
            "method": "neural_spai",
            "device": "gpu",
            "setup_ms": "",
            "setup_ms_std": "",
            "inference_ms": "",
            "inference_ms_std": "",
            "solve_ms": "",
            "solve_ms_std": "",
            "iters": "",
            "iters_std": "",
            "total_ms": "",
            "total_ms_std": "",
            "n_frames": "",
            "checkpoint": ckpt,
            "elapsed_sec": elapsed_sec,
            "infer_csv": str(infer_csv),
            "all_infer_csv": str(all_csv),
            "infer_log": infer_log,
            "status": "missing_neural_key",
        }
    )
    print(f"[summary] wrote missing-key row to {task_csv}")
    raise SystemExit(0)

solve = pd.to_numeric(sub["Solve Time (ms)"], errors="coerce")
prec = pd.to_numeric(sub["Precond Time (ms)"], errors="coerce")
iters = pd.to_numeric(sub["#Iteration"], errors="coerce")
total = solve + prec

def mu_std(series):
    x = series.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return ("", "")
    mu = float(x.mean())
    if x.size < 2:
        return (f"{mu:.4f}", "")
    std = float(x.std(ddof=1))
    return (f"{mu:.4f}", f"{std:.4f}")

total_mu, total_std = mu_std(total)
solve_mu, solve_std = mu_std(solve)
prec_mu, prec_std = mu_std(prec)
iter_mu, iter_std = mu_std(iters)
n_frames = int(max(total.dropna().size, solve.dropna().size, prec.dropna().size, iters.dropna().size))

write_row(
    {
        "model_name": "neural_spai_4096",
        "train_split": "multiphase_v2_4096",
        "eval_cell": eval_cell,
        "method": "neural_spai",
        "device": "gpu",
        "setup_ms": "",
        "setup_ms_std": "",
        "inference_ms": prec_mu,
        "inference_ms_std": prec_std,
        "solve_ms": solve_mu,
        "solve_ms_std": solve_std,
        "iters": iter_mu,
        "iters_std": iter_std,
        "total_ms": total_mu,
        "total_ms_std": total_std,
        "n_frames": n_frames,
        "checkpoint": ckpt,
        "elapsed_sec": elapsed_sec,
        "infer_csv": str(infer_csv),
        "all_infer_csv": str(all_csv),
        "infer_log": infer_log,
        "status": "ok",
    }
)
print(f"[summary] wrote {task_csv}")
PY

echo "================================================================"
echo "Finished: $(date)"
echo "Elapsed: ${ELAPSED_SEC}s"
echo "Log: ${INFER_LOG}"
echo "Task CSV: ${TASK_CSV}"
echo "================================================================"
