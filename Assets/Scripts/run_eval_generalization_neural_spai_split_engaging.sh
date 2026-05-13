#!/bin/bash
# ORCD Engaging (MIT): evaluate split-trained NeuralSPAI models on matching generalization cells.
#
# Fair-comparison evaluation: sequentially evaluate all split-trained models in one job.
# Task mapping:
#   [0] neural_spai_g4096_id                    -> eval: g4096_eval_id
#   [1] neural_spai_g4096_topo_no_closed        -> eval: g4096_eval_id, g4096_eval_closed_only
#   [2] neural_spai_g4096_param_low             -> eval: g4096_eval_id, g4096_eval_param_high
#   [3] neural_spai_g4096_combo_no_closed_low   -> eval: g4096_eval_id, g4096_eval_closed_only,
#                                                   g4096_eval_param_high, g4096_eval_combo_closed_high
#
# Outputs:
#   - per-task CSV:
#       Assets/Scripts/results/generalization_neural_spai_split_sweep_task_<task>.csv
#   - merged CSV (run merge snippet below):
#       Assets/Scripts/results/generalization_neural_spai_split_sweep.csv
#
# Merge:
#   python3 - <<'PY'
# from pathlib import Path
# import pandas as pd
# root = Path("/home/osbo/ondemand/fluid-sim/Assets/Scripts/results")
# parts = sorted(root.glob("generalization_neural_spai_split_sweep_task_*.csv"))
# out = root / "generalization_neural_spai_split_sweep.csv"
# if parts:
#     pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(out, index=False)
#     print(out)
# else:
#     print("No task CSVs found.")
# PY
#
# Submit:
#   sbatch run_eval_generalization_neural_spai_split_engaging.sh
#
#SBATCH -J lsp_eval_gen_split
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lsp_eval_gen_split_%j.out
#SBATCH -e /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lsp_eval_gen_split_%j.err

set -euo pipefail

TASK_IDS_STR="${TASK_IDS:-0 1 2 3}"
read -r -a TASK_IDS <<< "${TASK_IDS_STR}"

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
EVAL_CELLS=(
  g4096_eval_id
  g4096_eval_id,g4096_eval_closed_only
  g4096_eval_id,g4096_eval_param_high
  g4096_eval_id,g4096_eval_closed_only,g4096_eval_param_high,g4096_eval_combo_closed_high
)

FLUID_SIM_ROOT="/home/osbo/ondemand/fluid-sim"
LSP_REPO="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU"
RESULTS_DIR="${FLUID_SIM_ROOT}/Assets/Scripts/results"
LOG_DIR="${RESULTS_DIR}/lsp_generalization_infer_logs_split"
RAW_OUT_DIR="${RESULTS_DIR}/lsp_generalization_infer_raw_split"

mkdir -p "${LOG_DIR}" "${RAW_OUT_DIR}" "${FLUID_SIM_ROOT}/Assets/Scripts/slurm_logs"

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
  EVAL_CSV="${EVAL_CELLS[$TASK_ID]}"
  TASK_CSV="${RESULTS_DIR}/generalization_neural_spai_split_sweep_task_${TASK_ID}.csv"

  echo
  echo "----------------------------------------------------------------"
  echo "TASK ${TASK_ID}: ${MODEL_NAME}"
  echo "Train split: ${TRAIN_SPLIT}"
  echo "Exp name: ${EXP_NAME}"
  echo "Eval cells: ${EVAL_CSV}"
  echo "----------------------------------------------------------------"

  CKPT="$(
python3.12 - <<'PY' "${EXP_NAME}" "${LSP_REPO}"
import sys
from pathlib import Path

exp = sys.argv[1]
repo = Path(sys.argv[2])
cands = sorted(
    set(repo.glob(f"**/{exp}-simple-epoch=*.ckpt")) | set(repo.glob(f"**/{exp}-scaled-epoch=*.ckpt")),
    key=lambda p: p.stat().st_mtime,
)
if not cands:
    print("")
else:
    print(cands[-1])
PY
)"

  if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
    echo "[warn] Missing checkpoint for exp_name=${EXP_NAME}; writing status row." >&2
    python3.12 - <<'PY' "${TASK_CSV}" "${MODEL_NAME}" "${TRAIN_SPLIT}"
import csv, sys
from pathlib import Path

task_csv = Path(sys.argv[1])
model_name = sys.argv[2]
train_split = sys.argv[3]
fieldnames = [
    "model_name", "train_split", "eval_cell", "method", "device",
    "setup_ms", "setup_ms_std", "inference_ms", "inference_ms_std",
    "solve_ms", "solve_ms_std", "iters", "iters_std",
    "total_ms", "total_ms_std", "n_frames", "checkpoint",
    "elapsed_sec", "infer_csv", "all_infer_csv", "infer_log", "status",
]
task_csv.parent.mkdir(parents=True, exist_ok=True)
with task_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(
        {
            "model_name": model_name,
            "train_split": train_split,
            "eval_cell": "",
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
            "checkpoint": "",
            "elapsed_sec": "",
            "infer_csv": "",
            "all_infer_csv": "",
            "infer_log": "",
            "status": "missing_checkpoint",
        }
    )
print(f"[summary] wrote {task_csv}")
PY
    continue
  fi

  echo "Checkpoint: ${CKPT}"
  PRETRAINED_ESC="${CKPT//=/\\=}"

  python3.12 - <<'PY' "${TASK_CSV}"
import csv, sys
from pathlib import Path
task_csv = Path(sys.argv[1])
fieldnames = [
    "model_name", "train_split", "eval_cell", "method", "device",
    "setup_ms", "setup_ms_std", "inference_ms", "inference_ms_std",
    "solve_ms", "solve_ms_std", "iters", "iters_std",
    "total_ms", "total_ms_std", "n_frames", "checkpoint",
    "elapsed_sec", "infer_csv", "all_infer_csv", "infer_log", "status",
]
task_csv.parent.mkdir(parents=True, exist_ok=True)
with task_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
PY

  IFS=',' read -r -a CELLS <<< "${EVAL_CSV}"
  for EVAL_CELL in "${CELLS[@]}"; do
    DATA_PREFIX="${FLUID_SIM_ROOT}/Assets/Scripts/data/generalization_4096/${EVAL_CELL}/test"
    CONVERTED_PREFIX="${FLUID_SIM_ROOT}/Assets/Scripts/results/lsp_generalization_generated/${EVAL_CELL}"
    INFER_LOG="${LOG_DIR}/${MODEL_NAME}__${EVAL_CELL}.log"
    RUN_EXP_NAME="generalization_4096_${MODEL_NAME}_${EVAL_CELL}"

    if [ ! -d "${DATA_PREFIX}" ]; then
      echo "[warn] Missing data for ${EVAL_CELL}: ${DATA_PREFIX}" >&2
      python3.12 - <<'PY' "${TASK_CSV}" "${MODEL_NAME}" "${TRAIN_SPLIT}" "${EVAL_CELL}" "${CKPT}" "${INFER_LOG}"
import csv, sys
from pathlib import Path
task_csv = Path(sys.argv[1]); model_name=sys.argv[2]; train_split=sys.argv[3]
eval_cell=sys.argv[4]; ckpt=sys.argv[5]; infer_log=sys.argv[6]
with task_csv.open("a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([model_name, train_split, eval_cell, "neural_spai", "gpu", "", "", "", "", "", "", "", "", "", "", "", ckpt, "", "", "", infer_log, "missing_eval_data"])
PY
      continue
    fi

    python3 -u "${FLUID_SIM_ROOT}/Assets/Scripts/convert_generalization_to_lsp_generated.py" \
      --input-dir "${DATA_PREFIX}" \
      --out-dir "${CONVERTED_PREFIX}"

    cd "${LSP_REPO}"
    T0="$(date +%s)"
    set +e
    python3.12 -u infer.py \
      exp_name="${RUN_EXP_NAME}" \
      "pretrained=${PRETRAINED_ESC}" \
      "data.prefix=${CONVERTED_PREFIX}" \
      data.is_fixed_topology=true \
      data.has_shared_features=false \
      data.use_node_features=false \
      "+dataloader=infer" \
      "+out_dir=${RAW_OUT_DIR}" \
      "+infer_prefix=generalization_split_" \
      2>&1 | tee "${INFER_LOG}"
    INFER_RC=$?
    set -e
    T1="$(date +%s)"
    ELAPSED_SEC="$((T1 - T0))"

    python3.12 - <<'PY' \
    "${RAW_OUT_DIR}" "${TASK_CSV}" "${MODEL_NAME}" "${TRAIN_SPLIT}" "${EVAL_CELL}" "${CKPT}" "${INFER_LOG}" "${ELAPSED_SEC}" "${INFER_RC}" "${RUN_EXP_NAME}"
import csv
import sys
from pathlib import Path
import pandas as pd

raw_dir = Path(sys.argv[1]); task_csv = Path(sys.argv[2]); model_name = sys.argv[3]
train_split = sys.argv[4]; eval_cell = sys.argv[5]; ckpt = sys.argv[6]
infer_log = sys.argv[7]; elapsed_sec = int(sys.argv[8]); infer_rc = int(sys.argv[9]); run_exp_name = sys.argv[10]

def append_row(status, infer_csv="", all_infer_csv="", inference_ms="", inference_ms_std="", solve_ms="", solve_ms_std="", iters="", iters_std="", total_ms="", total_ms_std="", n_frames=""):
    with task_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            model_name, train_split, eval_cell, "neural_spai", "gpu",
            "", "", inference_ms, inference_ms_std, solve_ms, solve_ms_std,
            iters, iters_std, total_ms, total_ms_std, n_frames, ckpt,
            elapsed_sec, infer_csv, all_infer_csv, infer_log, status
        ])

if infer_rc != 0:
    append_row(f"infer_failed_rc_{infer_rc}")
    raise SystemExit(0)

infer_files = sorted(raw_dir.glob(f"infer_generalization_split_{run_exp_name}_*.csv"), key=lambda p: p.stat().st_mtime)
all_files = sorted(raw_dir.glob(f"all_infer_generalization_split_{run_exp_name}_*.csv"), key=lambda p: p.stat().st_mtime)
if not infer_files or not all_files:
    append_row("missing_infer_csv")
    raise SystemExit(0)

infer_csv = infer_files[-1]
all_csv = all_files[-1]
df = pd.read_csv(all_csv)
target_key = "Neural+CUDA" if (df["Key"] == "Neural+CUDA").any() else "Neural"
sub = df[df["Key"] == target_key].copy()
if sub.empty:
    append_row("missing_neural_key", infer_csv=str(infer_csv), all_infer_csv=str(all_csv))
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

append_row(
    "ok",
    infer_csv=str(infer_csv),
    all_infer_csv=str(all_csv),
    inference_ms=prec_mu,
    inference_ms_std=prec_std,
    solve_ms=solve_mu,
    solve_ms_std=solve_std,
    iters=iter_mu,
    iters_std=iter_std,
    total_ms=total_mu,
    total_ms_std=total_std,
    n_frames=n_frames,
)
PY
  done

  echo "[done] task ${TASK_ID} -> ${TASK_CSV}"
done

echo "================================================================"
echo "Finished all requested tasks: $(date)"
echo "================================================================"
