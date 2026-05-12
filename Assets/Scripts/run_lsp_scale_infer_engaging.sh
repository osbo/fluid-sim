#!/bin/bash
# ORCD Engaging: inference sweep for LearningSparsePreconditioner4GPU on multiphase_v2.
#
# One array task = one scale:
#   0 -> 1024
#   1 -> 2048
#   2 -> 4096
#   3 -> 8192
#   4 -> 16384
#
# GPU request intentionally generic (no type), per cluster availability.
#
# Outputs:
# - Full infer logs:
#     Assets/Scripts/results/lsp_scale_infer_logs/multiphase_v2_<scale>.log
# - Raw infer.py CSVs:
#     Assets/Scripts/results/lsp_scale_infer_raw/infer_sweep_multiphase_v2_<scale>_*.csv
#     Assets/Scripts/results/lsp_scale_infer_raw/all_infer_sweep_multiphase_v2_<scale>_*.csv
# - Per-task normalized CSV:
#     Assets/Scripts/results/lsp_scale_infer_task_<task>.csv
#
# Merge task CSVs:
#   python3 /home/osbo/ondemand/fluid-sim/Assets/Scripts/merge_lsp_scale_infer_csvs.py
#
#SBATCH -J lsp_infer_scales
#SBATCH --array=0-4
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lsp_infer_scale_%A_%a.out
#SBATCH -e /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lsp_infer_scale_%A_%a.err

set -euo pipefail

SCALES=(1024 2048 4096 8192 16384)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#SCALES[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID (expected 0..$((${#SCALES[@]}-1)))." >&2
  exit 2
fi

SCALE="${SCALES[$TASK_ID]}"
EXP_NAME="multiphase_v2_${SCALE}"

FLUID_SIM_ROOT="/home/osbo/ondemand/fluid-sim"
LSP_REPO="/orcd/home/002/osbo/ondemand/LearningSparsePreconditioner4GPU"
RESULTS_DIR="${FLUID_SIM_ROOT}/Assets/Scripts/results"
LOG_DIR="${RESULTS_DIR}/lsp_scale_infer_logs"
RAW_OUT_DIR="${RESULTS_DIR}/lsp_scale_infer_raw"
TASK_CSV="${RESULTS_DIR}/lsp_scale_infer_task_${TASK_ID}.csv"
INFER_LOG="${LOG_DIR}/${EXP_NAME}.log"

mkdir -p "${LOG_DIR}" "${RAW_OUT_DIR}" "${FLUID_SIM_ROOT}/Assets/Scripts/slurm_logs"

module load miniforge
module load cuda
source activate fluid

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID:-NA}, task ${TASK_ID} on $(hostname)"
echo "Started: $(date)"
echo "Scale: ${SCALE}"
echo "Repo: ${LSP_REPO}"
echo "================================================================"

cd "${LSP_REPO}"

CKPT="$(
python3.12 - <<'PY' "${EXP_NAME}" "${LSP_REPO}"
import sys
from pathlib import Path

exp = sys.argv[1]
repo = Path(sys.argv[2])

# Match the training naming scheme from train.py:
#   <exp_name>-<workspace>-epoch=XX.ckpt
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
  echo "Could not find checkpoint for ${EXP_NAME} under ${LSP_REPO}; writing skipped task CSV." >&2
  python3.12 - <<'PY' "${TASK_CSV}" "${SCALE}" "${EXP_NAME}" "${INFER_LOG}"
import csv
import sys
from pathlib import Path

task_csv = Path(sys.argv[1])
scale = int(sys.argv[2])
exp_name = sys.argv[3]
infer_log = sys.argv[4]

fieldnames = [
    "scale",
    "exp_name",
    "method",
    "total_time_ms",
    "solve_time_ms",
    "precond_time_ms",
    "iterations",
    "checkpoint",
    "elapsed_sec",
    "infer_csv",
    "all_infer_csv",
    "infer_log",
    "status",
]
task_csv.parent.mkdir(parents=True, exist_ok=True)
with task_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(
        {
            "scale": scale,
            "exp_name": exp_name,
            "method": "",
            "total_time_ms": "",
            "solve_time_ms": "",
            "precond_time_ms": "",
            "iterations": "",
            "checkpoint": "",
            "elapsed_sec": "",
            "infer_csv": "",
            "all_infer_csv": "",
            "infer_log": infer_log,
            "status": "missing_checkpoint",
        }
    )
print(f"[summary] wrote skipped row to {task_csv}")
PY
  exit 0
fi

PRETRAINED_ESC="${CKPT//=/\\=}"

echo "Checkpoint: ${CKPT}"

T0="$(date +%s)"
python3.12 -u infer.py \
  exp_name="${EXP_NAME}" \
  "pretrained=${PRETRAINED_ESC}" \
  data.prefix="generated/${EXP_NAME}_test" \
  data.is_fixed_topology=true \
  data.has_shared_features=false \
  data.use_node_features=false \
  out_dir="${RAW_OUT_DIR}" \
  infer_prefix="sweep_" \
  2>&1 | tee "${INFER_LOG}"
T1="$(date +%s)"
ELAPSED_SEC="$((T1 - T0))"

python3.12 - <<'PY' "${RAW_OUT_DIR}" "${TASK_CSV}" "${SCALE}" "${EXP_NAME}" "${CKPT}" "${INFER_LOG}" "${ELAPSED_SEC}"
import csv
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1])
task_csv = Path(sys.argv[2])
scale = int(sys.argv[3])
exp_name = sys.argv[4]
ckpt = sys.argv[5]
infer_log = sys.argv[6]
elapsed_sec = int(sys.argv[7])

infer_files = sorted(
    raw_dir.glob(f"infer_sweep_{exp_name}_*.csv"),
    key=lambda p: p.stat().st_mtime
)
all_files = sorted(
    raw_dir.glob(f"all_infer_sweep_{exp_name}_*.csv"),
    key=lambda p: p.stat().st_mtime
)
if not infer_files:
    raise SystemExit(f"No infer_sweep CSV found for {exp_name} in {raw_dir}")

infer_csv = infer_files[-1]
all_csv = all_files[-1] if all_files else None

with infer_csv.open(newline="", encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    rows = list(reader)

fieldnames = [
    "scale",
    "exp_name",
    "method",
    "total_time_ms",
    "solve_time_ms",
    "precond_time_ms",
    "iterations",
    "checkpoint",
    "elapsed_sec",
    "infer_csv",
    "all_infer_csv",
    "infer_log",
    "status",
]
task_csv.parent.mkdir(parents=True, exist_ok=True)
with task_csv.open("w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(
            {
                "scale": scale,
                "exp_name": exp_name,
                "method": r.get("Key", ""),
                "total_time_ms": r.get("Total Time (ms)", ""),
                "solve_time_ms": r.get("Solve Time (ms)", ""),
                "precond_time_ms": r.get("Precond Time (ms)", ""),
                "iterations": r.get("#Iteration", ""),
                "checkpoint": ckpt,
                "elapsed_sec": elapsed_sec,
                "infer_csv": str(infer_csv),
                "all_infer_csv": str(all_csv) if all_csv else "",
                "infer_log": infer_log,
                "status": "ok",
            }
        )

print(f"[summary] wrote {task_csv}")
PY

echo "================================================================"
echo "Finished: $(date)"
echo "Elapsed: ${ELAPSED_SEC}s"
echo "Infer log: ${INFER_LOG}"
echo "Task CSV: ${TASK_CSV}"
echo "================================================================"
