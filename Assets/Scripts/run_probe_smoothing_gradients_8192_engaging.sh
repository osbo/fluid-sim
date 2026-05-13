#!/bin/bash
# ORCD Engaging (MIT): compare probe smoothing effects on gradient balance at N=8192.
#
# Runs two baseline trainings (d128, L3, highways):
#   1) Default probe smoothing (2 Jacobi steps, omega=0.6)
#   2) No probe smoothing (0 Jacobi steps)
# Each run enables:
#   - --grad-mags      (prints per-group gradient magnitudes every 100 steps)
#   - --track-training (writes training profile CSV + step checkpoints every 2000 steps)
#
# Outputs:
#   logs:     Assets/Scripts/results/probe_smoothing_8192/*.log
#   weights:  Assets/Scripts/weights/*smooth*.bytes
#   profiles: Assets/Scripts/weights/*smooth*_training_profile.csv
#   plot/csv: Assets/Scripts/results/probe_smoothing_8192/gradient_balance_8192.*
#
# Submit:
#   sbatch run_probe_smoothing_gradients_8192_engaging.sh
#
#SBATCH -J lo_probe_grad_8192
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lo_probe_grad_8192_%j.out
#SBATCH -e /home/osbo/ondemand/fluid-sim/Assets/Scripts/slurm_logs/lo_probe_grad_8192_%j.err

set -euo pipefail

ROOT="/home/osbo/ondemand/fluid-sim/Assets/Scripts"
RESULTS_DIR="${ROOT}/results/probe_smoothing_8192"
WEIGHTS_DIR="${ROOT}/weights"
BASE_LR="${BASE_LR:-2e-4}"
# Default retry mode: rerun only smooth_zero at half the original LR.
ZERO_RETRY_LR="${ZERO_RETRY_LR:-1e-4}"
RUN_DEFAULT="${RUN_DEFAULT:-0}"
RUN_ZERO="${RUN_ZERO:-1}"

mkdir -p "${RESULTS_DIR}" "${WEIGHTS_DIR}" "${ROOT}/slurm_logs"

cd "${ROOT}"

module load miniforge
module load cuda
source activate fluid

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-NA} on $(hostname)"
echo "Started: $(date)"
echo "================================================================"

run_case () {
  local case_name="$1"
  local jacobi_steps="$2"
  local jacobi_omega="$3"
  local lr="$4"
  local weights_out="${WEIGHTS_DIR}/v2_8192_d128_L3_hw_${case_name}.bytes"
  local train_log="${RESULTS_DIR}/train_${case_name}.log"

  echo
  echo "----------------------------------------------------------------"
  echo "Case: ${case_name}"
  echo "probe_jacobi_steps=${jacobi_steps}, probe_jacobi_omega=${jacobi_omega}"
  echo "lr=${lr}"
  echo "weights_out=${weights_out}"
  echo "log=${train_log}"
  echo "----------------------------------------------------------------"

  python3 -u LeafOnly.py \
    --data-folder "${ROOT}/data/multiphase_v2_8192/train" \
    --leaf-size 128 \
    --max-mixed-size 8192 \
    --d-model 128 \
    --num-layers 3 \
    --num-heads 8 \
    --lr "${lr}" \
    --steps 100000 \
    --weights-out "${weights_out}" \
    --track-training \
    --auto-stop \
    --grad-mags \
    --probe-jacobi-steps "${jacobi_steps}" \
    --probe-jacobi-omega "${jacobi_omega}" \
    2>&1 | tee "${train_log}"
}

if [[ "${RUN_DEFAULT}" == "1" ]]; then
  run_case "smooth_default" 2 0.6 "${BASE_LR}"
else
  echo "Skipping smooth_default (RUN_DEFAULT=${RUN_DEFAULT})"
fi

if [[ "${RUN_ZERO}" == "1" ]]; then
  run_case "smooth_zero" 0 0.6 "${ZERO_RETRY_LR}"
else
  echo "Skipping smooth_zero (RUN_ZERO=${RUN_ZERO})"
fi

python3 -u "${ROOT}/plot_probe_smoothing_grad_balance.py" \
  --default-log "${RESULTS_DIR}/train_smooth_default.log" \
  --zero-log "${RESULTS_DIR}/train_smooth_zero.log" \
  --out-png "${RESULTS_DIR}/gradient_balance_8192.png" \
  --out-summary-csv "${RESULTS_DIR}/gradient_balance_8192_summary.csv"

echo "================================================================"
echo "Finished: $(date)"
echo "Results dir: ${RESULTS_DIR}"
echo "================================================================"
