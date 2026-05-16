#!/bin/bash
# ORCD Engaging (MIT): evaluate SAI-loss LeafOnly generalization models (N=4096).
# One array task = one trained SAI model, evaluated over its relevant generalization cells.
#
# Outputs:
#   results/generalization_sweep_sai_task_XXX.csv
#
# Merge after completion:
#   python3 EvalGeneralization.py --merge --out-stem generalization_sweep_sai
#
# Submit:
#   sbatch run_eval_generalization_sai_engaging.sh
#
#SBATCH -J lo_eval_gen_sai
#SBATCH --array=0-3
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o slurm_logs/lo_eval_gen_sai_%A_%a.out
#SBATCH -e slurm_logs/lo_eval_gen_sai_%A_%a.err

set -euo pipefail

cd /home/osbo/ondemand/fluid-sim || exit 1

module load miniforge
module load cuda
source activate fluid

mkdir -p results slurm_logs

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID:-NA}, task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
echo "Started: $(date)"
echo "Eval mode: SAI-trained weights (model suffix _sai)"
echo "================================================================"

python3 -u python/EvalGeneralization.py \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --max-frames 20 \
  --model-suffix _sai \
  --out-stem generalization_sweep_sai

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
