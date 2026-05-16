#!/bin/bash
# ORCD Engaging (MIT): train SAI-loss LeafOnly generalization models (N=4096).
# One array task = one train split, matching the existing generalization pattern.
#
# Array mapping:
#   [0] ID baseline
#   [1] topology holdout (no closed in train)
#   [2] parameter-range holdout (train low contrast)
#   [3] compositional holdout (no closed + low contrast in train)
#
# Outputs:
#   weights/g4096_*_d128_L3_hw_sai.bytes
#   weights/g4096_*_d128_L3_hw_sai_training_profile.csv
#
# Submit:
#   sbatch run_train_generalization_sai_engaging.sh
#
#SBATCH -J lo_train_gen_sai
#SBATCH --array=0-3
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o slurm_logs/lo_train_gen_sai_%A_%a.out
#SBATCH -e slurm_logs/lo_train_gen_sai_%A_%a.err

set -euo pipefail

cd /home/osbo/ondemand/fluid-sim || exit 1

module load miniforge
module load cuda
source activate fluid

mkdir -p weights slurm_logs

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID:-NA}, task ${SLURM_ARRAY_TASK_ID:-NA} on $(hostname)"
echo "Started: $(date)"
echo "Training mode: SAI loss (--loss-mode sai)"
echo "================================================================"

python3 -u python/train_generalization_configs.py \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --loss-mode sai \
  --name-suffix _sai

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
