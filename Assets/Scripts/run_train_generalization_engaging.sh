#!/bin/bash
# ORCD Engaging (MIT): GPU partition, one array task = one model.
#   GPU:   1x NVIDIA H200  (-G h200:1)
#   CPU:   8 cores         (--cpus-per-task)
#   RAM:   32 GiB          (--mem)
#   Node:  1 node, 1 task per array index
#SBATCH -J lo_train_gen
#SBATCH --array=0-3
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o slurm_logs/lo_train_gen_%A_%a.out
#SBATCH -e slurm_logs/lo_train_gen_%A_%a.err

# Generalization training sweep (N=4096) — 4 parallel jobs:
#   [0] ID baseline
#   [1] topology holdout (no closed in train)
#   [2] parameter-range holdout (train low contrast)
#   [3] compositional holdout (no closed + low contrast in train)
#
# Prereq datasets:
#   python3 generate_generalization_datasets.py
#
# Submit:
#   sbatch run_train_generalization_engaging.sh
#
# List configs:
#   python3 train_generalization_configs.py --list

cd /home/osbo/ondemand/fluid-sim/Assets/Scripts || exit 1

module load miniforge
module load cuda
source activate fluid

mkdir -p weights slurm_logs

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Started: $(date)"
echo "================================================================"

python3 -u train_generalization_configs.py --index "${SLURM_ARRAY_TASK_ID}" --max-frames 20

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
