#!/bin/bash
# ORCD Engaging (MIT): GPU partition, one array task = one node.
#   GPU:   1× NVIDIA H200  (-G h200:1)
#   CPU:   8 cores         (--cpus-per-task)
#   RAM:   32 GiB           (--mem)
#   Node:  1 node, 1 task per array index
#SBATCH -J lo_train
#SBATCH --array=0-19
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o slurm_logs/lo_train_%A_%a.out
#SBATCH -e slurm_logs/lo_train_%A_%a.err

# LeafOnly ablation study — 20 parallel training jobs on ORCD Engaging.
# Each array task trains one configuration (index = SLURM_ARRAY_TASK_ID).
#
# Submit:          sbatch run_train_all_engaging.sh
# Submit subset:   sbatch --array=0-4   run_train_all_engaging.sh   # scale sweep only
#                  sbatch --array=1-3,5-19 run_train_all_engaging.sh # ablation only
# List configs:    python3 train_configs.py --list
# Check status:    squeue -u $USER -o "%.8i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
#
# Weights land in:  Assets/Scripts/weights/v2_<scale>_<arch>.bytes
# Training CSVs in: Assets/Scripts/weights/v2_<scale>_<arch>_training_profile.csv
# Step ckpts in:    Assets/Scripts/weights/v2_<scale>_<arch>_step<N>.bytes  (every 2000 steps)
#
# After training, run EvalCheckpoints.py on each CSV to append pcg_iters / SAI loss.

cd /home/osbo/ondemand/fluid-sim/Assets/Scripts || exit 1

module load miniforge
module load cuda
source activate fluid

mkdir -p weights slurm_logs

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Started: $(date)"
echo "================================================================"

python3 -u train_configs.py --index "${SLURM_ARRAY_TASK_ID}"

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
