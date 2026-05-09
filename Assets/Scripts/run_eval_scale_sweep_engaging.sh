#!/bin/bash
# ORCD Engaging (MIT): GPU partition, one array task = one node.
#   GPU:   1× NVIDIA H200  (-G h200:1)
#   CPU:   8 cores         (--cpus-per-task; InspectModel + CUDA)
#   RAM:   32 GiB           (--mem)
#   Node:  1 node, 1 task per array index
#SBATCH -J lo_eval_sweep
#SBATCH --array=0-4
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o slurm_logs/lo_eval_sweep_%A_%a.out
#SBATCH -e slurm_logs/lo_eval_sweep_%A_%a.err

# Scale sweep evaluation — 5 parallel jobs (one per scale: 1024/2048/4096/8192/16384).
# Each task runs InspectModel --test-only for all 20 test frames and averages results.
#
# Submit:          sbatch run_eval_scale_sweep_engaging.sh
# After jobs done: python3 EvalScaleSweep.py --merge
# Output CSV:      Assets/Scripts/results/scale_sweep.csv
#
# Each task writes: results/scale_sweep_task_NNN.csv
# Merge step:       python3 EvalScaleSweep.py --merge
#
# Check status:    squeue -u $USER -o "%.8i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"

cd /home/osbo/ondemand/fluid-sim/Assets/Scripts || exit 1

module load miniforge
module load cuda
source activate fluid

mkdir -p results slurm_logs

echo "================================================================"
echo "Job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "Started: $(date)"
echo "================================================================"

python3 -u EvalScaleSweep.py --index "${SLURM_ARRAY_TASK_ID}"

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
