#!/bin/bash
#SBATCH -J lo_eval_ablation
#SBATCH --array=0-19
#SBATCH -p mit_normal_gpu
#SBATCH -G h200:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o slurm_logs/lo_eval_ablation_%A_%a.out
#SBATCH -e slurm_logs/lo_eval_ablation_%A_%a.err

# Architecture ablation evaluation — 20 parallel jobs (one per trained config).
# Each task runs InspectModel --test-only for all 20 test frames and averages results.
#
# Submit:          sbatch run_eval_ablation_engaging.sh
# Submit subset:   sbatch --array=0-4  run_eval_ablation_engaging.sh  # scale sweep configs only
#                  sbatch --array=5-19 run_eval_ablation_engaging.sh  # ablation configs only
# After jobs done: python3 EvalScaleSweep.py --ablation --merge
# Output CSV:      Assets/Scripts/results/ablation_sweep.csv
#
# Each task writes: results/ablation_sweep_task_NNN.csv
# List configs:     python3 train_configs.py --list
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

python3 -u EvalScaleSweep.py --ablation --index "${SLURM_ARRAY_TASK_ID}"

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
