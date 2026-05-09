#!/bin/bash
#SBATCH -J lo_sai
#SBATCH -p mit_normal_gpu
#SBATCH -G h200:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o slurm_logs/lo_sai_%j.out
#SBATCH -e slurm_logs/lo_sai_%j.err

# Train N=8192 d_model=128 L=3 hw baseline with SAI loss (--loss-mode sai).
# Weights: weights/v2_8192_d128_L3_hw_sai.bytes
# Training CSV: weights/v2_8192_d128_L3_hw_sai_training_profile.csv
# Step checkpoints (every 2000 steps): weights/v2_8192_d128_L3_hw_sai_step*.bytes
#
# After training, evaluate checkpoints with cosine-Hutchinson loss:
#   python3 EvalCheckpoints.py \
#     --csv weights/v2_8192_d128_L3_hw_sai_training_profile.csv \
#     --loss-mode cosine_hutchinson
#
# Submit: sbatch run_sai_train_engaging.sh

cd /home/osbo/ondemand/fluid-sim/Assets/Scripts || exit 1

module load miniforge
module load cuda
source activate fluid

mkdir -p weights slurm_logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Started: $(date)"
echo "================================================================"

python3 -u LeafOnly.py \
    --data-folder data/multiphase_v2_8192/train \
    --leaf-size 128 \
    --max-mixed-size 8192 \
    --d-model 128 \
    --num-layers 3 \
    --num-heads 8 \
    --lr 2e-4 \
    --steps 100000 \
    --weights-out weights/v2_8192_d128_L3_hw_sai.bytes \
    --loss-mode sai \
    --track-training \
    --auto-stop

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
