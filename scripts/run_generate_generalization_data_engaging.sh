#!/bin/bash
# ORCD Engaging (MIT): CPU partition for dataset generation.
#SBATCH -J lo_gen_data
#SBATCH -p mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -o slurm_logs/lo_gen_data_%j.out
#SBATCH -e slurm_logs/lo_gen_data_%j.err

# Generates all generalization splits under:
#   data/generalization_4096/
#
# Submit:
#   sbatch run_generate_generalization_data_engaging.sh
#
# Quick pilot:
#   python3 generate_generalization_datasets.py --pilot

cd /home/osbo/ondemand/fluid-sim || exit 1

module load miniforge
source activate fluid

mkdir -p slurm_logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Started: $(date)"
echo "================================================================"

python3 -u python/generate_generalization_datasets.py

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
