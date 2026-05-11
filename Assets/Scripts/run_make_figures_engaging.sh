#!/bin/bash
# ORCD Engaging: render paper figures from multiphase_v2 data.
# Needs GPU only for the Ours panel of the convergence figure.
#
#SBATCH -J mk_mp_figs
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0:45:00
#SBATCH -o slurm_logs/mk_mp_figs_%j.out
#SBATCH -e slurm_logs/mk_mp_figs_%j.err
#
# Submit:
#   sbatch run_make_figures_engaging.sh --fig all \
#       --ours-weights /orcd/home/002/osbo/ondemand/fluid-sim/Assets/StreamingAssets/leaf_only_weights_sim_8192.bytes \
#       --conv-scale 8192
#
# Override individual figures, e.g.:
#   sbatch run_make_figures_engaging.sh --fig hero --hero-scale 4096
#
# All extra args pass through to MakeMultiphaseFigures.py.

cd /home/osbo/ondemand/fluid-sim/Assets/Scripts || exit 1

module load miniforge
module load cuda

source activate fluid

mkdir -p ../../Paper/figures slurm_logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
echo "================================================================"

python3 -u MakeMultiphaseFigures.py "$@"

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
