#!/bin/bash
# ORCD Engaging (MIT): GPU partition, single-node job.
#   GPU:   1× NVIDIA H200  (-G h200:1)
#   CPU:   8 cores         (--cpus-per-task)
#   RAM:   32 GiB           (--mem)
#   Node:  1 node, 1 Slurm task
#SBATCH -J leaf_only
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o slurm_logs/leaf_only_%j.out
#SBATCH -e slurm_logs/leaf_only_%j.err

# LeafOnly training on ORCD Engaging: 1× H200, 8 CPUs, 32 GiB RAM, 6 h wall time.
# For "as long as possible" (48h, preemptable; job may requeue if preempted):
#   change to: #SBATCH -p mit_preemptable
#   change to: #SBATCH -t 48:00:00
#   add line:  #SBATCH --requeue

cd /home/osbo/ondemand/fluid-sim || exit 1
mkdir -p slurm_logs
module load miniforge
module load cuda
source activate fluid

python3 -u python/LeafOnly.py --data-folder data/multiphase_v2_16384/train --leaf-size 256 --max-mixed-size 16384 --lr 2e-4