#!/bin/bash
#SBATCH -J leaf_only
#SBATCH -p mit_normal_gpu
#SBATCH -G h200:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o leaf_only_%j.out
#SBATCH -e leaf_only_%j.err

# LeafOnly training on ORCD Engaging: 1x H200, 8 cores, 32 GB RAM, 6 hours.
# For "as long as possible" (48h, preemptable; job may requeue if preempted):
#   change to: #SBATCH -p mit_preemptable
#   change to: #SBATCH -t 48:00:00
#   add line:  #SBATCH --requeue

cd /home/osbo/ondemand/fluid-sim/Assets/Scripts || exit 1
module load miniforge
module load cuda
source activate fluid

python3 -u LeafOnly.py
