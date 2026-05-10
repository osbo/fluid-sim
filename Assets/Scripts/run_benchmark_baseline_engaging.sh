#!/bin/bash
# ORCD Engaging: H200 benchmark of the baseline LeafOnly system
# (scale=8192, leaf=128, d_model=128, num_layers=3, use_highways=True, cos-Hutchinson loss).
#
#   GPU:  1× NVIDIA H200  (-G h200:1)
#   CPU:  8 cores         (--cpus-per-task)
#   RAM:  64 GiB           (--mem)
#   Node: 1 node, 1 task
#SBATCH -J lo_bench_baseline
#SBATCH -p mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -G h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o slurm_logs/lo_bench_baseline_%j.out
#SBATCH -e slurm_logs/lo_bench_baseline_%j.err
#
# Submit:
#   sbatch run_benchmark_baseline_engaging.sh
#
# Override weights:
#   sbatch run_benchmark_baseline_engaging.sh --weights /path/to/v2_8192_d128_L3_hw.bytes
#
# Skip ncu (e.g. if Nsight Compute module is unavailable):
#   sbatch run_benchmark_baseline_engaging.sh --skip-ncu
#
# Pass-through CLI args go straight to BenchmarkBaseline.py.

cd /home/osbo/ondemand/fluid-sim/Assets/Scripts || exit 1

module load miniforge
module load cuda
# Nsight Compute may live under a separate module on some systems; tolerate absence.
module load nsight-compute 2>/dev/null || true

source activate fluid

# Install pynvml if it's not already in the env (cheap; skip silently otherwise).
python3 -c "import pynvml" 2>/dev/null || pip install --quiet --user pynvml || true

mkdir -p results/baseline_benchmark slurm_logs

echo "================================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Started: $(date)"
echo "GPU info:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "ncu: $(command -v ncu || echo MISSING)"
echo "================================================================"

python3 -u BenchmarkBaseline.py "$@"

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
