#!/bin/bash
#SBATCH --job-name=po_power_sim
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --qos=ssd
#SBATCH --array=0-4
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# -------------------------------------------------------------------
# HPC SLURM submission script for PO Incentives Power Simulation
#
# ssd partition specs: 48 cores/node, 192 GB RAM, max 5 nodes/user
#
# Allocation:
#   - 5 array tasks (one per node) -> 720 combos per task
#   - 48 cores per task -> full node, Python multiprocessing
#   - 192 GB RAM per task -> full node memory
#   - 4 hour wall time (conservative; expect ~1-2 hours)
#
# Total: 5 nodes x 48 cores = 240 CPUs (maxes out user limit)
# -------------------------------------------------------------------

mkdir -p logs output

module load python  # adjust to your module name if different

python run_power_sweep.py \
    --n_sims 1000 \
    --hpc \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --n_chunks 5

# After all array tasks complete, run merge:
#   python run_power_sweep.py --merge_chunks --n_chunks 5
#   python visualize.py
