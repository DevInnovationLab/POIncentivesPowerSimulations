#!/bin/bash
#SBATCH --job-name=po_power_sim
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --qos=ssd
#SBATCH --array=0-4
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# -------------------------------------------------------------------
# HPC SLURM submission script for PO Incentives Power Simulation
#
# ssd partition specs: 48 cores/node, 192 GB RAM, max 5 nodes/user
#
# Usage:
#   sbatch submit_hpc.sh                    # single-state sweep (3600 combos)
#   sbatch submit_hpc.sh --pooled           # pooled sweep
#   sbatch submit_hpc.sh --comparison       # comprehensive comparison sweep
# -------------------------------------------------------------------

mkdir -p logs output

module load python/3.11.9

MODE="${1:-single}"

if [ "$MODE" = "--comparison" ]; then
    echo "Running comparison sweep (chunk $SLURM_ARRAY_TASK_ID of 5)"
    python3 run_comparison_sweep.py \
        --n_sims 1000 \
        --hpc \
        --chunk_id $SLURM_ARRAY_TASK_ID \
        --n_chunks 5
elif [ "$MODE" = "--pooled" ]; then
    echo "Running pooled sweep (chunk $SLURM_ARRAY_TASK_ID of 5)"
    python3 run_power_sweep.py \
        --pooled \
        --n_sims 1000 \
        --hpc \
        --chunk_id $SLURM_ARRAY_TASK_ID \
        --n_chunks 5
else
    echo "Running single-state sweep (chunk $SLURM_ARRAY_TASK_ID of 5)"
    python3 run_power_sweep.py \
        --n_sims 1000 \
        --hpc \
        --chunk_id $SLURM_ARRAY_TASK_ID \
        --n_chunks 5
fi

# After all array tasks complete, merge and visualize:
#
# Single-state:
#   python3 run_power_sweep.py --merge_chunks --n_chunks 5
#   python3 visualize.py
#
# Pooled:
#   python3 run_power_sweep.py --pooled --merge_chunks --n_chunks 5
#   python3 visualize.py --pooled
#
# Comparison:
#   python3 run_comparison_sweep.py --merge_chunks --n_chunks 5
#   python3 visualize_comparison.py
