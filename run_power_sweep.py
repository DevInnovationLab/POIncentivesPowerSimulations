#!/usr/bin/env python3
"""Stage 3: Full power sweep across the parameter grid.

Supports both single-state (AP only) and pooled (AP + Odisha) modes.

Usage (Mac M1, default — single state):
    python run_power_sweep.py --n_sims 1000

Usage (pooled two-state):
    python run_power_sweep.py --pooled --n_sims 1000

Usage (HPC with SLURM job array):
    python run_power_sweep.py --n_sims 1000 --hpc --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 5
    python run_power_sweep.py --pooled --n_sims 1000 --hpc --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 5

    # After all chunks complete, merge:
    python run_power_sweep.py --merge_chunks --n_chunks 5 [--pooled]
"""

import argparse
import itertools
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PARAM_GRID, POOLED_PARAM_GRID, N_SIMS, SEED
from estimate import run_single_sim, run_pooled_sim

# Default workers: Mac M1 has 8 cores (4P + 4E), use 6 to leave headroom
DEFAULT_WORKERS_LOCAL = min(6, cpu_count())


def run_power_for_combo(args):
    """Run n_sims simulations for one single-state parameter combination."""
    params, n_sims, base_seed = args

    results = []
    for i in range(n_sims):
        res = run_single_sim(params, seed=base_seed + i)
        results.append(res)

    att_arr = np.array([r['att_hat'] for r in results], dtype=float)
    se_arr = np.array([r['se'] for r in results], dtype=float)
    rej_arr = np.array([r['rejected'] for r in results], dtype=float)

    return {
        'mu_baseline': params['mu_baseline'],
        'sigma_baseline': params['sigma_baseline'],
        'target_att': params.get('target_att', np.nan),
        'rho': params['rho'],
        'h_init': params['h_init'],
        'power': np.nanmean(rej_arr),
        'mean_att': np.nanmean(att_arr),
        'mean_se': np.nanmean(se_arr),
        'sd_att': np.nanstd(att_arr),
    }


def run_power_for_combo_pooled(args):
    """Run n_sims simulations for one pooled parameter combination.

    Returns dict with AP-only, Odisha-only, and pooled power.
    """
    params, n_sims, base_seed = args

    results = []
    for i in range(n_sims):
        res = run_pooled_sim(params, seed=base_seed + i)
        results.append(res)

    def _summarize(key_prefix):
        att = np.array([r[f'att_{key_prefix}'] for r in results], dtype=float)
        se = np.array([r[f'se_{key_prefix}'] for r in results], dtype=float)
        rej = np.array([r[f'rejected_{key_prefix}'] for r in results], dtype=float)
        return {
            f'power_{key_prefix}': np.nanmean(rej),
            f'mean_att_{key_prefix}': np.nanmean(att),
            f'mean_se_{key_prefix}': np.nanmean(se),
            f'sd_att_{key_prefix}': np.nanstd(att),
        }

    row = {
        'mu_baseline_ap': params['mu_baseline_ap'],
        'mu_baseline_od': params['mu_baseline_od'],
        'sigma_baseline': params['sigma_baseline'],
        'target_att': params['target_att'],
        'rho': params['rho'],
        'h_init': params['h_init'],
        'effect_ratio': params.get('effect_ratio', 1.0),
    }
    row.update(_summarize('ap'))
    row.update(_summarize('od'))
    row.update(_summarize('pooled'))

    return row


def build_task_list(param_grid, n_sims):
    """Generate all parameter combos and return list of (params, n_sims, base_seed)."""
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    all_combos = list(itertools.product(*param_values))

    tasks = []
    for combo_idx, combo in enumerate(all_combos):
        params = dict(zip(param_names, combo))
        base_seed = SEED + combo_idx * n_sims
        tasks.append((params, n_sims, base_seed))

    return tasks


def merge_chunks(output_dir, n_chunks, prefix='power_results'):
    """Merge chunk CSV files into a single results CSV."""
    chunks = []
    for i in range(n_chunks):
        path = os.path.join(output_dir, f'{prefix}_chunk{i}.csv')
        if os.path.isfile(path):
            chunks.append(pd.read_csv(path))
        else:
            print(f"WARNING: Missing chunk file: {path}")

    if not chunks:
        print("ERROR: No chunk files found.")
        return

    df = pd.concat(chunks, ignore_index=True)
    output_path = os.path.join(output_dir, f'{prefix}.csv')
    df.to_csv(output_path, index=False)
    print(f"Merged {len(chunks)} chunks ({len(df)} combos) -> {output_path}")
    print(f"Power range: {df.filter(like='power').min().min():.3f} - {df.filter(like='power').max().max():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run power sweep simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HPC mode (SLURM):
  Splits parameter combos across --n_chunks parallel jobs.
  Each job uses all available cores on its node.

  Example SLURM script:
    #SBATCH --array=0-4
    #SBATCH --cpus-per-task=48
    #SBATCH --mem=192G
    python run_power_sweep.py --n_sims 1000 --hpc \\
        --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 5

  After completion:
    python run_power_sweep.py --merge_chunks --n_chunks 5
""",
    )
    parser.add_argument('--n_sims', type=int, default=N_SIMS,
                        help=f"Simulations per combo (default: {N_SIMS})")
    parser.add_argument('--n_workers', type=int, default=None,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS_LOCAL} local, all cores on HPC)")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Output directory (default: output)")
    parser.add_argument('--pooled', action='store_true',
                        help="Run pooled (AP + Odisha) sweep instead of single-state")

    # HPC options
    parser.add_argument('--hpc', action='store_true',
                        help="HPC mode: use all cores, process only assigned chunk")
    parser.add_argument('--chunk_id', type=int, default=None,
                        help="HPC chunk index (0-based, e.g. $SLURM_ARRAY_TASK_ID)")
    parser.add_argument('--n_chunks', type=int, default=None,
                        help="Total number of HPC chunks")
    parser.add_argument('--merge_chunks', action='store_true',
                        help="Merge chunk CSVs into final results (run after all chunks complete)")

    args = parser.parse_args()

    prefix = 'pooled_power_results' if args.pooled else 'power_results'

    # Handle merge mode
    if args.merge_chunks:
        if args.n_chunks is None:
            parser.error("--merge_chunks requires --n_chunks")
        merge_chunks(args.output_dir, args.n_chunks, prefix=prefix)
        return

    # Set worker count
    if args.n_workers is not None:
        n_workers = args.n_workers
    elif args.hpc:
        n_workers = cpu_count()
    else:
        n_workers = DEFAULT_WORKERS_LOCAL

    os.makedirs(args.output_dir, exist_ok=True)

    # Build task list
    param_grid = POOLED_PARAM_GRID if args.pooled else PARAM_GRID
    worker_fn = run_power_for_combo_pooled if args.pooled else run_power_for_combo
    tasks = build_task_list(param_grid, args.n_sims)
    n_total = len(tasks)

    mode_label = "Pooled (AP + Odisha)" if args.pooled else "Single-state (AP)"
    print(f"Mode: {mode_label}")

    # HPC: select chunk subset
    if args.hpc:
        if args.chunk_id is None or args.n_chunks is None:
            parser.error("--hpc requires --chunk_id and --n_chunks")
        chunk_size = int(np.ceil(n_total / args.n_chunks))
        start = args.chunk_id * chunk_size
        end = min(start + chunk_size, n_total)
        tasks = tasks[start:end]
        print(f"HPC mode: chunk {args.chunk_id}/{args.n_chunks}, "
              f"combos {start}-{end-1} ({len(tasks)} combos)")

    print(f"Parameter combinations: {len(tasks)}")
    print(f"Simulations per combo:  {args.n_sims}")
    print(f"Total simulations:      {len(tasks) * args.n_sims:,}")
    print(f"Workers:                {n_workers}")
    print()

    # Run power sweep
    t0 = time.time()
    results = []

    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_fn, tasks),
            total=len(tasks),
            desc="Power sweep",
        ):
            results.append(result)

    elapsed = time.time() - t0

    df = pd.DataFrame(results)

    if args.hpc and args.chunk_id is not None:
        output_path = os.path.join(args.output_dir, f'{prefix}_chunk{args.chunk_id}.csv')
    else:
        output_path = os.path.join(args.output_dir, f'{prefix}.csv')
    df.to_csv(output_path, index=False)

    # Summary
    print()
    print("=" * 60)
    print(f"Completed {len(tasks)} combos x {args.n_sims} sims = {len(tasks) * args.n_sims:,} total")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {output_path}")
    power_cols = [c for c in df.columns if c.startswith('power')]
    for col in power_cols:
        label = col.replace('power_', '').replace('power', 'overall') if args.pooled else 'overall'
        print(f"  {label} power range: {df[col].min():.3f} - {df[col].max():.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
