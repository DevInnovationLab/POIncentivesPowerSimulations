#!/usr/bin/env python3
"""Stage 3: Full power sweep across the parameter grid.

Usage (Mac M1, default):
    python run_power_sweep.py --n_sims 1000

Usage (HPC with SLURM job array):
    # Submit as array job — each task handles a chunk of parameter combos
    python run_power_sweep.py --n_sims 1000 --hpc --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 5

    # After all chunks complete, merge:
    python run_power_sweep.py --merge_chunks --n_chunks 5
"""

import argparse
import itertools
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PARAM_GRID, N_SIMS, SEED, TREATMENT_WEEKS
from estimate import run_single_sim
from generate_data import generate_panel

# Default workers: Mac M1 has 8 cores (4P + 4E), use 6 to leave headroom
DEFAULT_WORKERS_LOCAL = min(6, cpu_count())


def run_power_for_combo(args):
    """Run n_sims simulations for one parameter combination.

    Parameters
    ----------
    args : tuple
        (params_dict, n_sims, base_seed)

    Returns
    -------
    dict
        All param values plus power, mean_att, mean_se, sd_att.
    """
    params, n_sims, base_seed = args

    att_list = []
    se_list = []
    rejected_list = []

    for i in range(n_sims):
        seed = base_seed + i
        res = run_single_sim(params, seed)
        att_list.append(res['att_hat'])
        se_list.append(res['se'])
        rejected_list.append(res['rejected'])

    att_arr = np.array(att_list, dtype=float)
    se_arr = np.array(se_list, dtype=float)
    rej_arr = np.array(rejected_list, dtype=float)

    # Derive tau for the record (using same finite-horizon formula as generate_data)
    target_att = params.get('target_att', None)
    rho = params['rho']
    if target_att is not None:
        T = TREATMENT_WEEKS
        if rho == 0:
            tau = target_att
        else:
            amp = (T - rho * (1 - rho ** T) / (1 - rho)) / (T * (1 - rho))
            tau = target_att / amp
    else:
        tau = params.get('tau', np.nan)

    result = {
        'mu_baseline': params['mu_baseline'],
        'sigma_baseline': params['sigma_baseline'],
        'target_att': target_att if target_att is not None else np.nan,
        'tau': tau,
        'rho': rho,
        'h_init': params['h_init'],
        'power': np.nanmean(rej_arr),
        'mean_att': np.nanmean(att_arr),
        'mean_se': np.nanmean(se_arr),
        'sd_att': np.nanstd(att_arr),
    }
    return result


def build_task_list(n_sims):
    """Generate all parameter combos and return (tasks, param_names, all_combos)."""
    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[k] for k in param_names]
    all_combos = list(itertools.product(*param_values))

    tasks = []
    for combo_idx, combo in enumerate(all_combos):
        params = dict(zip(param_names, combo))
        base_seed = SEED + combo_idx * n_sims
        tasks.append((params, n_sims, base_seed))

    return tasks, param_names, all_combos


def merge_chunks(output_dir, n_chunks):
    """Merge chunk CSV files into a single power_results.csv."""
    chunks = []
    for i in range(n_chunks):
        path = os.path.join(output_dir, f'power_results_chunk{i}.csv')
        if os.path.isfile(path):
            chunks.append(pd.read_csv(path))
        else:
            print(f"WARNING: Missing chunk file: {path}")

    if not chunks:
        print("ERROR: No chunk files found.")
        return

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(
        ['mu_baseline', 'sigma_baseline', 'target_att', 'rho', 'h_init']
    ).reset_index(drop=True)

    output_path = os.path.join(output_dir, 'power_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Merged {len(chunks)} chunks ({len(df)} combos) -> {output_path}")
    print(f"Power range: {df['power'].min():.3f} - {df['power'].max():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run power sweep simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HPC mode (SLURM):
  Splits the 3,600 parameter combos across --n_chunks parallel jobs.
  Each job uses all available cores on its node for within-combo parallelism.

  Example SLURM script:
    #SBATCH --array=0-35
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=8G
    #SBATCH --time=02:00:00
    python run_power_sweep.py --n_sims 1000 --hpc \\
        --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 36

  After completion:
    python run_power_sweep.py --merge_chunks --n_chunks 36

Recommended HPC allocation (3,600 combos x 1,000 sims):
  --n_chunks 5 (720 combos/chunk) with 48 cores and 192GB RAM per node (ssd partition)
  Estimated wall time: ~1-2 hours per chunk
""",
    )
    parser.add_argument('--n_sims', type=int, default=N_SIMS,
                        help=f"Simulations per combo (default: {N_SIMS})")
    parser.add_argument('--n_workers', type=int, default=None,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS_LOCAL} local, all cores on HPC)")
    parser.add_argument('--save_panels', action='store_true',
                        help="Save one example panel per combo to output/panels/")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Output directory (default: output)")

    # HPC options
    parser.add_argument('--hpc', action='store_true',
                        help="HPC mode: use all cores, process only assigned chunk")
    parser.add_argument('--chunk_id', type=int, default=None,
                        help="HPC chunk index (0-based, e.g. $SLURM_ARRAY_TASK_ID)")
    parser.add_argument('--n_chunks', type=int, default=None,
                        help="Total number of HPC chunks")
    parser.add_argument('--merge_chunks', action='store_true',
                        help="Merge chunk CSVs into power_results.csv (run after all chunks complete)")

    args = parser.parse_args()

    # Handle merge mode
    if args.merge_chunks:
        if args.n_chunks is None:
            parser.error("--merge_chunks requires --n_chunks")
        merge_chunks(args.output_dir, args.n_chunks)
        return

    # Set worker count
    if args.n_workers is not None:
        n_workers = args.n_workers
    elif args.hpc:
        n_workers = cpu_count()  # use all cores on HPC node
    else:
        n_workers = DEFAULT_WORKERS_LOCAL

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_panels:
        os.makedirs(os.path.join(args.output_dir, 'panels'), exist_ok=True)

    # Build full task list
    tasks, param_names, all_combos = build_task_list(args.n_sims)
    n_total = len(tasks)

    # HPC: select chunk subset
    if args.hpc:
        if args.chunk_id is None or args.n_chunks is None:
            parser.error("--hpc requires --chunk_id and --n_chunks")
        chunk_size = int(np.ceil(n_total / args.n_chunks))
        start = args.chunk_id * chunk_size
        end = min(start + chunk_size, n_total)
        tasks = tasks[start:end]
        all_combos = all_combos[start:end]
        print(f"HPC mode: chunk {args.chunk_id}/{args.n_chunks}, "
              f"combos {start}-{end-1} ({len(tasks)} combos)")

    print(f"Parameter combinations: {len(tasks)}")
    print(f"Simulations per combo:  {args.n_sims}")
    print(f"Total simulations:      {len(tasks) * args.n_sims:,}")
    print(f"Workers:                {n_workers}")
    print()

    # Optionally save example panels
    if args.save_panels:
        print("Saving example panels...")
        for combo_idx, (task_params, _, base_seed) in enumerate(tqdm(tasks, desc="Panels")):
            rng = np.random.default_rng(base_seed)
            panel = generate_panel(task_params, rng)
            panel.to_csv(
                os.path.join(args.output_dir, 'panels', f'panel_{combo_idx}.csv'),
                index=False,
            )
        print()

    # Run power sweep with multiprocessing
    t0 = time.time()
    results = []

    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(run_power_for_combo, tasks),
            total=len(tasks),
            desc="Power sweep",
        ):
            results.append(result)

    elapsed = time.time() - t0

    # Collect into DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values(
        ['mu_baseline', 'sigma_baseline', 'target_att', 'rho', 'h_init']
    ).reset_index(drop=True)

    if args.hpc and args.chunk_id is not None:
        output_path = os.path.join(args.output_dir, f'power_results_chunk{args.chunk_id}.csv')
    else:
        output_path = os.path.join(args.output_dir, 'power_results.csv')
    df.to_csv(output_path, index=False)

    # Summary
    print()
    print("=" * 60)
    print(f"Completed {len(tasks)} combos x {args.n_sims} sims = {len(tasks) * args.n_sims:,} total")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Results saved to: {output_path}")
    print(f"Power range: {df['power'].min():.3f} - {df['power'].max():.3f}")
    print(f"Mean power:  {df['power'].mean():.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
