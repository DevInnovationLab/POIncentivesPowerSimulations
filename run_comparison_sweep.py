#!/usr/bin/env python3
"""Stage 3b: Comprehensive MDE comparison sweep.

Compares MDE across scenarios:
  - AP-only vs AP+Odisha pooled
  - 2 vs 3 measurements per week
  - Study duration: 6 months, 1 year, 1.5 years from AP start

Usage (Mac M1):
    python run_comparison_sweep.py --n_sims 1000

Usage (HPC):
    python run_comparison_sweep.py --n_sims 1000 --hpc --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 10
    python run_comparison_sweep.py --merge_chunks --n_chunks 10
"""

import argparse
import itertools
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    COMPARISON_PARAM_GRID, COMPARISON_N_MEASUREMENTS, STUDY_DURATIONS,
    N_SIMS, SEED,
)
from estimate import run_single_sim, run_pooled_sim

DEFAULT_WORKERS_LOCAL = min(6, cpu_count())


def _run_one_scenario(args):
    """Run n_sims for one (scenario, params) combination.

    Returns a dict with scenario metadata + power results.
    """
    scenario, params, n_sims, base_seed = args
    mode = scenario['mode']
    n_meas = scenario['n_measurements']
    study_end = scenario['study_end_week']
    duration_label = scenario['duration_label']

    results_list = []
    for i in range(n_sims):
        seed = base_seed + i
        if mode == 'ap_only':
            sim_params = {**params, 'n_measurements': n_meas, 'study_end_week': study_end}
            res = run_single_sim(sim_params, seed)
            results_list.append({
                'att': res['att_hat'], 'se': res['se'], 'rejected': res['rejected'],
            })
        elif mode == 'pooled':
            sim_params = {
                'mu_baseline_ap': params['mu_baseline'],
                'mu_baseline_od': params['mu_baseline'],
                'sigma_baseline': params['sigma_baseline'],
                'target_att': params['target_att'],
                'rho': params['rho'],
                'h_init': params['h_init'],
                'effect_ratio': 1.0,
                'n_measurements': n_meas,
                'study_end_week': study_end,
            }
            res = run_pooled_sim(sim_params, seed)
            results_list.append({
                'att': res['att_pooled'], 'se': res['se_pooled'],
                'rejected': res['rejected_pooled'],
            })

    att_arr = np.array([r['att'] for r in results_list], dtype=float)
    se_arr = np.array([r['se'] for r in results_list], dtype=float)
    rej_arr = np.array([r['rejected'] for r in results_list], dtype=float)

    row = {
        'mode': mode,
        'n_measurements': n_meas,
        'study_end_week': study_end,
        'duration_label': duration_label,
        **params,
        'power': np.nanmean(rej_arr),
        'mean_att': np.nanmean(att_arr),
        'mean_se': np.nanmean(se_arr),
        'sd_att': np.nanstd(att_arr),
    }
    return row


def build_comparison_tasks(n_sims):
    """Build all (scenario x param combo) tasks."""
    param_names = list(COMPARISON_PARAM_GRID.keys())
    param_values = [COMPARISON_PARAM_GRID[k] for k in param_names]
    all_combos = list(itertools.product(*param_values))

    scenarios = []
    for mode in ['ap_only', 'pooled']:
        for n_meas in COMPARISON_N_MEASUREMENTS:
            for dur_label, study_end in STUDY_DURATIONS.items():
                scenarios.append({
                    'mode': mode,
                    'n_measurements': n_meas,
                    'study_end_week': study_end,
                    'duration_label': dur_label,
                })

    tasks = []
    task_idx = 0
    for scenario in scenarios:
        for combo in all_combos:
            params = dict(zip(param_names, combo))
            base_seed = SEED + task_idx * n_sims
            tasks.append((scenario, params, n_sims, base_seed))
            task_idx += 1

    return tasks


def merge_chunks(output_dir, n_chunks):
    """Merge chunk CSV files."""
    chunks = []
    for i in range(n_chunks):
        path = os.path.join(output_dir, f'comparison_results_chunk{i}.csv')
        if os.path.isfile(path):
            chunks.append(pd.read_csv(path))
        else:
            print(f"WARNING: Missing chunk file: {path}")

    if not chunks:
        print("ERROR: No chunk files found.")
        return

    df = pd.concat(chunks, ignore_index=True)
    output_path = os.path.join(output_dir, 'comparison_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Merged {len(chunks)} chunks ({len(df)} rows) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive MDE comparison sweep.")
    parser.add_argument('--n_sims', type=int, default=N_SIMS)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--hpc', action='store_true')
    parser.add_argument('--chunk_id', type=int, default=None)
    parser.add_argument('--n_chunks', type=int, default=None)
    parser.add_argument('--merge_chunks', action='store_true')
    args = parser.parse_args()

    if args.merge_chunks:
        if args.n_chunks is None:
            parser.error("--merge_chunks requires --n_chunks")
        merge_chunks(args.output_dir, args.n_chunks)
        return

    if args.n_workers is not None:
        n_workers = args.n_workers
    elif args.hpc:
        n_workers = cpu_count()
    else:
        n_workers = DEFAULT_WORKERS_LOCAL

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = build_comparison_tasks(args.n_sims)
    n_total = len(tasks)

    # Count scenarios
    n_modes = 2  # ap_only, pooled
    n_meas = len(COMPARISON_N_MEASUREMENTS)
    n_dur = len(STUDY_DURATIONS)
    n_param_combos = n_total // (n_modes * n_meas * n_dur)

    print(f"Comparison sweep:")
    print(f"  Modes:        {n_modes} (AP-only, Pooled)")
    print(f"  Measurements: {COMPARISON_N_MEASUREMENTS}")
    print(f"  Durations:    {list(STUDY_DURATIONS.keys())}")
    print(f"  Param combos: {n_param_combos}")
    print(f"  Total tasks:  {n_total}")
    print(f"  Sims/task:    {args.n_sims}")
    print(f"  Total sims:   {n_total * args.n_sims:,}")

    if args.hpc:
        if args.chunk_id is None or args.n_chunks is None:
            parser.error("--hpc requires --chunk_id and --n_chunks")
        chunk_size = int(np.ceil(n_total / args.n_chunks))
        start = args.chunk_id * chunk_size
        end = min(start + chunk_size, n_total)
        tasks = tasks[start:end]
        print(f"\nHPC: chunk {args.chunk_id}/{args.n_chunks}, tasks {start}-{end-1} ({len(tasks)})")

    print(f"  Workers:      {n_workers}")
    print()

    t0 = time.time()
    results = []

    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_one_scenario, tasks),
            total=len(tasks),
            desc="Comparison sweep",
        ):
            results.append(result)

    elapsed = time.time() - t0

    df = pd.DataFrame(results)

    if args.hpc and args.chunk_id is not None:
        output_path = os.path.join(args.output_dir, f'comparison_results_chunk{args.chunk_id}.csv')
    else:
        output_path = os.path.join(args.output_dir, 'comparison_results.csv')
    df.to_csv(output_path, index=False)

    print()
    print("=" * 60)
    print(f"Completed {len(tasks)} tasks x {args.n_sims} sims = {len(tasks) * args.n_sims:,} total")
    print(f"Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Saved: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
