"""Stage 2: DiD estimators for single-state and pooled (multi-state) analysis."""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from generate_data import generate_panel, generate_pooled_panel


def _compute_site_deltas(panel):
    """Compute site-level DiD scores: delta_i = mean(Y_treatment) - mean(Y_training).

    Returns DataFrame with columns: site_id, treated, state, delta.
    """
    group_cols = ['site_id', 'treated', 'phase']
    if 'state' in panel.columns:
        group_cols.insert(2, 'state')

    site_scores = panel.groupby(group_cols)['Y'].mean().reset_index()
    training = site_scores[site_scores['phase'] == 'training'].copy()
    treatment = site_scores[site_scores['phase'] == 'treatment'].copy()

    merge_cols = ['site_id', 'treated']
    if 'state' in panel.columns:
        merge_cols.append('state')

    merged = training.merge(treatment, on=merge_cols, suffixes=('_pre', '_post'))
    merged['delta'] = merged['Y_post'] - merged['Y_pre']

    keep_cols = merge_cols + ['delta']
    return merged[keep_cols]


def estimate_att(panel):
    """Single-state DiD estimator with Welch t-test.

    Returns (att_hat, se, rejected).
    """
    deltas = _compute_site_deltas(panel)

    treated_d = deltas[deltas['treated'] == 1]['delta'].dropna().values
    control_d = deltas[deltas['treated'] == 0]['delta'].dropna().values

    if len(treated_d) < 2 or len(control_d) < 2:
        return np.nan, np.nan, False

    t_result = stats.ttest_ind(treated_d, control_d, equal_var=False)
    att_hat = treated_d.mean() - control_d.mean()
    se = abs(att_hat / t_result.statistic) if t_result.statistic != 0 else np.nan
    rejected = bool(t_result.pvalue < 0.05)

    return att_hat, se, rejected


def estimate_att_pooled(panel):
    """Pooled multi-state DiD estimator with state fixed effects.

    Demeans delta_i within each state (absorbing state FE), then runs a
    Welch t-test on treated vs control demeaned deltas.

    Returns (att_hat, se, rejected).
    """
    deltas = _compute_site_deltas(panel)

    if 'state' not in deltas.columns:
        return estimate_att(panel)

    # State fixed effects: demean delta within each state
    state_means = deltas.groupby('state')['delta'].transform('mean')
    deltas['delta_demeaned'] = deltas['delta'] - state_means

    treated_d = deltas[deltas['treated'] == 1]['delta_demeaned'].dropna().values
    control_d = deltas[deltas['treated'] == 0]['delta_demeaned'].dropna().values

    if len(treated_d) < 2 or len(control_d) < 2:
        return np.nan, np.nan, False

    t_result = stats.ttest_ind(treated_d, control_d, equal_var=False)
    att_hat = treated_d.mean() - control_d.mean()
    se = abs(att_hat / t_result.statistic) if t_result.statistic != 0 else np.nan
    rejected = bool(t_result.pvalue < 0.05)

    return att_hat, se, rejected


# Backward-compatible alias
estimate_att_cs = estimate_att


def run_single_sim(params, seed):
    """Run one single-state simulation."""
    rng = np.random.default_rng(seed)
    panel = generate_panel(params, rng)
    att_hat, se, rejected = estimate_att(panel)
    return {**params, 'seed': seed, 'att_hat': att_hat, 'se': se, 'rejected': rejected}


def run_pooled_sim(params, seed):
    """Run one pooled (AP + Odisha) simulation.

    Returns dict with AP-only, Odisha-only, and pooled results.
    """
    rng = np.random.default_rng(seed)
    panel = generate_pooled_panel(params, rng)

    # AP only
    ap_panel = panel[panel['state'] == 'AP']
    att_ap, se_ap, rej_ap = estimate_att(ap_panel)

    # Odisha only
    od_panel = panel[panel['state'] == 'Odisha']
    att_od, se_od, rej_od = estimate_att(od_panel)

    # Pooled with state FE
    att_pool, se_pool, rej_pool = estimate_att_pooled(panel)

    return {
        **params, 'seed': seed,
        'att_ap': att_ap, 'se_ap': se_ap, 'rejected_ap': rej_ap,
        'att_od': att_od, 'se_od': se_od, 'rejected_od': rej_od,
        'att_pooled': att_pool, 'se_pooled': se_pool, 'rejected_pooled': rej_pool,
    }


def main():
    """Load a panel CSV and run estimation."""
    parser = argparse.ArgumentParser(description='Estimate ATT')
    parser.add_argument('--panel', type=str, default='output/example_panel.csv')
    parser.add_argument('--pooled', action='store_true',
                        help='Use pooled estimator with state FE')
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    print(f"Panel: {panel.shape[0]} rows, {panel['site_id'].nunique()} sites")

    if 'state' in panel.columns:
        for state in panel['state'].unique():
            sp = panel[panel['state'] == state]
            print(f"  {state}: {sp['site_id'].nunique()} sites")

    if args.pooled or 'state' in panel.columns:
        att, se, rej = estimate_att_pooled(panel)
        print(f"\n--- Pooled ATT (state FE) ---")
    else:
        att, se, rej = estimate_att(panel)
        print(f"\n--- ATT ---")

    print(f"ATT:      {att:.4f}")
    print(f"SE:       {se:.4f}")
    if se and se > 0:
        print(f"t-stat:   {att / se:.4f}")
    print(f"Rejected: {rej}")


if __name__ == '__main__':
    main()
