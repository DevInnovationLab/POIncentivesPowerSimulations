"""Stage 2: Callaway & Sant'Anna DiD estimator for the staggered rollout RCT."""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from generate_data import generate_panel


def estimate_att_cs(panel):
    """Callaway & Sant'Anna-style DiD estimator with site-level clustering.

    Computes site-level DiD scores (mean post-treatment Y minus pre-treatment
    Y for each site), then takes the difference between treated and control
    group means. SE is a standard two-sample clustered SE, which is provably
    correct because each site contributes one independent score.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with columns: site_id, calendar_week, Y, treated, cohort_g,
        phase.

    Returns
    -------
    tuple of (att_hat, se, rejected)
        att_hat : float — estimated aggregate ATT
        se : float — cluster-robust standard error (clustered at site level)
        rejected : bool — whether Welch's t-test p-value < 0.05
    """
    # Compute site-level DiD scores:
    # For each site, delta_i = mean(Y in treatment phase) - mean(Y in training phase)
    # This collapses the panel to one observation per site, eliminating
    # within-site serial correlation by construction.
    site_scores = panel.groupby(['site_id', 'treated', 'phase'])['Y'].mean().unstack('phase')

    if 'training' not in site_scores.columns or 'treatment' not in site_scores.columns:
        return np.nan, np.nan, False

    site_scores['delta'] = site_scores['treatment'] - site_scores['training']
    site_scores = site_scores.reset_index()

    treated_deltas = site_scores[site_scores['treated'] == 1]['delta'].dropna().values
    control_deltas = site_scores[site_scores['treated'] == 0]['delta'].dropna().values

    n_t = len(treated_deltas)
    n_c = len(control_deltas)

    if n_t < 2 or n_c < 2:
        return np.nan, np.nan, False

    # Welch's t-test (unequal variances, t-distribution with Satterthwaite df).
    # Each delta_i is one independent cluster-level observation.
    # Using scipy's ttest_ind for correct df calculation with only 25 per arm.
    t_result = stats.ttest_ind(treated_deltas, control_deltas, equal_var=False)
    att_hat = treated_deltas.mean() - control_deltas.mean()
    se = abs(att_hat / t_result.statistic) if t_result.statistic != 0 else np.nan
    rejected = bool(t_result.pvalue < 0.05)

    return att_hat, se, rejected


def run_single_sim(params, seed):
    """Run one simulation: generate data and estimate ATT.

    Parameters
    ----------
    params : dict
        DGP parameters.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: att_hat, se, rejected, plus all params.
    """
    rng = np.random.default_rng(seed)
    panel = generate_panel(params, rng)
    att_hat, se, rejected = estimate_att_cs(panel)
    return {**params, 'seed': seed, 'att_hat': att_hat, 'se': se, 'rejected': rejected}


def main():
    """Load a panel CSV and run estimation."""
    parser = argparse.ArgumentParser(description='Estimate ATT using Callaway & Sant\'Anna')
    parser.add_argument('--panel', type=str, default='output/example_panel.csv',
                        help='Path to panel CSV file')
    args = parser.parse_args()

    print(f"Loading panel from: {args.panel}")
    panel = pd.read_csv(args.panel)

    print(f"Panel shape: {panel.shape}")
    print(f"Unique sites: {panel['site_id'].nunique()}")
    print(f"Treated sites: {panel[panel['cohort_g'] > 0]['site_id'].nunique()}")
    print(f"Control sites: {panel[panel['cohort_g'] == 0]['site_id'].nunique()}")

    att_hat, se, rejected = estimate_att_cs(panel)

    print(f"\n--- Callaway & Sant'Anna Estimation ---")
    print(f"ATT_hat:  {att_hat:.4f}")
    print(f"SE:       {se:.4f}")
    print(f"t-stat:   {att_hat / se:.4f}" if se > 0 else "t-stat:   N/A")
    print(f"Rejected: {rejected}")


if __name__ == '__main__':
    main()
