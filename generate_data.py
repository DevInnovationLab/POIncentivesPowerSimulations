"""Stage 1: Data Generating Process for staggered rollout RCT simulation."""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from config import (
    N_SITES, N_TREATED, TOTAL_MONITORING_WEEKS, TREATMENT_WEEKS,
    make_install_schedule,
)

# Relative week range for observation (training wk 5-8, treatment wk 9-24)
REL_WEEK_START = 5
REL_WEEK_END = 8 + TREATMENT_WEEKS  # training ends at wk 8, then treatment period


def generate_panel(params, rng):
    """Generate a simulated panel dataset for one draw of the DGP.

    Parameters
    ----------
    params : dict
        Keys: mu_baseline, sigma_baseline, rho, h_init, and either
        'tau' (per-period impulse) or 'target_att' (desired dynamic effect,
        from which tau is derived as target_att * (1 - rho)).
    rng : numpy.random.Generator
        Seeded RNG instance.

    Returns
    -------
    pd.DataFrame
        Columns: site_id, calendar_week, relative_week, Y, treated,
                 treatment_active, cohort_g, phase, theta_i, p_it
    """
    mu = params['mu_baseline']
    sigma = params['sigma_baseline']
    rho = params['rho']
    h_init = params['h_init']

    # Derive tau from target_att if provided, otherwise use tau directly.
    # Uses finite-horizon AR(1) amplification formula:
    #   At treatment week k (0-indexed), cumulative effect = tau * sum_{j=0}^{k} rho^j
    #   Average over T weeks: target_att = tau * amplification_factor
    #   amplification_factor = [T - rho*(1 - rho^T)/(1-rho)] / (T*(1-rho))  for rho != 0
    if 'target_att' in params:
        target_att = params['target_att']
        T = TREATMENT_WEEKS
        if rho == 0:
            tau = target_att
        else:
            amp = (T - rho * (1 - rho ** T) / (1 - rho)) / (T * (1 - rho))
            tau = target_att / amp
    else:
        tau = params['tau']

    install_weeks = make_install_schedule()  # length 50

    # Draw site-level baseline propensities from TruncNormal(mu, sigma, 0, 1)
    a, b = (0 - mu) / sigma, (1 - mu) / sigma
    theta = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=N_SITES,
                          random_state=rng)

    # Treatment assignment: random 25 treated
    treated_idx = np.zeros(N_SITES, dtype=bool)
    treated_sites = rng.choice(N_SITES, size=N_TREATED, replace=False)
    treated_idx[treated_sites] = True

    # Cohort g: calendar week when treatment starts (install_week + 8)
    cohort_g = np.where(treated_idx, install_weeks + 8, 0)

    # Vectorized panel generation: loop over 20 weeks, vectorize across 50 sites
    n_weeks = REL_WEEK_END - REL_WEEK_START + 1  # 20
    rel_weeks = np.arange(REL_WEEK_START, REL_WEEK_END + 1)

    # Pre-allocate arrays: (n_weeks, N_SITES)
    Y_all = np.empty((n_weeks, N_SITES))
    p_all = np.empty((n_weeks, N_SITES))

    y_prev = theta.copy()  # initialize AR(1) with baseline propensity

    for w_idx, rel_wk in enumerate(rel_weeks):
        # Hawthorne effect (scalar, same for all sites at this relative week)
        hawthorne = h_init * max(0.0, 1.0 - (rel_wk - REL_WEEK_START) / TOTAL_MONITORING_WEEKS)

        # Treatment effect: only for treated sites in treatment period
        tx_effect = np.where(treated_idx & (rel_wk >= 9), tau, 0.0)

        # AR(1) propensity (vectorized across all sites)
        p_it = np.clip(
            (1 - rho) * theta + rho * y_prev + tx_effect + hawthorne,
            0.0, 1.0,
        )

        # Three independent Bernoulli measurements per site
        measurements = rng.binomial(1, np.tile(p_it, (3, 1)))  # (3, N_SITES)
        Y_it = measurements.mean(axis=0)  # (N_SITES,)

        Y_all[w_idx] = Y_it
        p_all[w_idx] = p_it
        y_prev = Y_it

    # Build DataFrame from pre-computed arrays
    site_ids = np.tile(np.arange(N_SITES), n_weeks)
    rel_week_col = np.repeat(rel_weeks, N_SITES)
    cal_week_col = np.repeat(rel_weeks, N_SITES) + np.tile(install_weeks - 1, n_weeks)

    panel = pd.DataFrame({
        'site_id': site_ids,
        'calendar_week': cal_week_col.astype(int),
        'relative_week': rel_week_col.astype(int),
        'Y': Y_all.ravel(),
        'treated': np.tile(treated_idx.astype(int), n_weeks),
        'treatment_active': np.tile(treated_idx.astype(int), n_weeks) * (np.repeat(rel_weeks >= 9, N_SITES).astype(int)),
        'cohort_g': np.tile(cohort_g.astype(int), n_weeks),
        'phase': np.where(np.repeat(rel_weeks <= 8, N_SITES), 'training', 'treatment'),
        'theta_i': np.tile(theta, n_weeks),
        'p_it': p_all.ravel(),
    })
    return panel


def main():
    """Generate an example panel, save CSV and timeseries plot."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    default_params = {
        'mu_baseline': 0.4,
        'sigma_baseline': 0.15,
        'target_att': 0.15,
        'rho': 0.5,
        'h_init': 0.0,
    }
    rng = np.random.default_rng(42)
    panel = generate_panel(default_params, rng)

    # Save CSV
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    panel.to_csv('output/example_panel.csv', index=False)
    print(f"Saved example panel: {len(panel)} rows, {panel['site_id'].nunique()} sites")

    # Summary stats
    print("\n--- Summary Statistics ---")
    print(f"Sites: {panel['site_id'].nunique()}")
    print(f"Treated: {panel[panel['treated'] == 1]['site_id'].nunique()}")
    print(f"Control: {panel[panel['treated'] == 0]['site_id'].nunique()}")
    print(f"Calendar weeks: {panel['calendar_week'].min()} to {panel['calendar_week'].max()}")
    print(f"Mean Y (training, treated):   {panel[(panel['phase'] == 'training') & (panel['treated'] == 1)]['Y'].mean():.3f}")
    print(f"Mean Y (training, control):   {panel[(panel['phase'] == 'training') & (panel['treated'] == 0)]['Y'].mean():.3f}")
    print(f"Mean Y (treatment, treated):  {panel[(panel['phase'] == 'treatment') & (panel['treated'] == 1)]['Y'].mean():.3f}")
    print(f"Mean Y (treatment, control):  {panel[(panel['phase'] == 'treatment') & (panel['treated'] == 0)]['Y'].mean():.3f}")

    # Plot timeseries for 6 sites (3 treated, 3 control)
    treated_sites = panel[panel['treated'] == 1]['site_id'].unique()[:3]
    control_sites = panel[panel['treated'] == 0]['site_id'].unique()[:3]
    plot_sites = np.concatenate([treated_sites, control_sites])

    fig, ax = plt.subplots(figsize=(12, 6))
    for sid in plot_sites:
        site_data = panel[panel['site_id'] == sid].sort_values('calendar_week')
        is_treated = site_data['treated'].iloc[0] == 1
        label = f"Site {sid} ({'T' if is_treated else 'C'})"
        color = 'tab:blue' if is_treated else 'tab:orange'
        linestyle = '-' if is_treated else '--'
        ax.plot(site_data['calendar_week'], site_data['Y'],
                label=label, color=color, linestyle=linestyle, alpha=0.8)

        # Vertical line at treatment start for treated sites
        if is_treated:
            tx_start_cal = site_data[site_data['phase'] == 'treatment']['calendar_week'].min()
            ax.axvline(tx_start_cal, color=color, alpha=0.3, linestyle=':')

    ax.set_xlabel('Calendar Week')
    ax.set_ylabel('Y (proportion chlorinated)')
    ax.set_title('Example Panel: Site-level Timeseries')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig('output/plots/example_panel_timeseries.png', dpi=150)
    plt.close(fig)
    print("\nSaved plot: output/plots/example_panel_timeseries.png")


if __name__ == '__main__':
    main()
