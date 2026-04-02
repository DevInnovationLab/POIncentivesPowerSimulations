"""Stage 1: Data Generating Process for staggered rollout RCT simulation.

Supports single-state and multi-state (pooled) panel generation with
variable treatment durations per site.
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from config import (
    N_MEASUREMENTS, STUDY_END_WEEK, TRAINING_WEEKS, STABILIZATION_WEEKS,
    TRAINING_REL_START, TREATMENT_REL_START,
    AP_CONFIG, make_install_weeks, mean_treatment_weeks,
)


def compute_tau(target_att, rho, treatment_weeks):
    """Derive per-period impulse tau from target ATT using finite-horizon AR(1) formula.

    Parameters
    ----------
    target_att : float
        Desired average dynamic treatment effect on outcomes.
    rho : float
        AR(1) behavioral persistence parameter.
    treatment_weeks : float
        Number of treatment weeks (can be average across sites).

    Returns
    -------
    float
        Per-period treatment impulse tau.
    """
    if target_att == 0:
        return 0.0
    T = treatment_weeks
    if rho == 0 or T <= 0:
        return target_att
    amp = (T - rho * (1 - rho ** T) / (1 - rho)) / (T * (1 - rho))
    return target_att / amp


def generate_state_panel(params, rng, state_config, state_label,
                         site_id_offset=0):
    """Generate a panel dataset for one state.

    Parameters
    ----------
    params : dict
        Keys: mu_baseline, sigma_baseline, rho, h_init, and either 'tau' or
        'target_att'.
    rng : numpy.random.Generator
    state_config : dict
        State configuration (from config.py).
    state_label : str
        Label for the state column.
    site_id_offset : int
        Offset for site_id to ensure unique IDs across states.

    Returns
    -------
    pd.DataFrame
    """
    mu = params['mu_baseline']
    sigma = params['sigma_baseline']
    rho = params['rho']
    h_init = params['h_init']
    n_measurements = params.get('n_measurements', N_MEASUREMENTS)
    study_end_week = params.get('study_end_week', STUDY_END_WEEK)
    n_sites = state_config['n_sites']
    n_treated = state_config['n_treated']

    install_weeks = make_install_weeks(state_config)
    avg_tx_weeks = mean_treatment_weeks(state_config, study_end_week)

    # Total monitoring weeks for Hawthorne decay (site-specific)
    # Each site's monitoring runs from training start to study end
    site_monitoring_weeks = study_end_week - (install_weeks + STABILIZATION_WEEKS)

    # Derive tau
    if 'tau' in params:
        tau = params['tau']
    elif 'target_att' in params:
        tau = compute_tau(params['target_att'], rho, avg_tx_weeks)
    else:
        tau = 0.0

    # Draw site-level baseline propensities
    a, b = (0 - mu) / sigma, (1 - mu) / sigma
    theta = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_sites,
                          random_state=rng)

    # Treatment assignment
    treated_idx = np.zeros(n_sites, dtype=bool)
    treated_sites = rng.choice(n_sites, size=n_treated, replace=False)
    treated_idx[treated_sites] = True

    cohort_g = np.where(treated_idx, install_weeks + STABILIZATION_WEEKS + TRAINING_WEEKS, 0)

    # Determine observation window per site: training start to study end
    # Relative weeks: 5 (training start) to whatever fits before study_end
    site_rel_end = study_end_week - install_weeks + 1  # max relative week per site
    max_rel_end = int(site_rel_end.max())

    # Generate week by week, vectorized across sites
    all_rows = []
    y_prev = theta.copy()

    for rel_wk in range(TRAINING_REL_START, max_rel_end + 1):
        # Which sites are still being observed at this relative week?
        cal_wk_arr = install_weeks + rel_wk - 1
        active = cal_wk_arr <= study_end_week

        if not active.any():
            break

        # Hawthorne: decays linearly from h_init to 0 over each site's monitoring window
        weeks_since_monitor_start = rel_wk - TRAINING_REL_START
        safe_monitoring = np.where(site_monitoring_weeks > 0, site_monitoring_weeks, 1.0)
        hawthorne = np.where(
            site_monitoring_weeks > 0,
            h_init * np.maximum(0.0, 1.0 - weeks_since_monitor_start / safe_monitoring),
            0.0,
        )

        # Treatment effect
        is_treatment_period = rel_wk >= TREATMENT_REL_START
        tx_effect = np.where(treated_idx & is_treatment_period & active, tau, 0.0)

        # AR(1) propensity
        p_it = np.clip(
            (1 - rho) * theta + rho * y_prev + tx_effect + hawthorne,
            0.0, 1.0,
        )

        # Bernoulli measurements
        measurements = rng.binomial(1, np.tile(p_it, (n_measurements, 1)))
        Y_it = measurements.mean(axis=0)

        # Store rows only for active sites
        phase = 'training' if rel_wk < TREATMENT_REL_START else 'treatment'
        for i in np.where(active)[0]:
            all_rows.append((
                i + site_id_offset,
                int(cal_wk_arr[i]),
                int(rel_wk),
                Y_it[i],
                int(treated_idx[i]),
                int(treated_idx[i] and is_treatment_period),
                int(cohort_g[i]),
                phase,
                state_label,
                theta[i],
                p_it[i],
            ))

        y_prev = Y_it

    panel = pd.DataFrame(all_rows, columns=[
        'site_id', 'calendar_week', 'relative_week', 'Y', 'treated',
        'treatment_active', 'cohort_g', 'phase', 'state', 'theta_i', 'p_it',
    ])
    return panel


def generate_panel(params, rng):
    """Generate a single-state (AP) panel. Backward compatible wrapper."""
    return generate_state_panel(params, rng, AP_CONFIG, 'AP')


def generate_pooled_panel(params, rng):
    """Generate a two-state (AP + Odisha) pooled panel.

    Parameters
    ----------
    params : dict
        Keys: mu_baseline_ap, mu_baseline_od, sigma_baseline, target_att,
        rho, h_init, effect_ratio.
    rng : numpy.random.Generator

    Returns
    -------
    pd.DataFrame
        Combined panel with 'state' column.
    """
    from config import ODISHA_CONFIG

    rho = params['rho']
    effect_ratio = params.get('effect_ratio', 1.0)

    # Pass through n_measurements and study_end_week if present
    extra = {}
    if 'n_measurements' in params:
        extra['n_measurements'] = params['n_measurements']
    if 'study_end_week' in params:
        extra['study_end_week'] = params['study_end_week']

    # AP params
    ap_params = {
        'mu_baseline': params['mu_baseline_ap'],
        'sigma_baseline': params['sigma_baseline'],
        'target_att': params['target_att'],
        'rho': rho,
        'h_init': params['h_init'],
        **extra,
    }

    # Odisha params: different baseline, scaled treatment effect
    od_params = {
        'mu_baseline': params['mu_baseline_od'],
        'sigma_baseline': params['sigma_baseline'],
        'target_att': params['target_att'] * effect_ratio,
        'rho': rho,
        'h_init': params['h_init'],
        **extra,
    }

    ap_panel = generate_state_panel(ap_params, rng, AP_CONFIG, 'AP',
                                     site_id_offset=0)
    od_panel = generate_state_panel(od_params, rng, ODISHA_CONFIG, 'Odisha',
                                     site_id_offset=AP_CONFIG['n_sites'])

    return pd.concat([ap_panel, od_panel], ignore_index=True)


def main():
    """Generate example panels, save CSVs and timeseries plots."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)

    rng = np.random.default_rng(42)

    # --- Single-state (AP) example ---
    ap_params = {
        'mu_baseline': 0.4, 'sigma_baseline': 0.15,
        'target_att': 0.15, 'rho': 0.5, 'h_init': 0.0,
    }
    panel_ap = generate_panel(ap_params, rng)
    panel_ap.to_csv('output/example_panel.csv', index=False)

    print(f"AP panel: {len(panel_ap)} rows, {panel_ap['site_id'].nunique()} sites")
    print(f"  Calendar weeks: {panel_ap['calendar_week'].min()}-{panel_ap['calendar_week'].max()}")
    tx_wks = panel_ap[panel_ap['phase'] == 'treatment'].groupby('site_id').size()
    print(f"  Treatment weeks per site: {tx_wks.min()}-{tx_wks.max()} (mean {tx_wks.mean():.0f})")

    # --- Pooled (AP + Odisha) example ---
    rng2 = np.random.default_rng(42)
    pooled_params = {
        'mu_baseline_ap': 0.4, 'mu_baseline_od': 0.3,
        'sigma_baseline': 0.15, 'target_att': 0.15,
        'rho': 0.5, 'h_init': 0.0, 'effect_ratio': 1.0,
    }
    panel_pooled = generate_pooled_panel(pooled_params, rng2)
    panel_pooled.to_csv('output/example_panel_pooled.csv', index=False)

    for state in ['AP', 'Odisha']:
        sp = panel_pooled[panel_pooled['state'] == state]
        tx_wks = sp[sp['phase'] == 'treatment'].groupby('site_id').size()
        print(f"\n{state} panel: {len(sp)} rows, {sp['site_id'].nunique()} sites")
        print(f"  Calendar weeks: {sp['calendar_week'].min()}-{sp['calendar_week'].max()}")
        print(f"  Treatment weeks per site: {tx_wks.min()}-{tx_wks.max()} (mean {tx_wks.mean():.0f})")

    # --- Plot pooled timeseries ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, state in zip(axes, ['AP', 'Odisha']):
        sp = panel_pooled[panel_pooled['state'] == state]
        treated_ids = sp[sp['treated'] == 1]['site_id'].unique()[:3]
        control_ids = sp[sp['treated'] == 0]['site_id'].unique()[:3]
        for sid in np.concatenate([treated_ids, control_ids]):
            sd = sp[sp['site_id'] == sid].sort_values('calendar_week')
            is_t = sd['treated'].iloc[0] == 1
            ax.plot(sd['calendar_week'], sd['Y'],
                    color='tab:blue' if is_t else 'tab:orange',
                    linestyle='-' if is_t else '--', alpha=0.8,
                    label=f"Site {sid} ({'T' if is_t else 'C'})")
            if is_t:
                tx_cal = sd[sd['phase'] == 'treatment']['calendar_week'].min()
                ax.axvline(tx_cal, color='tab:blue', alpha=0.2, linestyle=':')
        ax.set_title(state)
        ax.set_xlabel('Calendar Week')
        ax.set_ylabel('Y (proportion chlorinated)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Pooled Panel: AP + Odisha', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig('output/plots/example_panel_pooled.png', dpi=150)
    plt.close(fig)
    print("\nSaved: output/example_panel.csv, output/example_panel_pooled.csv")
    print("Saved: output/plots/example_panel_pooled.png")


if __name__ == '__main__':
    main()
