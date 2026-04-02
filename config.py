"""Shared configuration for the power simulation."""

import numpy as np

# Parameter grids
# NOTE: target_att is the expected dynamic effect on outcomes (the estimand).
# The per-period impulse tau is derived using the finite-horizon AR(1)
# amplification formula in generate_data.py (NOT the steady-state tau = target_att * (1-rho)).
PARAM_GRID = {
    'mu_baseline': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'sigma_baseline': [0.10, 0.15, 0.20, 0.25],
    'target_att': [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40],
    'rho': [0.5, 0.7, 0.9],
    'h_init': [-0.10, -0.05, 0.0, 0.05, 0.10],
}

N_SITES = 50
N_TREATED = 25
N_SIMS = 1000
SEED = 42
INSTALL_SCHEDULE = [2] + [6] * 8  # villages per calendar week (total=50)

# Phase durations (in weeks, relative to installation)
STABILIZATION_WEEKS = 4
TRAINING_WEEKS = 4
TREATMENT_WEEKS = 48
TOTAL_MONITORING_WEEKS = TRAINING_WEEKS + TREATMENT_WEEKS  # 52


def make_install_schedule():
    """Returns array of 50 installation calendar weeks."""
    weeks = []
    for cal_week_idx, n_villages in enumerate(INSTALL_SCHEDULE):
        weeks.extend([cal_week_idx + 1] * n_villages)
    return np.array(weeks)
