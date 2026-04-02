"""Shared configuration for the power simulation."""

import numpy as np

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
N_MEASUREMENTS = 2          # independent chlorine measurements per week
STUDY_END_WEEK = 78         # calendar week when data collection ends (18 months from AP start)
N_SIMS = 1000
SEED = 42

# Phase durations (in weeks, relative to installation)
STABILIZATION_WEEKS = 4
TRAINING_WEEKS = 4
TRAINING_REL_START = STABILIZATION_WEEKS + 1  # relative week 5
TREATMENT_REL_START = TRAINING_REL_START + TRAINING_WEEKS  # relative week 9

# ---------------------------------------------------------------------------
# State configurations
# ---------------------------------------------------------------------------
AP_CONFIG = {
    'name': 'AP',
    'install_schedule': [2] + [6] * 8,   # 50 villages over 9 weeks
    'cal_week_offset': 0,                  # starts at calendar week 1
    'n_sites': 50,
    'n_treated': 25,
}

ODISHA_CONFIG = {
    'name': 'Odisha',
    'install_schedule': [2] * 25,          # 50 villages, 2/week over 25 weeks
    'cal_week_offset': 12,                 # starts 3 months (13 weeks) after AP; first install at cal week 13
    'n_sites': 50,
    'n_treated': 25,
}


def make_install_weeks(state_config):
    """Returns array of installation calendar weeks for a state."""
    weeks = []
    for week_idx, n_villages in enumerate(state_config['install_schedule']):
        cal_week = state_config['cal_week_offset'] + week_idx + 1
        weeks.extend([cal_week] * n_villages)
    return np.array(weeks)


def mean_treatment_weeks(state_config, study_end_week=None):
    """Average treatment duration (weeks) for sites in a state."""
    if study_end_week is None:
        study_end_week = STUDY_END_WEEK
    install_wks = make_install_weeks(state_config)
    treatment_starts = install_wks + STABILIZATION_WEEKS + TRAINING_WEEKS
    durations = study_end_week - treatment_starts
    durations = np.maximum(durations, 0)
    return durations.mean()


# ---------------------------------------------------------------------------
# Single-state parameter grid (AP only, backward compatible)
# ---------------------------------------------------------------------------
# NOTE: target_att is the expected dynamic effect on outcomes (the estimand).
# tau is derived using the finite-horizon AR(1) amplification formula.
PARAM_GRID = {
    'mu_baseline': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'sigma_baseline': [0.10, 0.15, 0.20, 0.25],
    'target_att': [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40],
    'rho': [0.5, 0.7, 0.9],
    'h_init': [-0.10, -0.05, 0.0, 0.05, 0.10],
}

# ---------------------------------------------------------------------------
# Pooled (two-state) parameter grid
# ---------------------------------------------------------------------------
POOLED_PARAM_GRID = {
    'mu_baseline_ap': [0.3, 0.5, 0.7],
    'mu_baseline_od': [0.3, 0.5, 0.7],
    'sigma_baseline': [0.10, 0.20],
    'target_att': [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40],
    'rho': [0.5, 0.7, 0.9],
    'h_init': [-0.05, 0.0, 0.05],
    'effect_ratio': [0.5, 1.0, 1.5],
}

# ---------------------------------------------------------------------------
# Comprehensive comparison grid
# ---------------------------------------------------------------------------
# Sweeps across: mode (AP-only vs pooled), n_measurements (2 vs 3),
# study duration (6mo, 1yr, 1.5yr from AP start)
STUDY_DURATIONS = {
    '6mo': 26,    # 26 weeks from AP start
    '1yr': 52,    # 52 weeks
    '1.5yr': 78,  # 78 weeks (full study)
}

COMPARISON_N_MEASUREMENTS = [2, 3]

# Reduced grid for the comparison sweep (keeps total combos manageable)
COMPARISON_PARAM_GRID = {
    'mu_baseline': [0.3, 0.5, 0.7],
    'sigma_baseline': [0.10, 0.20],
    'target_att': [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40],
    'rho': [0.5, 0.7, 0.9],
    'h_init': [-0.05, 0.0, 0.05],
}
