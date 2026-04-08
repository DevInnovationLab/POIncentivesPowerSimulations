"""Shared configuration for the power simulation."""

import os
import numpy as np
import pandas as pd

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
# Load sweep ranges from CSV
# ---------------------------------------------------------------------------
_SWEEP_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_params.csv')
_sweep_df = pd.read_csv(_SWEEP_CSV)
_SWEEP = {
    row['parameter']: [float(v) for v in row['values'].split(',')]
    for _, row in _sweep_df.iterrows()
}

# ---------------------------------------------------------------------------
# Single-state parameter grid (AP only, backward compatible)
# ---------------------------------------------------------------------------
# NOTE: target_att is the expected dynamic effect on outcomes (the estimand).
# tau is derived using the finite-horizon AR(1) amplification formula.
PARAM_GRID = {
    'mu_baseline': _SWEEP['mu_baseline'],
    'sigma_baseline': _SWEEP['sigma_baseline'],
    'target_att': _SWEEP['target_att'],
    'rho': _SWEEP['rho'],
    'h_init': _SWEEP['h_init'],
}

# ---------------------------------------------------------------------------
# Pooled (two-state) parameter grid
# ---------------------------------------------------------------------------
POOLED_PARAM_GRID = {
    'mu_baseline_ap': _SWEEP['mu_baseline_ap'],
    'mu_baseline_od': _SWEEP['mu_baseline_od'],
    'sigma_baseline': _SWEEP['sigma_baseline'],
    'target_att': _SWEEP['target_att'],
    'rho': _SWEEP['rho'],
    'h_init': _SWEEP['h_init'],
    'effect_ratio': _SWEEP['effect_ratio'],
}

# ---------------------------------------------------------------------------
# Comprehensive comparison grid
# ---------------------------------------------------------------------------
# Sweeps across: mode (AP-only vs pooled), n_measurements, study duration
STUDY_DURATIONS = {
    f'{int(w)}wk': int(w) for w in _SWEEP['study_end_week']
}
# Override with readable labels
STUDY_DURATIONS = {'6mo': 26, '1yr': 52, '1.5yr': 78}

COMPARISON_N_MEASUREMENTS = [int(v) for v in _SWEEP['n_measurements']]

COMPARISON_PARAM_GRID = {
    'mu_baseline': _SWEEP['mu_baseline_ap'],
    'sigma_baseline': _SWEEP['sigma_baseline'],
    'target_att': _SWEEP['target_att'],
    'rho': _SWEEP['rho'],
    'h_init': _SWEEP['h_init'],
}
