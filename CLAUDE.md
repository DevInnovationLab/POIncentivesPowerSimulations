# PO Incentives Power Simulation

## Overview
Power simulation for a staggered rollout RCT across villages in AP and Odisha, estimating minimum detectable effect (MDE) at 80% power for pump operator chlorination incentives. Uses site-level DiD (Callaway & Sant'Anna style). **No TWFE.**

## Running
```bash
pip install -r requirements.txt
python3 generate_data.py
python3 estimate.py --panel output/example_panel.csv
python3 run_power_sweep.py --n_sims 1000 --hpc --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 5
python3 run_power_sweep.py --merge_chunks --n_chunks 5
python3 visualize.py
```

## Key Parameters
- All sweep ranges are defined in `sweep_params.csv` — edit this file to change what gets swept
- `target_att`: the expected dynamic effect on chlorination rates (the estimand)
- `tau` (per-period impulse): derived from target_att using finite-horizon AR(1) formula
- Treatment period: variable per site (determined by install date and study_end_week)
- Training/baseline: 4 weeks (relative weeks 5–8)

## File Structure
- `sweep_params.csv` — Single source of truth for all parameter sweep ranges
- `config.py` — Loads sweep_params.csv, parameter grids, constants, install schedules
- `generate_data.py` — DGP: single-state and pooled multi-state panels
- `estimate.py` — Site-level DiD: single-state and pooled with state FE
- `run_power_sweep.py` — Parallel sweep with `--pooled` and `--hpc` modes
- `run_comparison_sweep.py` — Comprehensive comparison sweep (AP vs pooled, measurements, durations)
- `visualize.py` — Single-state and pooled plots and MDE tables
- `visualize_comparison.py` — Comparison plots: MDE over time, power gain from pooling
- `submit_hpc.sh` — SLURM submission (supports `--comparison`, `--pooled`)

## Rules
- **No TWFE** — only use Callaway & Sant'Anna style DiD
- **Don't run large simulations locally** — use HPC for full sweeps
- **Always keep README.md and INSTRUCTIONS.md up to date** when making changes to the codebase, parameters, or run procedures
- **Commit with descriptive messages** after significant code changes
