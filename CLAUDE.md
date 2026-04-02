# PO Incentives Power Simulation

## Overview
Power simulation for a staggered rollout RCT across 50 villages, estimating minimum detectable effect (MDE) at 80% power for pump operator chlorination incentives. Uses site-level DiD (Callaway & Sant'Anna style). **No TWFE.**

## Running
```bash
pip install -r requirements.txt
python generate_data.py
python estimate.py --panel output/example_panel.csv
python run_power_sweep.py --n_sims 1000 --hpc --chunk_id $SLURM_ARRAY_TASK_ID --n_chunks 5
python run_power_sweep.py --merge_chunks --n_chunks 5
python visualize.py
```

## Key Parameters
- `target_att`: the expected dynamic effect on chlorination rates (the estimand)
- `tau` (per-period impulse): derived from target_att using finite-horizon AR(1) formula
- Treatment period: 48 weeks (relative weeks 9–56)
- Training/baseline: 4 weeks (relative weeks 5–8)

## File Structure
- `config.py` — Parameter grids, constants, `make_install_schedule()`
- `generate_data.py` — DGP: `generate_panel(params, rng)` → DataFrame
- `estimate.py` — Site-level DiD: `estimate_att_cs(panel)` → (att, se, rejected)
- `run_power_sweep.py` — Parallel sweep with `--hpc` mode for SLURM
- `visualize.py` — Power curves, heatmaps, MDE table
