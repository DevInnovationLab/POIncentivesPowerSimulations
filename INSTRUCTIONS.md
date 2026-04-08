# Running the Power Simulation

## Prerequisites

- Python 3.11.9
- Access to UChicago RCC Midway3 (ssd partition) for full runs

## Local Smoke Test

Verify everything works before submitting to HPC:

```bash
pip3 install -r requirements.txt
python3 run_comparison_sweep.py --n_sims 5 --n_workers 2
python3 visualize_comparison.py
```

This runs 5 sims per combo across all 6,480 tasks (~2-3 min). Check `output/plots/` for generated figures.

## Transfer to HPC

```bash
tar czf posim.tar.gz --exclude='output' --exclude='__pycache__' --exclude='logs' --exclude='.git' --exclude='output.tar.gz' .
scp posim.tar.gz <user>@midway3.rcc.uchicago.edu:~/POIncentivesPowerSim/
```

## HPC Setup (one time)

```bash
ssh <user>@midway3.rcc.uchicago.edu
mkdir -p ~/POIncentivesPowerSim && cd ~/POIncentivesPowerSim
tar xzf posim.tar.gz
module load python/3.11.9
pip3 install --user -r requirements.txt
```

## Run the Full Comparison Sweep

This sweeps across 12 scenarios (AP-only vs pooled, 2 vs 3 measurements/week, 6mo/1yr/1.5yr duration) x 540 parameter combinations x 1000 simulations each.

```bash
cd ~/POIncentivesPowerSim
sbatch submit_hpc.sh --comparison
```

This submits 5 array jobs (one per node, 48 cores each, 180GB RAM, 4hr wall time).

Monitor progress:

```bash
squeue -u $USER
```

After all 5 jobs show as COMPLETED:

```bash
module load python/3.11.9
python3 run_comparison_sweep.py --merge_chunks --n_chunks 5
python3 visualize_comparison.py
```

## Retrieve Results

On HPC:

```bash
cd ~/POIncentivesPowerSim
tar czf output.tar.gz output/
```

On your laptop:

```bash
scp <user>@midway3.rcc.uchicago.edu:~/POIncentivesPowerSim/output.tar.gz .
tar xzf output.tar.gz
```

## Output Files

After visualization, `output/` contains:

| File | Description |
|------|-------------|
| `comparison_results.csv` | Raw power results for all 6,480 scenario-param combinations |
| `comparison_mde_table.csv` | MDE at 80% power for each combination |
| `plots/mde_over_time_rho*.png` | MDE trajectories across 6mo/1yr/1.5yr by persistence level |
| `plots/mde_comparison_bars.png` | Bar chart comparing MDE across all scenarios at 1.5yr |
| `plots/comparison_power_curves_rho*.png` | Power curves by duration, AP-only vs pooled |

## Other Sweep Modes

Single-state only (AP, original parameter grid):

```bash
sbatch submit_hpc.sh
# After completion:
python3 run_power_sweep.py --merge_chunks --n_chunks 5
python3 visualize.py
```

Pooled only (AP + Odisha, with effect ratio sweep):

```bash
sbatch submit_hpc.sh --pooled
# After completion:
python3 run_power_sweep.py --pooled --merge_chunks --n_chunks 5
python3 visualize.py --pooled
```
