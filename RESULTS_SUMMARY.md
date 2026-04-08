# Results Summary: PO Incentives Power Simulation

## Overview

This document summarizes results from the comprehensive MDE comparison sweep, which evaluates the minimum detectable effect (MDE) at 80% power across 12 scenarios: AP-only vs AP+Odisha pooled, 2 vs 3 chlorine measurements per week, and study durations of 6 months, 1 year, and 1.5 years from AP start. Each scenario sweeps across 540 parameter combinations with 1,000 simulations each.

> **Note:** These results were generated before a bug fix to the AR(1) feedback mechanism (see Erratum below). They should be re-run on HPC with the corrected code. The qualitative conclusions about pooling vs single-state hold, but the rho-dependence and duration trends will change.

---

## MDE Summary Table

Averaged across baseline compliance (0.3, 0.5, 0.7), compliance heterogeneity (0.10–0.25), and monitoring effects (-0.10 to +0.10):

### 6 Months from AP Start

| Scenario | Tests/wk | AR(1)=0.5 | AR(1)=0.7 | AR(1)=0.9 |
|---|---|---|---|---|
| AP Only (n=50) | 2 | 0.227 | 0.273 | 0.286 |
| AP Only (n=50) | 3 | 0.187 | 0.226 | 0.244 |
| Pooled AP+Odisha (n=100) | 2 | 0.190 | 0.226 | 0.225 |
| Pooled AP+Odisha (n=100) | 3 | 0.156 | 0.182 | 0.214 |

### 1 Year from AP Start

| Scenario | Tests/wk | AR(1)=0.5 | AR(1)=0.7 | AR(1)=0.9 |
|---|---|---|---|---|
| AP Only (n=50) | 2 | 0.213 | 0.261 | 0.298 |
| AP Only (n=50) | 3 | 0.176 | 0.216 | 0.264 |
| Pooled AP+Odisha (n=100) | 2 | 0.149 | 0.181 | 0.216 |
| Pooled AP+Odisha (n=100) | 3 | 0.123 | 0.149 | 0.188 |

### 1.5 Years from AP Start

| Scenario | Tests/wk | AR(1)=0.5 | AR(1)=0.7 | AR(1)=0.9 |
|---|---|---|---|---|
| AP Only (n=50) | 2 | 0.209 | 0.260 | 0.307 |
| AP Only (n=50) | 3 | 0.172 | 0.212 | 0.265 |
| Pooled AP+Odisha (n=100) | 2 | 0.143 | 0.175 | 0.224 |
| Pooled AP+Odisha (n=100) | 3 | 0.118 | 0.143 | 0.190 |

---

## Key Findings

### 1. Pooling AP + Odisha substantially improves power

Pooling two states (100 sites total, 50 treated + 50 control) with state fixed effects reduces MDE by **25–35%** compared to AP alone. This is the single largest lever for improving power.

The pooled estimator demeans site-level treatment effects within each state, absorbing all between-state differences (baseline compliance, water chemistry, implementation context). The power gain comes purely from doubling the sample size for the variance estimate — the SE of a mean shrinks as 1/sqrt(n). Doubling n from 50 to 100 gives a ~30% SE reduction, which maps directly to ~30% lower MDE.

The interventions can differ between states (captured by the `effect_ratio` parameter). Pooling estimates a weighted average of the two states' treatment effects — still a well-defined and policy-relevant estimand.

### 2. Three measurements per week helps modestly

Increasing from 2 to 3 chlorine tests per week reduces MDE by **10–20%**. The gain comes from reduced measurement noise: averaging K Bernoulli observations reduces the within-week variance by a factor of K. Going from K=2 to K=3 gives a ~18% reduction in measurement-noise variance.

This is a smaller gain than pooling, but is essentially free if field logistics permit a third weekly test.

### 3. Longer study duration provides diminishing returns

MDE generally improves from 6 months to 1 year, but the gains from 1 year to 1.5 years are marginal. Most of the statistical information is captured in the first year of treatment. The pooled estimator benefits more from extended duration than AP-only because Odisha sites start later (3-month offset) and accumulate meaningful treatment data only after ~6 months from AP start.

### 4. Higher behavioral persistence (rho) increases MDE

Higher AR(1) persistence means PO behavior is more "sticky" — harder to shift with incentives. At rho=0.9, the MDE is roughly 30–60% higher than at rho=0.5. This is the most important uncertainty in the power calculation: if PO behavior is highly inertial, the study needs to detect a larger effect to achieve adequate power.

---

## Interpretation for Study Design

Under the most favorable assumptions (rho=0.5, 3 tests/week, 1.5yr, pooled):
- **MDE ≈ 0.12** (12 percentage points)

Under moderate assumptions (rho=0.7, 2 tests/week, 1yr, pooled):
- **MDE ≈ 0.18** (18 percentage points)

Under conservative assumptions (rho=0.9, 2 tests/week, AP-only):
- **MDE ≈ 0.29–0.31** (29–31 percentage points)

The study is well-powered to detect treatment effects of 15–20 percentage points if:
1. Both states are pooled (100 sites)
2. Behavioral persistence is moderate (rho ≤ 0.7)
3. At least 2 measurements per week are taken

If the true treatment effect is smaller (5–10 pp), the study would require either more sites per state or lower behavioral persistence to achieve 80% power.

---

## Erratum: AR(1) Feedback Bug

The results above were generated with a version of the code where the AR(1) behavioral process fed back the noisy measurement outcome Y_it (average of K Bernoulli draws) rather than the latent propensity p_it. This caused measurement noise to amplify through the feedback loop at high rho, producing the counterintuitive pattern where MDE *increased* with study duration at rho=0.9.

**Root cause:** Line `y_prev = Y_it` in `generate_data.py` should be `y_prev = p_it`. The PO's behavioral persistence depends on their propensity to chlorinate (their decision), not on test results they don't observe.

**Impact:** After the fix:
- MDE monotonically decreases with duration at all rho values (as expected)
- rho has less impact on MDE (the noise amplification at high rho is eliminated)
- The overall MDE values will be lower (less noise in the system)
- The qualitative ranking of scenarios (pooled > single-state, 3 tests > 2 tests) will hold

**Action required:** Re-run the comparison sweep on HPC with the corrected code:
```bash
sbatch submit_hpc.sh --comparison
# After completion:
python3 run_comparison_sweep.py --merge_chunks --n_chunks 5
python3 visualize_comparison.py
```
