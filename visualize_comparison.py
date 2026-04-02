#!/usr/bin/env python3
"""Stage 4b: Visualize comprehensive MDE comparison results.

Generates:
  - MDE comparison tables (AP-only vs pooled, 2 vs 3 measurements, by duration)
  - MDE over time plots (6mo, 1yr, 1.5yr)
  - Power curve comparison plots

Usage:
    python visualize_comparison.py [--input output/comparison_results.csv]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

SAVE_DPI = 150

LABELS = {
    'target_att': 'Target Effect on Chlorination Rate',
    'mu_baseline': 'Baseline Compliance Rate',
    'sigma_baseline': 'Compliance Heterogeneity (SD)',
    'rho': 'Behavioral Persistence (AR1)',
    'h_init': 'Initial Monitoring Effect',
    'power': 'Statistical Power',
    'mde': 'Min. Detectable Effect (pp)',
}

DURATION_ORDER = ['6mo', '1yr', '1.5yr']
DURATION_NICE = {'6mo': '6 Months', '1yr': '1 Year', '1.5yr': '1.5 Years'}
MODE_NICE = {'ap_only': 'AP Only (n=50)', 'pooled': 'Pooled AP+Odisha (n=100)'}


def interpolate_mde(grp, power_threshold=0.80):
    """Find MDE via linear interpolation for a sorted group."""
    grp = grp.sort_values('target_att')
    target_atts = grp['target_att'].values
    powers = grp['power'].values

    if (powers >= power_threshold).any():
        idx = np.where(powers >= power_threshold)[0][0]
        if idx == 0:
            return target_atts[0]
        p_lo, p_hi = powers[idx - 1], powers[idx]
        t_lo, t_hi = target_atts[idx - 1], target_atts[idx]
        if p_hi > p_lo:
            return t_lo + (power_threshold - p_lo) / (p_hi - p_lo) * (t_hi - t_lo)
        return t_hi
    return np.nan


def compute_mde_comparison(df):
    """Compute MDE for each (mode, n_measurements, duration, param combo)."""
    scenario_cols = ['mode', 'n_measurements', 'duration_label']
    param_cols = ['mu_baseline', 'sigma_baseline', 'rho', 'h_init']
    group_cols = scenario_cols + param_cols

    records = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row['mde'] = interpolate_mde(grp)
        records.append(row)

    return pd.DataFrame(records)


def plot_mde_over_time(mde_df, output_dir):
    """MDE over time (6mo, 1yr, 1.5yr), comparing modes and measurement frequencies.

    One figure per rho value. Lines show average MDE across param combos.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    rho_vals = sorted(mde_df['rho'].unique())

    # Scenario key = (mode, n_measurements)
    scenario_styles = {
        ('ap_only', 2):  {'color': 'tab:blue',  'linestyle': '-',  'marker': 'o', 'label': 'AP only, 2 tests/wk'},
        ('ap_only', 3):  {'color': 'tab:blue',  'linestyle': '--', 'marker': 's', 'label': 'AP only, 3 tests/wk'},
        ('pooled', 2):   {'color': 'tab:red',   'linestyle': '-',  'marker': 'o', 'label': 'Pooled, 2 tests/wk'},
        ('pooled', 3):   {'color': 'tab:red',   'linestyle': '--', 'marker': 's', 'label': 'Pooled, 3 tests/wk'},
    }

    for rho in rho_vals:
        fig, ax = plt.subplots(figsize=(8, 5))

        sub = mde_df[mde_df['rho'] == rho]
        # Average across mu_baseline, sigma_baseline, h_init
        avg = sub.groupby(['mode', 'n_measurements', 'duration_label'])['mde'].mean().reset_index()

        for (mode, n_meas), style in scenario_styles.items():
            sc = avg[(avg['mode'] == mode) & (avg['n_measurements'] == n_meas)]
            # Order by duration
            sc = sc.set_index('duration_label').reindex(DURATION_ORDER).reset_index()
            x_labels = [DURATION_NICE[d] for d in sc['duration_label']]

            ax.plot(x_labels, sc['mde'], **style, linewidth=2, markersize=7)

        ax.set_xlabel('Study Duration (from AP start)')
        ax.set_ylabel(LABELS['mde'])
        ax.set_title(f'MDE Over Time\n{LABELS["rho"]}={rho}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'mde_over_time_rho{rho}.png'), dpi=SAVE_DPI)
        plt.close(fig)

    print(f"  MDE over time plots saved ({len(rho_vals)} figures)")


def plot_mde_comparison_bars(mde_df, output_dir):
    """Bar chart: MDE by scenario, at 1.5yr duration, grouped by rho.

    Shows AP-only vs pooled, 2 vs 3 measurements.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    sub = mde_df[mde_df['duration_label'] == '1.5yr']
    avg = sub.groupby(['mode', 'n_measurements', 'rho'])['mde'].mean().reset_index()
    avg = avg.dropna(subset=['mde'])

    if avg.empty:
        print("  WARNING: No combos achieved 80% power at 1.5yr; skipping comparison bars.")
        return

    rho_vals = sorted(avg['rho'].unique())
    scenarios = [
        ('ap_only', 2, 'AP, 2/wk', 'tab:blue'),
        ('ap_only', 3, 'AP, 3/wk', 'cornflowerblue'),
        ('pooled', 2, 'Pooled, 2/wk', 'tab:red'),
        ('pooled', 3, 'Pooled, 3/wk', 'salmon'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rho_vals))
    n_sc = len(scenarios)
    w = 0.8 / n_sc

    for i, (mode, n_meas, label, color) in enumerate(scenarios):
        vals = []
        for rho in rho_vals:
            row = avg[(avg['mode'] == mode) & (avg['n_measurements'] == n_meas) & (avg['rho'] == rho)]
            vals.append(row['mde'].values[0] if len(row) > 0 else np.nan)

        offset = (i - (n_sc - 1) / 2) * w
        bars = ax.bar(x + offset, vals, width=w, label=label,
                      color=color, edgecolor='black', linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'AR(1)={r}' for r in rho_vals])
    ax.set_ylabel(LABELS['mde'])
    ax.set_title('MDE Comparison at 1.5 Years\n'
                 '(averaged across baseline, heterogeneity, and monitoring effects)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'mde_comparison_bars.png'), dpi=SAVE_DPI)
    plt.close(fig)
    print("  MDE comparison bar chart saved")


def plot_power_curves_comparison(df, output_dir):
    """Power curves for each duration, comparing AP-only vs pooled at n_meas=2.

    One figure per (rho, duration), averaged across other params.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    rho_vals = sorted(df['rho'].unique())
    count = 0

    for rho in rho_vals:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        for col_idx, dur in enumerate(DURATION_ORDER):
            ax = axes[col_idx]
            sub = df[(df['rho'] == rho) & (df['duration_label'] == dur)]

            for mode, n_meas, style in [
                ('ap_only', 2, {'color': 'tab:blue', 'linestyle': '-', 'label': 'AP, 2/wk'}),
                ('ap_only', 3, {'color': 'tab:blue', 'linestyle': '--', 'label': 'AP, 3/wk'}),
                ('pooled', 2, {'color': 'tab:red', 'linestyle': '-', 'label': 'Pooled, 2/wk'}),
                ('pooled', 3, {'color': 'tab:red', 'linestyle': '--', 'label': 'Pooled, 3/wk'}),
            ]:
                sc = sub[(sub['mode'] == mode) & (sub['n_measurements'] == n_meas)]
                avg = sc.groupby('target_att')['power'].mean().reset_index().sort_values('target_att')
                ax.plot(avg['target_att'], avg['power'], marker='o', markersize=3,
                        linewidth=1.5, **style)

            ax.axhline(0.80, color='grey', linestyle='--', linewidth=1)
            ax.set_title(f'{DURATION_NICE[dur]}')
            ax.set_xlabel(LABELS['target_att'])
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel(LABELS['power'])
                ax.legend(fontsize=8, loc='lower right')

        fig.suptitle(f'Power Curves by Duration — {LABELS["rho"]}={rho}',
                     fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(plots_dir, f'comparison_power_curves_rho{rho}.png'), dpi=SAVE_DPI)
        plt.close(fig)
        count += 1

    print(f"  Comparison power curve plots saved ({count} figures)")


def print_mde_summary_table(mde_df):
    """Print a concise MDE summary table to stdout."""
    # Average across mu_baseline, sigma_baseline, h_init
    avg = mde_df.groupby(['mode', 'n_measurements', 'duration_label', 'rho'])['mde'].mean().reset_index()
    avg['mode_label'] = avg['mode'].map(MODE_NICE)
    avg['dur_label'] = avg['duration_label'].map(DURATION_NICE)

    print("\n" + "=" * 90)
    print("MDE Summary (averaged across baseline, heterogeneity, and monitoring effects)")
    print("=" * 90)

    for dur in DURATION_ORDER:
        print(f"\n--- {DURATION_NICE[dur]} ---")
        sub = avg[avg['duration_label'] == dur]
        pivot = sub.pivot_table(
            index=['mode_label', 'n_measurements'],
            columns='rho',
            values='mde',
        )
        pivot.index.names = ['Scenario', 'Tests/wk']
        pivot.columns = [f'AR(1)={r}' for r in pivot.columns]
        display = pivot.map(lambda v: f'{v:.3f}' if pd.notna(v) else '>0.40')
        print(display.to_string())

    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize MDE comparison results.")
    parser.add_argument('--input', type=str, default='output/comparison_results.csv')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("Run 'python run_comparison_sweep.py' first.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"  Modes: {df['mode'].unique().tolist()}")
    print(f"  Measurements: {sorted(df['n_measurements'].unique())}")
    print(f"  Durations: {df['duration_label'].unique().tolist()}")
    print()

    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    print("Computing MDE table...")
    mde_df = compute_mde_comparison(df)
    mde_path = os.path.join(args.output_dir, 'comparison_mde_table.csv')
    mde_df.to_csv(mde_path, index=False)
    print(f"  MDE table saved to {mde_path}")

    print_mde_summary_table(mde_df)

    print("Generating MDE over time plots...")
    plot_mde_over_time(mde_df, args.output_dir)

    print("Generating MDE comparison bars...")
    plot_mde_comparison_bars(mde_df, args.output_dir)

    print("Generating comparison power curves...")
    plot_power_curves_comparison(df, args.output_dir)

    print("\nAll comparison visualizations complete.")


if __name__ == '__main__':
    main()
