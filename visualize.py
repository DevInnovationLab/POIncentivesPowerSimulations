#!/usr/bin/env python3
"""Stage 4: Visualization and MDE table from power simulation results.

Supports both single-state and pooled (AP + Odisha) results.

Usage:
    python visualize.py [--input output/power_results.csv] [--output_dir output]
    python visualize.py --pooled [--input output/pooled_power_results.csv] [--output_dir output]
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

# ---------------------------------------------------------------------------
# Plot style defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

SAVE_DPI = 150

# Descriptive labels for parameters
LABELS = {
    'target_att': 'Target Effect on Chlorination Rate',
    'mu_baseline': 'Baseline Compliance Rate',
    'mu_baseline_ap': 'AP Baseline Compliance',
    'mu_baseline_od': 'Odisha Baseline Compliance',
    'sigma_baseline': 'Compliance Heterogeneity (SD)',
    'rho': 'Behavioral Persistence (AR1)',
    'h_init': 'Initial Monitoring Effect',
    'effect_ratio': 'Odisha Effect Ratio',
    'power': 'Statistical Power',
    'mde': 'Min. Detectable Effect (pp)',
}


# ---------------------------------------------------------------------------
# Single-state power curve plots
# ---------------------------------------------------------------------------

def plot_power_curves(df, output_dir):
    """Power vs target_att, one figure per (rho, h_init), subplots per sigma_baseline."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    rho_vals = sorted(df['rho'].unique())
    h_vals = sorted(df['h_init'].unique())
    sigma_vals = sorted(df['sigma_baseline'].unique())
    mu_vals = sorted(df['mu_baseline'].unique())

    colors = sns.color_palette('tab10', n_colors=len(mu_vals))

    for rho in rho_vals:
        for h in h_vals:
            n_sigma = len(sigma_vals)
            ncols = min(n_sigma, 2)
            nrows = int(np.ceil(n_sigma / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                                     squeeze=False)

            for idx, sigma in enumerate(sigma_vals):
                ax = axes[idx // ncols, idx % ncols]
                subset = df[(df['rho'] == rho) & (df['h_init'] == h) &
                            (df['sigma_baseline'] == sigma)]

                for mu_idx, mu in enumerate(mu_vals):
                    sub = subset[subset['mu_baseline'] == mu].sort_values('target_att')
                    ax.plot(sub['target_att'], sub['power'], marker='o', markersize=4,
                            color=colors[mu_idx],
                            label=f'Baseline={mu}', linewidth=1.5)

                ax.axhline(0.80, color='grey', linestyle='--', linewidth=1,
                           label='80% power' if idx == 0 else None)
                ax.set_title(f'{LABELS["sigma_baseline"]}={sigma}')
                ax.set_xlabel(LABELS['target_att'])
                ax.set_ylabel(LABELS['power'])
                ax.set_ylim(-0.02, 1.05)
                ax.set_xlim(subset['target_att'].min(), subset['target_att'].max())
                ax.grid(True, alpha=0.3)

                if idx == 0:
                    ax.legend(loc='lower right', fontsize=8, ncol=2)

            for idx in range(n_sigma, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            fig.suptitle(
                f'Power Curves\n{LABELS["rho"]}={rho}, {LABELS["h_init"]}={h}',
                fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.93])

            fname = f'power_curves_persistence{rho}_monitoring{h}.png'
            fig.savefig(os.path.join(plots_dir, fname), dpi=SAVE_DPI)
            plt.close(fig)

    print(f"  Power curve plots saved ({len(rho_vals) * len(h_vals)} figures)")


# ---------------------------------------------------------------------------
# Single-state power heatmaps
# ---------------------------------------------------------------------------

def plot_power_heatmaps(df, output_dir):
    """Heatmap of power (mu_baseline x target_att) for each (rho, h_init, sigma_baseline)."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    rho_vals = sorted(df['rho'].unique())
    h_vals = sorted(df['h_init'].unique())
    sigma_vals = sorted(df['sigma_baseline'].unique())
    count = 0

    for rho in rho_vals:
        for h in h_vals:
            for sigma in sigma_vals:
                subset = df[(df['rho'] == rho) & (df['h_init'] == h) &
                            (df['sigma_baseline'] == sigma)]

                pivot = subset.pivot_table(
                    index='mu_baseline', columns='target_att', values='power'
                )
                pivot = pivot.sort_index(ascending=False)

                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                            vmin=0, vmax=1, linewidths=0.5, ax=ax,
                            cbar_kws={'label': LABELS['power']})
                ax.set_title(
                    f'{LABELS["power"]}\n'
                    f'{LABELS["rho"]}={rho}, {LABELS["h_init"]}={h}, '
                    f'{LABELS["sigma_baseline"]}={sigma}',
                    fontsize=13, fontweight='bold')
                ax.set_xlabel(LABELS['target_att'])
                ax.set_ylabel(LABELS['mu_baseline'])
                fig.tight_layout()

                fname = f'power_heatmap_persistence{rho}_monitoring{h}_sd{sigma}.png'
                fig.savefig(os.path.join(plots_dir, fname), dpi=SAVE_DPI)
                plt.close(fig)
                count += 1

    print(f"  Power heatmap plots saved ({count} figures)")


# ---------------------------------------------------------------------------
# MDE table with linear interpolation
# ---------------------------------------------------------------------------

def compute_mde_table(df, power_col='power', group_cols=None):
    """Find minimum target_att achieving 80% power for each parameter combo.

    Uses linear interpolation between adjacent target_att values.
    """
    if group_cols is None:
        group_cols = ['mu_baseline', 'sigma_baseline', 'rho', 'h_init']

    records = []
    for keys, grp in df.groupby(group_cols):
        grp_sorted = grp.sort_values('target_att')
        target_atts = grp_sorted['target_att'].values
        powers = grp_sorted[power_col].values
        mde = np.nan

        if (powers >= 0.80).any():
            first_idx = np.where(powers >= 0.80)[0][0]
            if first_idx == 0:
                mde = target_atts[0]
            else:
                p_lo, p_hi = powers[first_idx - 1], powers[first_idx]
                t_lo, t_hi = target_atts[first_idx - 1], target_atts[first_idx]
                if p_hi > p_lo:
                    mde = t_lo + (0.80 - p_lo) / (p_hi - p_lo) * (t_hi - t_lo)
                else:
                    mde = t_hi

        if isinstance(keys, tuple):
            row = dict(zip(group_cols, keys))
        else:
            row = {group_cols[0]: keys}
        row['mde'] = mde
        records.append(row)

    return pd.DataFrame(records)


def plot_mde_summary(mde_df, output_dir):
    """Grouped bar chart showing MDE by mu_baseline, grouped by rho."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    summary = mde_df.groupby(['mu_baseline', 'rho'])['mde'].mean().reset_index()
    summary = summary.dropna(subset=['mde'])

    if summary.empty:
        print("  WARNING: No combos achieved 80% power; skipping MDE summary plot.")
        return

    mu_vals = sorted(summary['mu_baseline'].unique())
    rho_vals = sorted(summary['rho'].unique())
    n_rho = len(rho_vals)
    bar_width = 0.8 / n_rho
    colors = sns.color_palette('Set2', n_colors=n_rho)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(mu_vals))

    for i, rho in enumerate(rho_vals):
        sub = summary[summary['rho'] == rho]
        mde_vals = []
        for mu in mu_vals:
            row = sub[sub['mu_baseline'] == mu]
            mde_vals.append(row['mde'].values[0] if len(row) > 0 else np.nan)
        offset = (i - (n_rho - 1) / 2) * bar_width
        bars = ax.bar(x + offset, mde_vals, width=bar_width,
                       label=f'{LABELS["rho"]}={rho}',
                       color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, mde_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mu_vals])
    ax.set_xlabel(LABELS['mu_baseline'])
    ax.set_ylabel(LABELS['mde'])
    ax.set_title('Minimum Detectable Effect on Chlorination Rate\n'
                 '(averaged across monitoring effect and compliance heterogeneity)',
                 fontsize=13, fontweight='bold')
    ax.legend(title=LABELS['rho'])
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'mde_summary.png'), dpi=SAVE_DPI)
    plt.close(fig)
    print("  MDE summary plot saved")


# ---------------------------------------------------------------------------
# Pooled visualization: AP-only vs Pooled power comparison
# ---------------------------------------------------------------------------

def plot_pooled_power_comparison(df, output_dir):
    """Power curves comparing AP-only vs Odisha-only vs Pooled estimation.

    One figure per (rho, h_init, effect_ratio), subplots per sigma_baseline.
    Lines colored by estimator (AP/Odisha/Pooled), averaged over mu_baseline combos.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    rho_vals = sorted(df['rho'].unique())
    h_vals = sorted(df['h_init'].unique())
    er_vals = sorted(df['effect_ratio'].unique())
    sigma_vals = sorted(df['sigma_baseline'].unique())

    count = 0
    for rho in rho_vals:
        for h in h_vals:
            for er in er_vals:
                n_sigma = len(sigma_vals)
                ncols = min(n_sigma, 2)
                nrows = int(np.ceil(n_sigma / ncols))
                fig, axes = plt.subplots(nrows, ncols,
                                         figsize=(6 * ncols, 4.5 * nrows),
                                         squeeze=False)

                for idx, sigma in enumerate(sigma_vals):
                    ax = axes[idx // ncols, idx % ncols]
                    subset = df[(df['rho'] == rho) & (df['h_init'] == h) &
                                (df['effect_ratio'] == er) &
                                (df['sigma_baseline'] == sigma)]

                    # Average power across mu_baseline_ap and mu_baseline_od
                    avg = subset.groupby('target_att')[
                        ['power_ap', 'power_od', 'power_pooled']
                    ].mean().reset_index().sort_values('target_att')

                    ax.plot(avg['target_att'], avg['power_ap'],
                            marker='s', markersize=4, color='tab:blue',
                            label='AP only (n=50)', linewidth=1.5)
                    ax.plot(avg['target_att'], avg['power_od'],
                            marker='^', markersize=4, color='tab:green',
                            label='Odisha only (n=50)', linewidth=1.5)
                    ax.plot(avg['target_att'], avg['power_pooled'],
                            marker='o', markersize=5, color='tab:red',
                            label='Pooled with state FE (n=100)', linewidth=2)

                    ax.axhline(0.80, color='grey', linestyle='--', linewidth=1)
                    ax.set_title(f'{LABELS["sigma_baseline"]}={sigma}')
                    ax.set_xlabel(LABELS['target_att'])
                    ax.set_ylabel(LABELS['power'])
                    ax.set_ylim(-0.02, 1.05)
                    ax.grid(True, alpha=0.3)

                    if idx == 0:
                        ax.legend(loc='lower right', fontsize=8)

                for idx in range(n_sigma, nrows * ncols):
                    axes[idx // ncols, idx % ncols].set_visible(False)

                fig.suptitle(
                    f'Power Comparison: AP vs Odisha vs Pooled\n'
                    f'{LABELS["rho"]}={rho}, {LABELS["h_init"]}={h}, '
                    f'{LABELS["effect_ratio"]}={er}',
                    fontsize=14, fontweight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.91])

                fname = f'pooled_power_comparison_rho{rho}_h{h}_er{er}.png'
                fig.savefig(os.path.join(plots_dir, fname), dpi=SAVE_DPI)
                plt.close(fig)
                count += 1

    print(f"  Pooled power comparison plots saved ({count} figures)")


def plot_pooled_power_gain(df, output_dir):
    """Heatmap of power gain from pooling (pooled power - AP power).

    Averaged across mu_baseline combos. One heatmap per (sigma, effect_ratio).
    Rows = rho, columns = target_att.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    df['power_gain'] = df['power_pooled'] - df['power_ap']

    sigma_vals = sorted(df['sigma_baseline'].unique())
    er_vals = sorted(df['effect_ratio'].unique())
    count = 0

    for sigma in sigma_vals:
        for er in er_vals:
            subset = df[(df['sigma_baseline'] == sigma) & (df['effect_ratio'] == er)]
            # Average across baselines and h_init
            avg = subset.groupby(['rho', 'target_att'])['power_gain'].mean().reset_index()
            pivot = avg.pivot_table(index='rho', columns='target_att', values='power_gain')
            pivot = pivot.sort_index(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(pivot, annot=True, fmt='+.2f', cmap='RdBu_r',
                        center=0, vmin=-0.2, vmax=0.3,
                        linewidths=0.5, ax=ax,
                        cbar_kws={'label': 'Power Gain (Pooled - AP only)'})
            ax.set_title(
                f'Power Gain from Pooling (AP + Odisha)\n'
                f'{LABELS["sigma_baseline"]}={sigma}, {LABELS["effect_ratio"]}={er}',
                fontsize=13, fontweight='bold')
            ax.set_xlabel(LABELS['target_att'])
            ax.set_ylabel(LABELS['rho'])
            fig.tight_layout()

            fname = f'pooled_power_gain_sd{sigma}_er{er}.png'
            fig.savefig(os.path.join(plots_dir, fname), dpi=SAVE_DPI)
            plt.close(fig)
            count += 1

    print(f"  Pooled power gain heatmaps saved ({count} figures)")


def compute_pooled_mde_table(df):
    """Compute MDE for AP-only, Odisha-only, and pooled estimators."""
    group_cols = ['mu_baseline_ap', 'mu_baseline_od', 'sigma_baseline',
                  'rho', 'h_init', 'effect_ratio']

    mde_ap = compute_mde_table(df, power_col='power_ap', group_cols=group_cols)
    mde_ap = mde_ap.rename(columns={'mde': 'mde_ap'})

    mde_od = compute_mde_table(df, power_col='power_od', group_cols=group_cols)
    mde_od = mde_od.rename(columns={'mde': 'mde_od'})

    mde_pool = compute_mde_table(df, power_col='power_pooled', group_cols=group_cols)
    mde_pool = mde_pool.rename(columns={'mde': 'mde_pooled'})

    merged = mde_ap.merge(mde_od, on=group_cols).merge(mde_pool, on=group_cols)
    return merged


def plot_pooled_mde_summary(mde_df, output_dir):
    """Bar chart comparing MDE across AP-only, Odisha-only, and pooled."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Average across baselines and h_init, group by rho and effect_ratio
    summary = mde_df.groupby(['rho', 'effect_ratio'])[
        ['mde_ap', 'mde_od', 'mde_pooled']
    ].mean().reset_index()
    summary = summary.dropna(subset=['mde_ap', 'mde_pooled'], how='all')

    if summary.empty:
        print("  WARNING: No combos achieved 80% power; skipping pooled MDE summary.")
        return

    er_vals = sorted(summary['effect_ratio'].unique())
    rho_vals = sorted(summary['rho'].unique())

    fig, axes = plt.subplots(1, len(er_vals), figsize=(6 * len(er_vals), 5),
                             squeeze=False, sharey=True)

    for col_idx, er in enumerate(er_vals):
        ax = axes[0, col_idx]
        sub = summary[summary['effect_ratio'] == er]

        x = np.arange(len(rho_vals))
        w = 0.25

        for i, (label, col, color) in enumerate([
            ('AP only', 'mde_ap', 'tab:blue'),
            ('Odisha only', 'mde_od', 'tab:green'),
            ('Pooled', 'mde_pooled', 'tab:red'),
        ]):
            vals = [sub[sub['rho'] == r][col].values[0]
                    if len(sub[sub['rho'] == r]) > 0 else np.nan
                    for r in rho_vals]
            bars = ax.bar(x + (i - 1) * w, vals, width=w, label=label,
                          color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in rho_vals])
        ax.set_xlabel(LABELS['rho'])
        ax.set_title(f'{LABELS["effect_ratio"]}={er}')
        ax.grid(axis='y', alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel(LABELS['mde'])
            ax.legend(fontsize=8)

    fig.suptitle('MDE Comparison: AP vs Odisha vs Pooled\n'
                 '(averaged across baselines and monitoring effects)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(plots_dir, 'pooled_mde_summary.png'), dpi=SAVE_DPI)
    plt.close(fig)
    print("  Pooled MDE summary plot saved")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize power simulation results.")
    parser.add_argument('--input', type=str, default=None,
                        help="Path to results CSV (default: auto-detect)")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Output directory (default: output)")
    parser.add_argument('--pooled', action='store_true',
                        help="Visualize pooled (AP + Odisha) results")
    args = parser.parse_args()

    # Auto-detect input file
    if args.input is None:
        if args.pooled:
            args.input = os.path.join(args.output_dir, 'pooled_power_results.csv')
        else:
            args.input = os.path.join(args.output_dir, 'power_results.csv')

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("Run 'python run_power_sweep.py' first to generate power results.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print()

    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    if args.pooled:
        print("Generating pooled power comparison plots...")
        plot_pooled_power_comparison(df, args.output_dir)

        print("Generating pooled power gain heatmaps...")
        plot_pooled_power_gain(df, args.output_dir)

        print("Computing pooled MDE table...")
        mde_df = compute_pooled_mde_table(df)
        mde_path = os.path.join(args.output_dir, 'pooled_mde_table.csv')
        mde_df.to_csv(mde_path, index=False)
        print(f"  Pooled MDE table saved to {mde_path}")

        # Print summary
        avg = mde_df[['mde_ap', 'mde_od', 'mde_pooled']].mean()
        print(f"\n  Average MDE — AP: {avg['mde_ap']:.3f}, "
              f"Odisha: {avg['mde_od']:.3f}, Pooled: {avg['mde_pooled']:.3f}")

        print("\nGenerating pooled MDE summary plot...")
        plot_pooled_mde_summary(mde_df, args.output_dir)

    else:
        print("Generating power curve plots...")
        plot_power_curves(df, args.output_dir)

        print("Generating power heatmaps...")
        plot_power_heatmaps(df, args.output_dir)

        print("Computing MDE table...")
        mde_df = compute_mde_table(df)
        mde_path = os.path.join(args.output_dir, 'mde_table.csv')
        mde_df.to_csv(mde_path, index=False)
        print(f"  MDE table saved to {mde_path}")

        display_df = mde_df.copy()
        display_df['mde'] = display_df['mde'].apply(
            lambda v: f'{v:.3f}' if pd.notna(v) else '>0.40'
        )
        display_df = display_df.rename(columns={
            'mu_baseline': 'Baseline Compliance',
            'sigma_baseline': 'Compliance SD',
            'rho': 'Persistence',
            'h_init': 'Monitoring Effect',
            'mde': 'Min. Detectable Effect',
        })
        print()
        print("MDE Table (min target effect on chlorination rate for 80% power):")
        print("-" * 80)
        print(display_df.to_string(index=False))
        print()

        print("Generating MDE summary plot...")
        plot_mde_summary(mde_df, args.output_dir)

    print()
    print("All visualizations complete.")


if __name__ == '__main__':
    main()
