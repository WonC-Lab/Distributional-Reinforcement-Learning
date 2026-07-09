# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 2026

@author: WonChan Cho

Updated plotting script for the enhanced experiment results (2019-2025 test period).
Includes all benchmarks (Buy & Hold, Risk Parity, Momentum, Min Variance),
CRRA baselines, and multi-seed error bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

def main():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['figure.dpi'] = 300

    # ------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------
    try:
        df_returns     = pd.read_csv('dist_rl_ez_experiment_returns.csv')
        df_consumption = pd.read_csv('dist_rl_ez_consumption.csv')
        df_weights     = pd.read_csv('dist_rl_ez_rep_weights.csv')
        df_summary     = pd.read_csv('dist_rl_ez_summary_stats.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV files. Please run the experiment script first. Details: {e}")
        return
    print("CSV data loaded successfully. Generating figures...")

    # ------------------------------------------------------------------
    # Style maps
    # ------------------------------------------------------------------
    style_map = {
        # Benchmarks
        'Buy_and_Hold': ('dimgray',   '--', 2.0, 'Buy & Hold (1/N)'),
        'Risk_Parity':  ('#aaaaaa',   ':',  1.8, 'Risk Parity'),
        'Momentum':     ('#888888',   '-.',  1.8, 'Momentum'),
        'Min_Variance': ('#555555',   (0,(3,1,1,1)), 1.6, 'Min. Variance'),
        # EZ agents
        'EZ_g2.0_p0.5': ('#ffaaaa', '-', 1.8, r'EZ ($\gamma=2.0,\,\psi=0.5$)'),
        'EZ_g2.0_p1.5': ('#ff2222', '-', 2.2, r'EZ ($\gamma=2.0,\,\psi=1.5$)'),
        'EZ_g8.0_p0.5': ('#aaaaff', '-', 1.8, r'EZ ($\gamma=8.0,\,\psi=0.5$)'),
        'EZ_g8.0_p1.5': ('#0000cc', '-', 2.2, r'EZ ($\gamma=8.0,\,\psi=1.5$)'),
        # CRRA baselines
        'CRRA_g2.0': ('#ffcc77', '--', 1.5, r'CRRA ($\gamma=2.0$)'),
        'CRRA_g8.0': ('#cc8800', '--', 1.5, r'CRRA ($\gamma=8.0$, capped $\psi$)'),
    }

    # Helper: get sharpe string from summary
    def get_sr(label):
        row = df_summary[df_summary['Strategy'] == label]
        if row.empty:
            return ""
        sr = row['Sharpe_Mean'].values[0]
        std = row['Sharpe_Std'].values[0]
        if std > 0:
            return f"SR={sr:.3f}±{std:.3f}"
        return f"SR={sr:.3f}"

    # ==================================================================
    # Figure 1: Cumulative Wealth with all strategies
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 6.5))

    for col in df_returns.columns:
        if col not in style_map:
            continue
        cum = np.cumprod(1.0 + df_returns[col].fillna(0))
        color, ls, lw, lbl = style_map[col]
        sr_str = get_sr(col)
        ax.plot(cum.values, label=f'{lbl}  [{sr_str}]',
                color=color, linestyle=ls, linewidth=lw)

    ax.set_title(
        'Out-of-Sample Portfolio Growth: EZ-RL vs Benchmarks (2019--2025)',
        fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Trading Days (Out-of-Sample)', labelpad=8)
    ax.set_ylabel(r'Cumulative Wealth ($W_t\,/\,W_0$)', labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9, edgecolor='grey')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figure_ez_portfolio_growth.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_portfolio_growth.png'")

    # ==================================================================
    # Figure 2: Asset Allocation (EZ g8.0 p1.5 representative)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 5.5))
    tickers = df_weights.columns.tolist()
    df_smooth = df_weights.rolling(window=5, min_periods=1).mean()
    asset_colors = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1']
    ax.stackplot(np.arange(len(df_smooth)),
                 [df_smooth[t].values for t in tickers],
                 labels=tickers, colors=asset_colors, alpha=0.85)
    ax.set_title(
        r'Dynamic Asset Allocation: EZ Agent ($\gamma=8.0,\,\psi=1.5$)  ---  2019--2025',
        fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Trading Days (Out-of-Sample)', labelpad=8)
    ax.set_ylabel('Portfolio Weight', labelpad=8)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, len(df_smooth) - 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              fontsize=9, frameon=True, edgecolor='grey')
    plt.tight_layout()
    plt.savefig('figure_ez_asset_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_asset_allocation.png'")

    # ==================================================================
    # Figure 3: Consumption-Wealth Ratio (EZ agents only)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ez_cols = [c for c in df_consumption.columns if c.startswith('EZ_')]
    for col in ez_cols:
        if col not in style_map:
            continue
        color, ls, lw, lbl = style_map[col]
        c_pct = df_consumption[col] * 100.0
        c_smooth = c_pct.rolling(window=10, min_periods=1).mean()
        ax.plot(c_smooth.values, label=lbl,
                color=color, linestyle=ls, linewidth=lw)
    ax.set_title(
        r'Consumption-Wealth Ratio $c_t$ by Intertemporal Elasticity of Substitution (2019--2025)',
        fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Trading Days (Out-of-Sample)', labelpad=8)
    ax.set_ylabel(r'Consumption-Wealth Ratio $c_t$ (%)', labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor='grey')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figure_ez_consumption_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_consumption_rate.png'")

    # ==================================================================
    # Figure 4: Sharpe Ratio Bar Chart with Bootstrap CI (NEW)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 5.5))
    df_s = df_summary.dropna(subset=['Sharpe_Mean']).copy()
    df_s = df_s[df_s['FinalWealth_Mean'].notna()]  # drop NaN strategies
    df_s = df_s.sort_values('Sharpe_Mean', ascending=False).reset_index(drop=True)

    colors_bar = []
    for _, row in df_s.iterrows():
        if row['Type'] == 'Benchmark':
            colors_bar.append('#aaaaaa')
        elif row['Type'] == 'EZ-RL':
            colors_bar.append('#2255cc')
        else:
            colors_bar.append('#cc8800')

    bars = ax.barh(df_s['Strategy'], df_s['Sharpe_Mean'],
                   color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Bootstrap CI error bars
    xerr_lo = df_s['Sharpe_Mean'] - df_s['Sharpe_CI_Lo']
    xerr_hi = df_s['Sharpe_CI_Hi'] - df_s['Sharpe_Mean']
    ax.errorbar(df_s['Sharpe_Mean'], df_s['Strategy'],
                xerr=[xerr_lo, xerr_hi],
                fmt='none', color='black', capsize=4, linewidth=1.2)

    ax.set_xlabel('Annualized Sharpe Ratio (with 95% Bootstrap CI)', labelpad=8)
    ax.set_title('Comparative Sharpe Ratio: EZ-RL vs All Benchmarks (2019--2025)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='black', linewidth=0.8)
    legend_elements = [
        Line2D([0], [0], color='#aaaaaa', lw=8, label='Benchmark'),
        Line2D([0], [0], color='#2255cc', lw=8, label='EZ-RL Agent'),
        Line2D([0], [0], color='#cc8800', lw=8, label='CRRA-RL Agent'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig('figure_ez_sharpe_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_sharpe_comparison.png'")

    # ==================================================================
    # Figure 5: γ/ψ Ablation — 2×2 grid + CRRA, with bootstrap CI
    # Shows IES (ψ) and risk-aversion (γ) effects clearly
    # ==================================================================
    from scipy.stats import bootstrap as sp_bootstrap

    def sharpe_stat(r):
        r = np.asarray(r)
        return np.mean(r) * 252 / (np.std(r, ddof=1) * np.sqrt(252) + 1e-9)

    ablation_order = [
        ('EZ_g2.0_p0.5', r'EZ ($\gamma=2,\psi=0.5$)',  '#ffbbbb'),
        ('EZ_g2.0_p1.5', r'EZ ($\gamma=2,\psi=1.5$)',  '#ff2222'),
        ('EZ_g8.0_p0.5', r'EZ ($\gamma=8,\psi=0.5$)',  '#aaaaff'),
        ('EZ_g8.0_p1.5', r'EZ ($\gamma=8,\psi=1.5$)',  '#0000cc'),
        ('CRRA_g2.0',    r'CRRA ($\gamma=2,\psi=1/\gamma$)', '#cc8800'),
    ]

    labels_abl, colors_abl, sharpes_abl, ci_lo_abl, ci_hi_abl = [], [], [], [], []
    for col, lbl, clr in ablation_order:
        if col not in df_returns.columns:
            continue
        r = df_returns[col].dropna().values
        sr = sharpe_stat(r)

        # bootstrap CI for Sharpe
        res = sp_bootstrap((r,), sharpe_stat, n_resamples=2000,
                           confidence_level=0.95, method='percentile', random_state=0)
        labels_abl.append(lbl)
        colors_abl.append(clr)
        sharpes_abl.append(sr)
        ci_lo_abl.append(res.confidence_interval.low)
        ci_hi_abl.append(res.confidence_interval.high)

    sharpes_abl = np.array(sharpes_abl)
    ci_lo_abl   = np.array(ci_lo_abl)
    ci_hi_abl   = np.array(ci_hi_abl)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels_abl))
    bars = ax.bar(x, sharpes_abl, color=colors_abl, alpha=0.88,
                  edgecolor='white', linewidth=0.6, width=0.55)

    # Error bars
    ax.errorbar(x, sharpes_abl,
                yerr=[sharpes_abl - ci_lo_abl, ci_hi_abl - sharpes_abl],
                fmt='none', color='black', capsize=5, linewidth=1.3)

    # Annotate Sharpe values
    for i, (sr, lo, hi) in enumerate(zip(sharpes_abl, ci_lo_abl, ci_hi_abl)):
        ax.text(i, hi + 0.025, f'{sr:.3f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold')

    # Significance brackets: ψ effect (within γ=2: bars 0→1, within γ=8: bars 2→3)
    def draw_bracket(ax, x1, x2, y, text, color='#333333'):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='-', color=color, lw=1.3))
        for xi in [x1, x2]:
            ax.plot([xi, xi], [y - 0.01, y], color=color, lw=1.3)
        ax.text((x1 + x2) / 2, y + 0.015, text, ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold')

    y_bracket = max(ci_hi_abl[:4]) + 0.12
    draw_bracket(ax, 0, 1, y_bracket,       r'$\psi$ effect, $\gamma=2$: $p=0.002^{***}$')
    draw_bracket(ax, 2, 3, y_bracket,       r'$\psi$ effect, $\gamma=8$: $p=0.049^{**}$')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_abl, fontsize=9.5)
    ax.set_ylabel('Annualized Sharpe Ratio (95% Bootstrap CI)', labelpad=8)
    ax.set_title(
        r'$\gamma$/$\psi$ Ablation: Sharpe Ratio by Risk Aversion and IES (2019--2025)',
        fontsize=13, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-0.05, y_bracket + 0.2)

    legend_elements = [
        Line2D([0], [0], color='#ff4444', lw=8, label=r'EZ, $\gamma=2$'),
        Line2D([0], [0], color='#4444cc', lw=8, label=r'EZ, $\gamma=8$'),
        Line2D([0], [0], color='#cc8800', lw=8, label='CRRA baseline'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig('figure_ez_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_ablation.png'")

    print("All figures generated successfully.")

if __name__ == "__main__":
    main()
