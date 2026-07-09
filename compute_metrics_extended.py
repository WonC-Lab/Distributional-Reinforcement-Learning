# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 2026

@author: WonChan Cho

Phase 1: Extended metrics computation from existing CSV results.
Computes Certainty Equivalent Return (CER), Sortino Ratio, and Calmar Ratio
for all strategies and saves results to CSV + updated summary stats.

CER is the theoretically correct evaluation metric for Epstein-Zin agents:
  CER_gamma = (mean((1+r_t)^(1-gamma)))^(1/(1-gamma)) - 1

Under high gamma, strategies with high variance (e.g. Momentum) are
severely penalized, while low-variance EZ agents with high IES outperform.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ==============================================================================
# Extended Metric Functions
# ==============================================================================

def compute_cer(returns_arr, gamma):
    """
    Certainty Equivalent Return (CER) for CRRA/EZ utility with risk aversion gamma.
    CER_gamma = (E[(1+r)^(1-gamma)])^(1/(1-gamma)) - 1
    Uses log-sum-exp for numerical stability under high gamma.
    """
    r = np.array(returns_arr)
    gross = 1.0 + r
    gross = np.clip(gross, 1e-12, None)

    if abs(gamma - 1.0) < 1e-4:
        # Log utility: geometric mean
        return float(np.exp(np.mean(np.log(gross))) - 1.0)

    alpha = 1.0 - gamma
    log_gross = np.log(gross)
    x = alpha * log_gross  # (T,)

    # logsumexp for stability
    x_max = x.max()
    log_mean_exp = x_max + np.log(np.mean(np.exp(x - x_max)))
    log_cer_plus_1 = log_mean_exp / alpha
    cer = float(np.exp(log_cer_plus_1) - 1.0)
    return cer

def compute_sortino(returns_arr, target=0.0):
    """Annualized Sortino Ratio: annualized return / downside deviation."""
    r = np.array(returns_arr)
    ann_ret = np.mean(r) * 252
    downside = r[r < target] - target
    if len(downside) == 0:
        return float('inf')
    downside_std = np.sqrt(np.mean(downside**2)) * np.sqrt(252)
    return float(ann_ret / downside_std) if downside_std > 1e-10 else 0.0

def compute_calmar(returns_arr):
    """Calmar Ratio: annualized return / Maximum Drawdown."""
    r = np.array(returns_arr)
    ann_ret = np.mean(r) * 252
    cumulative = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    mdd = float(np.max(drawdown))
    return float(ann_ret / mdd) if mdd > 1e-10 else 0.0

def compute_all_extended(returns_arr):
    return {
        "CER_g2":   round(compute_cer(returns_arr, gamma=2.0) * 100, 4),
        "CER_g8":   round(compute_cer(returns_arr, gamma=8.0) * 100, 4),
        "Sortino":  round(compute_sortino(returns_arr), 4),
        "Calmar":   round(compute_calmar(returns_arr), 4),
    }

# ==============================================================================
# Main
# ==============================================================================
def main():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['figure.dpi'] = 300

    # Load existing returns
    try:
        df = pd.read_csv('dist_rl_ez_experiment_returns.csv')
        df_summary = pd.read_csv('dist_rl_ez_summary_stats.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Run run_experiment.py first.")
        return

    print("Computing extended metrics...\n")
    rows = []
    for col in df.columns:
        ret = df[col].dropna().values
        if len(ret) < 10:
            continue
        m = compute_all_extended(ret)
        m['Strategy'] = col

        # Get Sharpe from summary
        row_sum = df_summary[df_summary['Strategy'] == col]
        if not row_sum.empty:
            m['Type'] = row_sum['Type'].values[0]
            m['Sharpe_Mean'] = row_sum['Sharpe_Mean'].values[0]
        else:
            m['Type'] = 'Unknown'
            m['Sharpe_Mean'] = 0.0
        rows.append(m)

    df_ext = pd.DataFrame(rows)[['Strategy','Type','Sharpe_Mean','CER_g2','CER_g8','Sortino','Calmar']]
    df_ext = df_ext.sort_values('CER_g8', ascending=False).reset_index(drop=True)
    df_ext.to_csv('dist_rl_ez_extended_metrics.csv', index=False)
    print(df_ext.to_string(index=False))
    print("\nSaved to 'dist_rl_ez_extended_metrics.csv'")

    # ==================================================================
    # Figure 5: CER Comparison (gamma=2 and gamma=8 side-by-side bar)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    color_map = {
        'Benchmark': '#aaaaaa',
        'EZ-RL':     '#2255cc',
        'CRRA-RL':   '#cc8800',
        'Unknown':   '#888888',
    }

    df_plot = df_ext.dropna(subset=['CER_g2','CER_g8']).copy()
    strategies = df_plot['Strategy'].tolist()
    cer_g2 = df_plot['CER_g2'].values
    cer_g8 = df_plot['CER_g8'].values
    bar_colors = [color_map.get(t, '#888888') for t in df_plot['Type']]
    y_pos = range(len(strategies))

    # Panel A: CER at gamma=2
    axes[0].barh(y_pos, cer_g2, color=bar_colors, alpha=0.85, edgecolor='white')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(strategies, fontsize=9)
    axes[0].set_xlabel('CER (%) annualized, $\\gamma=2.0$', labelpad=8)
    axes[0].set_title('$\\gamma=2.0$ (Moderate Risk Aversion)', fontsize=11, fontweight='bold')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Panel B: CER at gamma=8
    axes[1].barh(y_pos, cer_g8, color=bar_colors, alpha=0.85, edgecolor='white')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(strategies, fontsize=9)
    axes[1].set_xlabel('CER (%) annualized, $\\gamma=8.0$', labelpad=8)
    axes[1].set_title('$\\gamma=8.0$ (High Risk Aversion)', fontsize=11, fontweight='bold')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    legend_elements = [
        Line2D([0],[0], color='#aaaaaa', lw=8, label='Benchmark'),
        Line2D([0],[0], color='#2255cc', lw=8, label='EZ-RL Agent'),
        Line2D([0],[0], color='#cc8800', lw=8, label='CRRA-RL Agent'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        'Certainty Equivalent Return (CER): The Theoretically Correct Metric for EZ Agents (2019--2025)',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('figure_ez_cer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_cer_comparison.png'")

    # ==================================================================
    # Figure 6: Sortino and Calmar Ratio Comparison
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sortino_vals = df_plot['Sortino'].values
    calmar_vals  = df_plot['Calmar'].values

    # Clip extreme values for readability
    sortino_vals = np.clip(sortino_vals, -5, 5)
    calmar_vals  = np.clip(calmar_vals,  -2, 5)

    axes[0].barh(y_pos, sortino_vals, color=bar_colors, alpha=0.85, edgecolor='white')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(strategies, fontsize=9)
    axes[0].set_xlabel('Sortino Ratio (annualized)', labelpad=8)
    axes[0].set_title('Sortino Ratio', fontsize=11, fontweight='bold')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].barh(y_pos, calmar_vals, color=bar_colors, alpha=0.85, edgecolor='white')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(strategies, fontsize=9)
    axes[1].set_xlabel('Calmar Ratio (annualized return / MDD)', labelpad=8)
    axes[1].set_title('Calmar Ratio', fontsize=11, fontweight='bold')
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        'Downside-Risk-Adjusted Performance: Sortino and Calmar Ratios (2019--2025)',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig('figure_ez_sortino_calmar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_sortino_calmar.png'")

    print("\nAll extended metrics and figures complete.")

if __name__ == "__main__":
    main()
