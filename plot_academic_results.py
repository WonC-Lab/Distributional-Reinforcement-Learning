# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 2026

@author: WonChan Cho
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    # Academic style configuration
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['figure.dpi'] = 300
    
    # ----------------------------------------------------
    # Load Data
    # ----------------------------------------------------
    try:
        df_returns = pd.read_csv('dist_rl_ez_experiment_returns.csv')
        df_consumption = pd.read_csv('dist_rl_ez_consumption.csv')
        df_weights = pd.read_csv('dist_rl_ez_rep_weights.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV files. Please run the experiment script first! Details: {e}")
        return
        
    print("CSV data loaded successfully. Generating figures...")
    
    # ----------------------------------------------------
    # Figure 1: Cumulative Wealth (Portfolio Growth)
    # ----------------------------------------------------
    plt.figure(figsize=(11, 6))
    ax = plt.subplot(111)
    
    colors = {
        'Buy_and_Hold': ('dimgray', '--', 'Buy & Hold (1/N)'),
        'EZ_g2.0_p0.5': ('#ff9999', '-', r'EZ ($\gamma=2.0, \psi=0.5$)'),
        'EZ_g2.0_p1.5': ('#ff3333', '-', r'EZ ($\gamma=2.0, \psi=1.5$)'),
        'EZ_g8.0_p0.5': ('#9999ff', '-', r'EZ ($\gamma=8.0, \psi=0.5$)'),
        'EZ_g8.0_p1.5': ('#000099', '-', r'EZ ($\gamma=8.0, \psi=1.5$)')
    }
    
    for col in df_returns.columns:
        if col not in colors:
            continue
        # Cumulative Wealth = cumprod(1 + R)
        cum_wealth = np.cumprod(1.0 + df_returns[col])
        color, ls, label = colors[col]
        
        # Calculate Sharpe Ratio
        std = np.std(df_returns[col])
        sr = (np.mean(df_returns[col]) / std) * np.sqrt(252) if std > 1e-6 else 0.0
        
        ax.plot(cum_wealth, label=f'{label} [SR: {sr:.2f}]', color=color, linestyle=ls, linewidth=2.0)
        
    ax.set_title('Impact of Epstein-Zin Preferences on Out-of-Sample Portfolio Growth (2022-2025)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Trading Days (Out-of-Sample)', labelpad=10)
    ax.set_ylabel('Cumulative Wealth ($W_t / W_0$)', labelpad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, edgecolor='black')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('figure_ez_portfolio_growth.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_portfolio_growth.png'")

    # ----------------------------------------------------
    # Figure 2: Portfolio Allocation Stacked Area Chart (EZ g8.0 p1.5)
    # ----------------------------------------------------
    plt.figure(figsize=(11, 6))
    ax = plt.subplot(111)
    
    tickers = df_weights.columns
    time_index = np.arange(len(df_weights))
    
    # Smooth weights slightly to look better in publication
    df_weights_smooth = df_weights.rolling(window=5, min_periods=1).mean()
    
    # Clean color palette for assets
    asset_colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']
    
    ax.stackplot(time_index, [df_weights_smooth[t].values for t in tickers], labels=tickers, colors=asset_colors, alpha=0.85)
    
    ax.set_title('Out-of-Sample Dynamic Asset Allocation Allocation Trajectory (EZ: $\gamma=8.0, \psi=1.5$)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Trading Days (Out-of-Sample)', labelpad=10)
    ax.set_ylabel('Portfolio Weights', labelpad=10)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, len(df_weights) - 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('figure_ez_asset_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_asset_allocation.png'")

    # ----------------------------------------------------
    # Figure 3: Consumption-Wealth Ratio Plot (c_t)
    # ----------------------------------------------------
    plt.figure(figsize=(11, 6))
    ax = plt.subplot(111)
    
    for col in df_consumption.columns:
        if col not in colors:
            continue
        # Convert to percentage
        c_percent = df_consumption[col] * 100.0
        color, ls, label = colors[col]
        
        # Smoothed line to show the trend clearly
        c_smooth = c_percent.rolling(window=10, min_periods=1).mean()
        
        ax.plot(c_smooth, label=label, color=color, linestyle=ls, linewidth=2.0)
        
    ax.set_title('Out-of-Sample Dynamics of the Consumption-Wealth Ratio ($c_t$)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Trading Days (Out-of-Sample)', labelpad=10)
    ax.set_ylabel('Consumption-Wealth Ratio (%)', labelpad=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', frameon=True, edgecolor='black')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('figure_ez_consumption_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved 'figure_ez_consumption_rate.png'")
    
    print("All figures generated successfully.")

if __name__ == "__main__":
    main()
