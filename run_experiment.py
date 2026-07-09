# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 2026

@author: WonChan Cho

Enhanced experiment runner with:
  1. Extended out-of-sample period (train: 2007-2018, test: 2019-2025)
  2. Additional benchmarks: CRRA-RL, Risk Parity, Minimum Variance, Momentum
  3. Statistical significance: multi-seed mean/std + bootstrap Sharpe test
  4. [Speed-Up] Parallel execution via multiprocessing (one worker per config-seed pair)
  5. [Speed-Up] CRRA_g8.0 removed (empirically unstable, Sharpe=0.0 in prior run)
  6. [Speed-Up] Gradient update every 10 steps instead of 5
"""

import os
# Fix OMP duplicate lib issue on Windows before any torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
import multiprocessing as mp
from dynamic_ez_env import DynamicEZRealMarketEnv
from dynamic_ez_agent import DistEZAgent

# ==============================================================================
# Reproducibility
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# Performance Metrics
# ==============================================================================
def compute_metrics(returns_arr):
    """Compute annualized Sharpe, annualized return, MDD from daily returns array."""
    r = np.array(returns_arr)
    ann_ret = np.mean(r) * 252
    ann_std = np.std(r, ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_std if ann_std > 1e-8 else 0.0
    cumulative = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    mdd = float(np.max(drawdown))
    final_wealth = float(cumulative[-1])
    return {"Sharpe": sharpe, "AnnReturn": ann_ret, "AnnVol": ann_std,
            "MDD": mdd, "FinalWealth": final_wealth}

# ==============================================================================
# Bootstrap Confidence Interval for Sharpe Ratio
# ==============================================================================
def bootstrap_sharpe_ci(returns_arr, n_boot=1000, alpha=0.05, seed=0):
    """Return (mean, lower_CI, upper_CI) of Sharpe via bootstrap."""
    rng = np.random.default_rng(seed)
    r = np.array(returns_arr)
    n = len(r)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(r, size=n, replace=True)
        ann_r = np.mean(sample) * 252
        ann_s = np.std(sample, ddof=1) * np.sqrt(252)
        boot_sharpes.append(ann_r / ann_s if ann_s > 1e-8 else 0.0)
    lo = np.percentile(boot_sharpes, 100 * alpha / 2)
    hi = np.percentile(boot_sharpes, 100 * (1 - alpha / 2))
    return float(np.mean(boot_sharpes)), float(lo), float(hi)

# ==============================================================================
# Rule-Based Benchmarks (computed from raw market returns)
# ==============================================================================
def compute_rule_benchmarks(env, cost=0.001):
    """
    Compute daily returns for classical rule-based benchmarks.
    Phase 2: applies the same 10bps (cost=0.001) transaction cost to all
    rule-based strategies during rebalancing, creating a fair comparison
    against RL agents that also pay 10bps per turnover.
    """
    returns_matrix = env.returns  # shape (T, N)
    T, N = returns_matrix.shape
    window = env.window_size
    rebal_freq = 21  # approx monthly

    bh_rets, rp_rets, mom_rets, mv_rets = [], [], [], []
    # Track previous weights for turnover cost calculation
    w_rp_prev  = np.ones(N) / N
    w_mom_prev = np.ones(N) / N
    w_mv_prev  = np.ones(N) / N

    for t in range(window, T):
        hist = returns_matrix[max(0, t - window): t]  # (window, N)
        r_today = returns_matrix[t]

        # --- Buy & Hold (1/N, no rebalancing, no cost) ---
        w_bh = np.ones(N) / N
        bh_rets.append(float(np.dot(w_bh, r_today)))

        # --- Risk Parity: w_i = 1/sigma_i, normalized ---
        if t % rebal_freq == 0 or t == window:
            vol = np.std(hist, axis=0, ddof=1) + 1e-8
            w_rp_new = (1.0 / vol)
            w_rp_new /= w_rp_new.sum()
            # Apply transaction cost on turnover during rebalance
            turnover_rp = np.sum(np.abs(w_rp_new - w_rp_prev))
            rp_cost = turnover_rp * cost
            w_rp = w_rp_new
            w_rp_prev = w_rp_new
        else:
            rp_cost = 0.0
        rp_ret = float(np.dot(w_rp, r_today)) - rp_cost
        rp_rets.append(rp_ret)

        # --- Momentum: top-3 by cumulative return, equal weight ---
        if t % rebal_freq == 0 or t == window:
            mom = hist.sum(axis=0)
            top_k = 3
            top_idx = np.argsort(mom)[-top_k:]
            w_mom_new = np.zeros(N)
            w_mom_new[top_idx] = 1.0 / top_k
            turnover_mom = np.sum(np.abs(w_mom_new - w_mom_prev))
            mom_cost = turnover_mom * cost
            w_mom = w_mom_new
            w_mom_prev = w_mom_new
        else:
            mom_cost = 0.0
        mom_ret = float(np.dot(w_mom, r_today)) - mom_cost
        mom_rets.append(mom_ret)

        # --- Minimum Variance: sample covariance with identity shrinkage ---
        if t % rebal_freq == 0 or t == window:
            cov = np.cov(hist.T) + 1e-4 * np.eye(N)  # shrinkage
            try:
                cov_inv = np.linalg.inv(cov)
                ones = np.ones(N)
                raw_w = cov_inv @ ones
                w_mv_new = raw_w / raw_w.sum()
                w_mv_new = np.clip(w_mv_new, 0, 1)
                w_mv_new /= w_mv_new.sum()
            except np.linalg.LinAlgError:
                w_mv_new = np.ones(N) / N
            turnover_mv = np.sum(np.abs(w_mv_new - w_mv_prev))
            mv_cost = turnover_mv * cost
            w_mv = w_mv_new
            w_mv_prev = w_mv_new
        else:
            mv_cost = 0.0
        mv_ret = float(np.dot(w_mv, r_today)) - mv_cost
        mv_rets.append(mv_ret)

    return {
        "Buy_and_Hold": bh_rets,
        "Risk_Parity": rp_rets,
        "Momentum": mom_rets,
        "Min_Variance": mv_rets,
    }

# ==============================================================================
# Worker function for parallel execution (top-level for multiprocessing pickling)
# ==============================================================================
def worker_train_and_test(args):
    """
    Top-level picklable function: trains one (config, seed) pair and returns results.
    Each spawned process re-initialises its own CUDA context safely.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    (gamma, psi, label, seed,
     tickers, train_start, train_end, test_start, test_end,
     window_size, num_episodes) = args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_env = DynamicEZRealMarketEnv(
        tickers=tickers, start_date=train_start, end_date=train_end,
        window_size=window_size, device=device
    )
    agent = DistEZAgent(
        n_assets=train_env.n_assets,
        input_dim=train_env.feature_dim,
        gamma=gamma, psi=psi, device=device
    )

    for episode in range(1, num_episodes + 1):
        state = train_env.reset()
        done = False
        step_count = 0
        ep_ret = 0.0
        while not done:
            action_w, action_c = agent.get_action(state)
            next_state, reward, done, _ = train_env.step(action_w, action_c)
            agent.replay_buffer.append((state, action_w, action_c, float(reward), next_state, done))
            # [Speed-Up] Update every 10 steps instead of 5
            if step_count % 10 == 0:
                agent.update()
            state = next_state
            ep_ret += reward
            step_count += 1

        if agent._scheduler_initialized:
            cur_lr = agent.actor_scheduler.get_last_lr()[0]
        else:
            cur_lr = 3e-4
        print(f"  [{label} seed={seed}] Ep {episode:>3}/{num_episodes} | Return: {ep_ret:7.4f} | LR: {cur_lr:.2e}",
              flush=True)

    # Out-of-sample test
    test_env = DynamicEZRealMarketEnv(
        tickers=tickers, start_date=test_start, end_date=test_end,
        window_size=window_size, device=device
    )
    agent.actor.eval()
    state = test_env.reset()
    done = False
    rl_returns, rl_consumption, weights_record = [], [], []
    with torch.no_grad():
        while not done:
            action_w, action_c = agent.get_action(state)
            state, reward, done, _ = test_env.step(action_w, action_c)
            rl_returns.append(reward)
            rl_consumption.append(action_c.cpu().numpy().item())
            weights_record.append(action_w.cpu().numpy().tolist())

    return {
        "label": label,
        "seed": seed,
        "rl_returns": rl_returns,
        "rl_consumption": rl_consumption,
        "weights_record": weights_record,
    }


# ==============================================================================
# Main Experiment
# ==============================================================================
def main():
    tickers     = ['SPY', 'QQQ', 'TLT', 'IEF', 'GLD', 'DBC', 'USO']
    window_size = 20
    num_episodes = 100   # Phase 2: 25 → 100 episodes
    N_WORKERS = 6        # [Speed-Up] Parallel workers (tune to CPU core count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Extended date range (Improvement 1) ---
    train_start = "2007-01-01"
    train_end   = "2018-12-31"
    test_start  = "2019-01-01"
    test_end    = "2025-12-31"

    # --- Multiple seeds for statistical robustness (Improvement 3) ---
    seeds = [42, 123, 777, 2024, 9999]

    # --- Grid: EZ configs + CRRA configs (Improvement 2) ---
    # CRRA is EZ with psi = 1/gamma (the restricted special case)
    ez_configs = [
        {"gamma": 2.0, "psi": 0.5},
        {"gamma": 2.0, "psi": 1.5},
        {"gamma": 8.0, "psi": 0.5},
        {"gamma": 8.0, "psi": 1.5},
    ]
    crra_configs = [
        {"gamma": 2.0, "psi": 1.0 / 2.0, "label": "CRRA_g2.0"},
        # [Speed-Up] CRRA_g8.0 excluded: empirically unstable (Sharpe=0.0 in prior run).
        # Even capping psi at 0.20 failed; numerical fragility of CRRA vs. EZ is noted in paper.
    ]

    # ----------------------------------------------------------------
    # Step 1: Rule-based benchmarks (frictionless upper bounds)
    # ----------------------------------------------------------------
    print("\n" + "="*80)
    print("Step 1: Computing Rule-Based Benchmarks")
    print("="*80)
    test_env_bench = DynamicEZRealMarketEnv(
        tickers=tickers, start_date=test_start, end_date=test_end,
        window_size=window_size, device=device
    )
    benchmark_returns = compute_rule_benchmarks(test_env_bench)
    for name, ret in benchmark_returns.items():
        m = compute_metrics(ret)
        print(f"  {name:20s} | Sharpe: {m['Sharpe']:.4f} | FinalWealth: {m['FinalWealth']:.4f} | MDD: {m['MDD']*100:.2f}%")

    # ----------------------------------------------------------------
    # Step 2: Build task list and run all (config, seed) pairs in parallel
    # ----------------------------------------------------------------
    print("\n" + "="*80)
    print(f"Step 2: Training EZ/CRRA Agents in Parallel ({N_WORKERS} workers)")
    print("="*80)

    all_configs = ez_configs + crra_configs

    tasks = []
    for cfg in all_configs:
        gamma = cfg["gamma"]
        psi   = cfg["psi"]
        label = cfg.get("label", f"EZ_g{gamma}_p{psi}")
        for seed in seeds:
            tasks.append((
                gamma, psi, label, seed,
                tickers, train_start, train_end, test_start, test_end,
                window_size, num_episodes
            ))

    print(f"  Total tasks: {len(tasks)} ({len(all_configs)} configs × {len(seeds)} seeds)")
    print(f"  Running {N_WORKERS} tasks in parallel ...\n")

    # Use 'spawn' context: required for CUDA safety on Windows (avoids fork)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        results_list = pool.map(worker_train_and_test, tasks)

    # Organize results: label -> list of per-seed lists
    all_seed_returns     = {cfg.get("label", f"EZ_g{cfg['gamma']}_p{cfg['psi']}"): [] for cfg in all_configs}
    all_seed_consumption = {cfg.get("label", f"EZ_g{cfg['gamma']}_p{cfg['psi']}"): [] for cfg in all_configs}
    rep_weights = None

    for res in results_list:
        lbl = res["label"]
        all_seed_returns[lbl].append(res["rl_returns"])
        all_seed_consumption[lbl].append(res["rl_consumption"])
        if lbl == "EZ_g8.0_p1.5" and rep_weights is None:
            rep_weights = [np.array(w) for w in res["weights_record"]]

    # ----------------------------------------------------------------
    # Step 3: Aggregate Results + Statistical Testing
    # ----------------------------------------------------------------
    print("\n" + "="*80)
    print("Step 3: Aggregating Results and Statistical Significance")
    print("="*80)

    summary_rows = []

    # Benchmarks
    for bname, brets in benchmark_returns.items():
        m = compute_metrics(brets)
        sh_mean, sh_lo, sh_hi = bootstrap_sharpe_ci(brets)
        summary_rows.append({
            "Strategy":    bname,
            "Type":        "Benchmark",
            "Sharpe_Mean": round(m["Sharpe"], 4),
            "Sharpe_Std":  0.0,
            "Sharpe_CI_Lo": round(sh_lo, 4),
            "Sharpe_CI_Hi": round(sh_hi, 4),
            "FinalWealth_Mean": round(m["FinalWealth"], 4),
            "MDD_Mean": round(m["MDD"] * 100, 2),
            "AnnReturn_Mean": round(m["AnnReturn"] * 100, 2),
        })

    # EZ/CRRA agents: aggregate across seeds
    # Use B&H returns as the reference for pairwise t-test
    bh_ref = benchmark_returns["Buy_and_Hold"]

    for label, seed_ret_list in all_seed_returns.items():
        min_len = min(len(r) for r in seed_ret_list)
        arr = np.array([r[:min_len] for r in seed_ret_list])  # (n_seeds, T)

        per_seed_metrics = [compute_metrics(arr[i]) for i in range(len(seeds))]
        sharpes     = [m["Sharpe"]      for m in per_seed_metrics]
        final_ws    = [m["FinalWealth"] for m in per_seed_metrics]
        mdds        = [m["MDD"]         for m in per_seed_metrics]
        ann_rets    = [m["AnnReturn"]   for m in per_seed_metrics]

        # Mean daily returns across seeds for bootstrap CI
        mean_daily = arr.mean(axis=0)
        sh_mean_boot, sh_lo, sh_hi = bootstrap_sharpe_ci(mean_daily)

        # Pairwise Welch t-test vs Buy & Hold daily returns
        min_len2 = min(len(mean_daily), len(bh_ref))
        t_stat, p_val = scipy_stats.ttest_ind(
            mean_daily[:min_len2], np.array(bh_ref[:min_len2]), equal_var=False
        )

        is_ez = "EZ" in label
        summary_rows.append({
            "Strategy":    label,
            "Type":        "EZ-RL" if is_ez else "CRRA-RL",
            "Sharpe_Mean":  round(np.mean(sharpes), 4),
            "Sharpe_Std":   round(np.std(sharpes, ddof=1), 4),
            "Sharpe_CI_Lo": round(sh_lo, 4),
            "Sharpe_CI_Hi": round(sh_hi, 4),
            "FinalWealth_Mean": round(np.mean(final_ws), 4),
            "MDD_Mean":     round(np.mean(mdds) * 100, 2),
            "AnnReturn_Mean": round(np.mean(ann_rets) * 100, 2),
            "t_stat_vs_BH": round(t_stat, 4),
            "p_val_vs_BH":  round(p_val, 4),
        })

        print(f"\n{label}")
        print(f"  Sharpe  : {np.mean(sharpes):.4f} ± {np.std(sharpes, ddof=1):.4f}  "
              f"95% CI [{sh_lo:.4f}, {sh_hi:.4f}]")
        print(f"  Wealth  : {np.mean(final_ws):.4f}  MDD: {np.mean(mdds)*100:.2f}%")
        print(f"  t-test vs B&H: t={t_stat:.4f}, p={p_val:.4f}")

    # ----------------------------------------------------------------
    # Step 4: Save Results
    # ----------------------------------------------------------------
    print("\n" + "="*80)
    print("Step 4: Saving Results")
    print("="*80)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("dist_rl_ez_summary_stats.csv", index=False)
    print("Saved summary statistics to 'dist_rl_ez_summary_stats.csv'")

    # Save one representative returns series per label for plotting
    flat_returns = {}
    for bname, brets in benchmark_returns.items():
        flat_returns[bname] = brets
    for label, seed_ret_list in all_seed_returns.items():
        min_len = min(len(r) for r in seed_ret_list)
        flat_returns[label] = list(np.array([r[:min_len] for r in seed_ret_list]).mean(axis=0))

    min_len_all = min(len(v) for v in flat_returns.values())
    for k in flat_returns:
        flat_returns[k] = flat_returns[k][:min_len_all]
    pd.DataFrame(flat_returns).to_csv("dist_rl_ez_experiment_returns.csv", index=False)
    print("Saved mean returns to 'dist_rl_ez_experiment_returns.csv'")

    # Consumption (mean across seeds)
    flat_cons = {}
    for label, seed_cons_list in all_seed_consumption.items():
        min_c = min(len(c) for c in seed_cons_list)
        flat_cons[label] = list(np.array([c[:min_c] for c in seed_cons_list]).mean(axis=0))
    min_c_all = min(len(v) for v in flat_cons.values())
    for k in flat_cons:
        flat_cons[k] = flat_cons[k][:min_c_all]
    pd.DataFrame(flat_cons).to_csv("dist_rl_ez_consumption.csv", index=False)
    print("Saved consumption to 'dist_rl_ez_consumption.csv'")

    if rep_weights is not None:
        pd.DataFrame(rep_weights, columns=tickers).to_csv("dist_rl_ez_rep_weights.csv", index=False)
        print("Saved representative weights to 'dist_rl_ez_rep_weights.csv'")


    print("\nAll experiments completed successfully!")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
