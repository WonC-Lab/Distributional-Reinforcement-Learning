# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 2026

@author: WonChan Cho
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from dynamic_ez_env import DynamicEZRealMarketEnv
from dynamic_ez_agent import DistEZAgent

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    
    tickers = ['SPY', 'QQQ', 'TLT', 'IEF', 'GLD', 'DBC', 'USO']
    window_size = 20
    num_episodes = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_start = "2015-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2025-12-31"
    
    # Grid search parameters for Epstein-Zin preferences
    # Separating Risk Aversion (gamma) and IES (psi)
    gamma_values = [2.0, 8.0]
    psi_values = [0.5, 1.5]
    
    # We will log out-of-sample metrics for each combination
    all_test_returns = {}
    all_test_consumption = {}
    representative_weights = None
    rep_label = None
    
    # ----------------------------------------------------
    # 1. Buy & Hold Benchmark Calculation (1/N Equal Weight)
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("Step 1: Compute Out-of-Sample Buy & Hold (1/N) Benchmark")
    print("="*80)
    
    test_env = DynamicEZRealMarketEnv(
        tickers=tickers, start_date=test_start, end_date=test_end, 
        window_size=window_size, device=device
    )
    
    bh_returns = []
    test_env.reset()
    done = False
    while not done:
        # Equal weight allocation
        bh_weight = np.ones(test_env.n_assets) / test_env.n_assets
        market_return = test_env.returns[test_env.current_step]
        bh_ret = np.sum(bh_weight * market_return)
        bh_returns.append(bh_ret)
        
        # Advance env (consumption is ignored for B&H, dummy action)
        _, _, done, _ = test_env.step(bh_weight, 0.05)
        
    all_test_returns['Buy_and_Hold'] = bh_returns
    print(f"Calculated B&H returns. Length: {len(bh_returns)}")

    # ----------------------------------------------------
    # 2. Grid Search Training and Testing
    # ----------------------------------------------------
    for gamma in gamma_values:
        for psi in psi_values:
            label = f"EZ_g{gamma}_p{psi}"
            print("\n" + "="*80)
            print(f"Step 2: Training Agent for {label}")
            print(f"Parameters: Risk Aversion (gamma) = {gamma}, IES (psi) = {psi}")
            print("="*80)
            
            # Initialize training environment
            train_env = DynamicEZRealMarketEnv(
                tickers=tickers, start_date=train_start, end_date=train_end, 
                window_size=window_size, device=device
            )
            
            # Initialize agent
            agent = DistEZAgent(
                n_assets=train_env.n_assets, 
                input_dim=train_env.feature_dim, 
                gamma=gamma, 
                psi=psi, 
                device=device
            )
            
            # Training loop
            for episode in range(1, num_episodes + 1):
                state = train_env.reset()
                episode_reward = 0.0
                done = False
                
                loss_c_list = []
                loss_a_list = []
                
                step_count = 0
                while not done:
                    action_w, action_c = agent.get_action(state)
                    next_state, reward, done, info = train_env.step(action_w, action_c)
                    
                    # Store experience
                    agent.replay_buffer.append((state, action_w, action_c, float(reward), next_state, done))
                    
                    # Update model every 5 steps
                    if step_count % 5 == 0:
                        loss_c, loss_a = agent.update()
                        if loss_c != 0.0:
                            loss_c_list.append(loss_c)
                            loss_a_list.append(loss_a)
                        
                    state = next_state
                    episode_reward += reward
                    step_count += 1
                
                avg_loss_c = np.mean(loss_c_list) if loss_c_list else 0.0
                avg_loss_a = np.mean(loss_a_list) if loss_a_list else 0.0
                print(f"Ep {episode:>2}/{num_episodes} | Return: {episode_reward:7.4f} | Critic Loss: {avg_loss_c:7.4e} | Actor Loss: {avg_loss_a:7.4f}")
            
            # Out-of-Sample Testing
            print(f"\n--- Testing Agent {label} Out-of-Sample ---")
            test_env_rl = DynamicEZRealMarketEnv(
                tickers=tickers, start_date=test_start, end_date=test_end, 
                window_size=window_size, device=device
            )
            agent.actor.eval()
            
            test_state = test_env_rl.reset()
            test_done = False
            
            rl_returns = []
            rl_consumption = []
            weights_record = []
            
            with torch.no_grad():
                while not test_done:
                    # Get deterministic action
                    action_w, action_c = agent.get_action(test_state)
                    
                    # Step environment
                    test_state, reward, test_done, info = test_env_rl.step(action_w, action_c)
                    
                    rl_returns.append(reward)
                    rl_consumption.append(action_c.cpu().numpy().item())
                    weights_record.append(action_w.cpu().numpy())
                    
            all_test_returns[label] = rl_returns
            all_test_consumption[label] = rl_consumption
            
            # Save the highest risk-aversion, high-IES weights for visualization analysis
            if gamma == 8.0 and psi == 1.5:
                representative_weights = np.array(weights_record)
                rep_label = label
                
    # ----------------------------------------------------
    # 3. Save Results to CSV
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("Step 3: Saving Experiment Results to CSV")
    print("="*80)
    
    # Synchronize lengths to avoid pandas DataFrame errors
    min_len = min([len(v) for v in all_test_returns.values()])
    for k in all_test_returns.keys():
        all_test_returns[k] = all_test_returns[k][:min_len]
    for k in all_test_consumption.keys():
        all_test_consumption[k] = all_test_consumption[k][:min_len]
        
    df_returns = pd.DataFrame(all_test_returns)
    df_returns.to_csv("dist_rl_ez_experiment_returns.csv", index=False)
    print("Saved returns to 'dist_rl_ez_experiment_returns.csv'")
    
    df_consumption = pd.DataFrame(all_test_consumption)
    df_consumption.to_csv("dist_rl_ez_consumption.csv", index=False)
    print("Saved consumption to 'dist_rl_ez_consumption.csv'")
    
    if representative_weights is not None:
        df_weights = pd.DataFrame(representative_weights, columns=tickers)
        df_weights.to_csv("dist_rl_ez_rep_weights.csv", index=False)
        print(f"Saved representative weights of {rep_label} to 'dist_rl_ez_rep_weights.csv'")
        
    print("\nAll experiments successfully completed!")

if __name__ == "__main__":
    main()
