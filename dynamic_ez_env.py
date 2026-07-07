# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 2026

@author: WonChan Cho
"""

import os
import hashlib
import numpy as np
import pandas as pd
import yfinance as yf
import torch

class DynamicEZRealMarketEnv:
    def __init__(self, tickers, start_date="2015-01-01", end_date="2025-12-31", window_size=20, cost=0.001, device=None):
        self.tickers = sorted(tickers)
        self.macro_tickers = ['^VIX', '^TNX', '^IRX']  # VIX, 10Y Yield, 2Y Yield
        self.window_size = window_size
        self.cost = cost
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Unique cache file name based on tickers and dates
        ticker_str = "_".join(self.tickers) + "_" + "_".join(self.macro_tickers)
        ticker_hash = hashlib.md5(ticker_str.encode()).hexdigest()[:8]
        self.cache_filename = f"market_ez_cache_{start_date}_{end_date}_{ticker_hash}.csv"
        
        # Load or download data
        self._load_data(start_date, end_date)
        
        self.n_assets = len(self.tickers)
        self.feature_dim = self.window_size + 2 + 3  # Window returns (20) + Vol (1) + Mom (1) + VIX (1) + TNX (1) + Spread (1)
        self.max_steps = len(self.returns) - 1

    def _load_data(self, start_date, end_date):
        all_tickers = self.tickers + self.macro_tickers
        
        df = None
        if os.path.exists(self.cache_filename):
            print(f"[Env] Loading data from local cache: {self.cache_filename}")
            df = pd.read_csv(self.cache_filename, index_col=0, parse_dates=True)
            
            # Check if cache is valid (contains all columns, no all-NaN columns, and sufficient rows)
            is_valid = True
            if df.empty or len(df) < self.window_size + 10:
                is_valid = False
            else:
                for col in all_tickers:
                    if col not in df.columns or df[col].isnull().all():
                        is_valid = False
                        break
            
            if not is_valid:
                print(f"[Env] Local cache {self.cache_filename} was invalid. Deleting and re-downloading...")
                try:
                    os.remove(self.cache_filename)
                except Exception:
                    pass
                df = None
                
        if df is None:
            print(f"[Env] Downloading asset and macro data from Yahoo Finance (threads=False)...")
            data = yf.download(all_tickers, start=start_date, end=end_date, threads=False)
            
            # extract Close prices
            if 'Close' in data.columns:
                df = data['Close']
            else:
                df = data
                
            if df.empty:
                raise ValueError("yfinance download failed. Verify connection or rate limits.")
            
            # Verify download completeness before caching
            is_valid = True
            for col in all_tickers:
                if col not in df.columns or df[col].isnull().all():
                    is_valid = False
                    break
            
            if not is_valid:
                raise ValueError(f"Download complete but some columns are missing or entirely NaN. Tickers: {all_tickers}")
                
            df.to_csv(self.cache_filename)
            print(f"[Env] Saved download cache to {self.cache_filename}")
            
        df = df.ffill().bfill().dropna()
        
        if len(df) < self.window_size + 2:
            raise ValueError(f"Dataframe after dropping NaNs is too short (length: {len(df)}). Must be at least {self.window_size + 2}.")
            
        # Separate asset prices and macro indicators
        self.asset_prices = df[self.tickers]
        self.macro_prices = df[self.macro_tickers]
        
        # Compute asset returns
        self.returns = self.asset_prices.pct_change().dropna().values
        
        # Rolling stats for assets
        df_returns = pd.DataFrame(self.returns, columns=self.tickers)
        self.volatility = df_returns.rolling(window=self.window_size).std().fillna(0).values
        self.momentum = df_returns.rolling(window=self.window_size).sum().fillna(0).values
        
        # Sync macro indicators (shift by 1 day to prevent lookahead bias)
        # We use the index of returns (which starts from index 1 of the original prices)
        macro_idx = df.index[1:]
        self.vix = self.macro_prices.loc[macro_idx, '^VIX'].values / 100.0  # Normalize VIX
        self.tnx = self.macro_prices.loc[macro_idx, '^TNX'].values / 100.0  # Normalize 10Y Yield
        irx = self.macro_prices.loc[macro_idx, '^IRX'].values / 100.0       # Normalize 2Y Yield
        self.term_spread = self.tnx - irx                                   # Term spread (10Y - 2Y)
        
    def reset(self):
        self.current_step = self.window_size
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_state()

    def step(self, action_w, action_c):
        """
        action_w: Portfolio weights (N_assets,)
        action_c: Consumption fraction (1,)
        """
        if torch.is_tensor(action_w):
            action_w = action_w.cpu().numpy()
        if torch.is_tensor(action_c):
            action_c = action_c.cpu().numpy().item()
            
        # Ensure weights are normalized (safety check)
        weights = np.clip(action_w, 1e-8, 1.0)
        weights = weights / np.sum(weights)
        
        # Asset daily return at current step
        asset_returns = self.returns[self.current_step]
        
        # Compute turnover with previous period post-return weights
        # w_prev_post = w_prev * (1 + R) / sum(w_prev * (1 + R))
        prev_returns = self.returns[self.current_step - 1]
        w_prev_post = self.current_weights * (1 + prev_returns)
        w_prev_post_sum = np.sum(w_prev_post)
        if w_prev_post_sum > 1e-8:
            w_prev_post = w_prev_post / w_prev_post_sum
        else:
            w_prev_post = self.current_weights
            
        turnover = np.sum(np.abs(weights - w_prev_post))
        transaction_cost = turnover * self.cost
        
        # Portfolio gross return
        portfolio_return = np.sum(weights * asset_returns)
        
        # Portfolio net return (including transaction cost)
        net_return = portfolio_return - transaction_cost
        
        # Save weights for next step
        self.current_weights = weights
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        
        next_state = self._get_state() if not done else torch.zeros((self.n_assets, self.feature_dim)).to(self.device)
        
        info = {
            'raw_return': portfolio_return,
            'net_return': net_return,
            'transaction_cost': transaction_cost,
            'turnover': turnover,
            'consumption_fraction': action_c
        }
        
        return next_state, net_return, done, info

    def _get_state(self):
        # Window of asset returns: (window_size, N_assets) -> Transpose to (N_assets, window_size)
        window_returns = self.returns[self.current_step - self.window_size : self.current_step]
        state = window_returns.T
        
        # Vol and Mom: (N_assets,)
        vol = self.volatility[self.current_step - 1]
        mom = self.momentum[self.current_step - 1]
        
        # Concatenate vol and mom: (N_assets, window_size + 2)
        state = np.column_stack((state, vol, mom))
        
        # Macro variables for this step: VIX, TNX, term_spread (scalar values)
        vix_val = self.vix[self.current_step - 1]
        tnx_val = self.tnx[self.current_step - 1]
        spread_val = self.term_spread[self.current_step - 1]
        
        # Replicate macro variables across all assets: (N_assets, 3)
        macro_block = np.tile([vix_val, tnx_val, spread_val], (self.n_assets, 1))
        
        # Concatenate asset state with macro state: (N_assets, window_size + 2 + 3)
        full_state = np.column_stack((state, macro_block))
        
        return torch.FloatTensor(full_state).to(self.device)
