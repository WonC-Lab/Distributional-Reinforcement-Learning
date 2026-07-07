# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 2026

@author: WonChan Cho
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class AssetTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (Batch, N_assets, input_dim)
        emb = self.embedding(x)
        attn_out, _ = self.attn(emb, emb, emb)
        return self.ln(emb + attn_out)

class Actor(nn.Module):
    def __init__(self, input_dim, n_assets, embed_dim=64):
        super().__init__()
        self.encoder = AssetTransformer(input_dim=input_dim, embed_dim=embed_dim)
        
        # Portfolio weight score head
        self.fc_weights = nn.Linear(embed_dim, 1)
        
        # Consumption head (uses mean features across all assets)
        self.fc_consumption = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
    def forward(self, state):
        # state shape: (Batch, N_assets, input_dim)
        features = self.encoder(state) # (Batch, N_assets, embed_dim)
        
        # Weights head: (Batch, N_assets)
        scores = self.fc_weights(features).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        
        # Consumption head: (Batch, 1)
        global_features = features.mean(dim=1) # (Batch, embed_dim)
        cons_logit = self.fc_consumption(global_features)
        
        # Map to realistic consumption fraction range: [0.01, 0.20] of wealth
        consumption = 0.01 + 0.19 * torch.sigmoid(cons_logit)
        
        return weights, consumption

class QuantileEmbedding(nn.Module):
    def __init__(self, num_cosines=64, embed_dim=192):
        super().__init__()
        self.num_cosines = num_cosines
        self.fc = nn.Sequential(
            nn.Linear(num_cosines, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, taus):
        # taus shape: (Batch, K, 1)
        device = taus.device
        i_pi = torch.arange(1, self.num_cosines + 1, device=device).float() * np.pi
        i_pi = i_pi.view(1, 1, self.num_cosines)
        
        cosines = torch.cos(taus * i_pi)
        return self.fc(cosines)

class IQNCritic(nn.Module):
    def __init__(self, n_assets, input_dim, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = AssetTransformer(input_dim=input_dim, embed_dim=embed_dim)
        
        # Action encoders
        self.w_encoder = nn.Sequential(
            nn.Linear(n_assets, embed_dim),
            nn.ReLU()
        )
        self.c_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        
        # Quantile embedding (3 * embed_dim to match concatenated state-action features)
        self.phi = QuantileEmbedding(num_cosines=64, embed_dim=embed_dim * 3)
        
        # Value head
        self.fc_val = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action_w, action_c, taus):
        # 1. State feature: (Batch, embed_dim)
        state_feat = self.encoder(state).mean(dim=1)
        
        # 2. Action features: (Batch, embed_dim)
        w_feat = self.w_encoder(action_w)
        c_feat = self.c_encoder(action_c)
        
        # 3. Combine state and actions: (Batch, 3 * embed_dim)
        sa_feat = torch.cat([state_feat, w_feat, c_feat], dim=-1)
        
        # 4. Quantile embedding: (Batch, K, 3 * embed_dim)
        tau_emb = self.phi(taus)
        
        # 5. Combine and project: (Batch, K, 1)
        sa_expanded = sa_feat.unsqueeze(1).expand(-1, taus.shape[1], -1)
        combined = sa_expanded * tau_emb
        
        return self.fc_val(combined)

class DistEZAgent:
    def __init__(self, n_assets, input_dim, gamma=5.0, psi=1.5, beta=0.99, lr=3e-4, hidden_dim=128, device=None):
        self.gamma = gamma
        self.psi = psi
        self.beta = beta
        self.n_quantiles = 32
        self.batch_size = 64
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor
        self.actor = Actor(input_dim=input_dim, n_assets=n_assets, embed_dim=hidden_dim // 2).to(self.device)
        
        # Critic & Target Critic
        self.critic = IQNCritic(n_assets=n_assets, input_dim=input_dim, embed_dim=hidden_dim // 2, hidden_dim=hidden_dim).to(self.device)
        self.target_critic = IQNCritic(n_assets=n_assets, input_dim=input_dim, embed_dim=hidden_dim // 2, hidden_dim=hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay Buffer
        self.replay_buffer = deque(maxlen=100000)

    def get_action(self, state):
        with torch.no_grad():
            # state shape: (N_assets, Feature_dim) -> (1, N_assets, Feature_dim)
            state = state.unsqueeze(0)
            weights, consumption = self.actor(state)
            return weights.squeeze(0), consumption.squeeze(0)

    def calc_certainty_equivalent(self, quantile_values):
        """
        Computes CE = (E[Y^(1-gamma)])^(1/(1-gamma))
        quantile_values shape: (Batch, K, 1)
        """
        # Ensure values are strictly positive to avoid NaNs
        val = torch.clamp(quantile_values.squeeze(-1), min=1e-12) # (Batch, K)
        
        if abs(self.gamma - 1.0) < 1e-4:
            # Geometric mean for gamma = 1
            log_val = torch.log(val)
            ce = torch.exp(log_val.mean(dim=1, keepdim=True))
        else:
            # Use Log-Sum-Exp trick for numerical stability to prevent float32 overflow:
            # CE = exp( (1 / (1-gamma)) * ( log(sum(exp((1-gamma)*log(val)))) - log(K) ) )
            K = val.shape[1]
            log_val = torch.log(val)
            x = (1.0 - self.gamma) * log_val # (Batch, K)
            
            # logsumexp over dim 1 (the quantile sample dimension)
            log_sum_exp = torch.logsumexp(x, dim=1, keepdim=True) # (Batch, 1)
            
            log_ce = (1.0 / (1.0 - self.gamma)) * (log_sum_exp - np.log(K))
            ce = torch.exp(log_ce)
            
        return ce # (Batch, 1)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
            
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        s, a_w, a_c, r, ns, d = zip(*batch)
        
        state = torch.stack(s)                  # (B, N, F)
        action_w = torch.stack(a_w)              # (B, N)
        action_c = torch.stack(a_c)              # (B, 1)
        reward = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device) # (B, 1)
        next_state = torch.stack(ns)            # (B, N, F)
        dones = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(self.device)  # (B, 1)
        
        # ----------------------------------------------------
        # 1. Critic Update (Quantile Bellman Operator)
        # ----------------------------------------------------
        with torch.no_grad():
            # (1) Next Action
            next_w, next_c = self.actor(next_state)
            
            # (2) Next Certainty Equivalent (estimated by target critic)
            next_taus = torch.rand(self.batch_size, self.n_quantiles, 1, device=self.device)
            next_quantiles = self.target_critic(next_state, next_w, next_c, next_taus) # (B, K, 1)
            ce_next = self.calc_certainty_equivalent(next_quantiles) # (B, 1)
            
            # (3) Non-linear Epstein-Zin value aggregator for next period
            power_sign = 1.0 - 1.0 / self.psi
            term1 = (1.0 - self.beta) * torch.pow(next_c, power_sign)
            term2 = self.beta * torch.pow(1.0 - next_c, power_sign) * torch.pow(ce_next, power_sign)
            v_next = torch.pow(torch.clamp(term1 + term2, min=1e-8), 1.0 / power_sign)
            
            # Boundary condition for done state: V_T = (1-beta)^(1/(1-1/psi)) * c_T
            term_cons = ((1.0 - self.beta) ** (1.0 / power_sign)) * next_c
            v_next_combined = (1.0 - dones) * v_next + dones * term_cons
            
            # (4) Targets: Y_t+1 = v(s_t+1) * R_gross_t+1
            # reward is the net return of portfolio (e.g. 0.005 = 0.5% return),
            # so R_gross = 1 + reward
            R_gross = 1.0 + reward
            targets = v_next_combined * R_gross # (B, 1)
            
        # (5) Current prediction
        curr_taus = torch.rand(self.batch_size, self.n_quantiles, 1, device=self.device)
        curr_quantiles = self.critic(state, action_w, action_c, curr_taus) # (B, K, 1)
        
        # (6) Quantile Huber Loss (pairwise mismatch mapping)
        # targets: (B, 1, 1), curr_quantiles: (B, K, 1) -> expand pairwise
        diff = targets.unsqueeze(2) - curr_quantiles.unsqueeze(1) # (B, 1, K, 1) -> squeeze to (B, 1, K)
        diff = diff.squeeze(-1) # (B, 1, K)
        
        huber = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
        
        tau_trans = curr_taus.squeeze(-1).unsqueeze(1) # (B, 1, K)
        weight = torch.abs(tau_trans - (diff.detach() < 0.0).float())
        critic_loss = (weight * huber).mean()
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # ----------------------------------------------------
        # 2. Actor Update (Maximize Lifetime Value)
        # ----------------------------------------------------
        pred_w, pred_c = self.actor(state)
        
        # Evaluate current policy under Critic
        actor_taus = torch.rand(self.batch_size, self.n_quantiles, 1, device=self.device)
        policy_quantiles = self.critic(state, pred_w, pred_c, actor_taus)
        ce_policy = self.calc_certainty_equivalent(policy_quantiles) # (B, 1)
        
        # Calculate utility aggregator
        term1_act = (1.0 - self.beta) * torch.pow(pred_c, power_sign)
        term2_act = self.beta * torch.pow(1.0 - pred_c, power_sign) * torch.pow(ce_policy, power_sign)
        v_policy = torch.pow(torch.clamp(term1_act + term2_act, min=1e-8), 1.0 / power_sign)
        
        # Loss is negative normalized value
        actor_loss = -v_policy.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Soft update target network
        self.soft_update()
        
        return critic_loss.item(), actor_loss.item()

    def soft_update(self):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
