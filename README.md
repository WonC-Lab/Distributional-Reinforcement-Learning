# Deep Distributional Reinforcement Learning for Dynamic Asset Allocation under Epstein-Zin Recursive Utility

Developed by **WonChan Cho** (Department of Mathematics, Sungkyunkwan University).

---

## Project Overview

This repository contains the implementation of a mathematically rigorous framework combining **Deep Distributional Reinforcement Learning (DRL)** with **Epstein-Zin (EZ) recursive preferences** to solve the dynamic, high-dimensional consumption-portfolio choice problem in the presence of transaction costs.

By utilizing an **Implicit Quantile Network (IQN)** to learn the full utility-adjusted return distribution, this framework decoupling the coefficient of relative risk aversion ($\gamma$) and the intertemporal elasticity of substitution (IES, $\psi$), allowing for independent tuning of risk-hedging and consumption-smoothing behavior.

---

## Theoretical Framework

### 1. Epstein-Zin Recursive Utility
We consider an investor on a filtered probability space $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t=0}^\infty, \mathbb{P})$ who dynamically chooses a consumption fraction $c_t \equiv C_t/W_t \in (0, 1)$ and portfolio weights $\mathbf{w}_t \in \Delta^N$. The lifetime recursive utility is defined as:

$$V_t = \left[ (1 - \beta) C_t^{1 - \frac{1}{\psi}} + \beta \left( \mathcal{R}_t(V_{t+1}) \right)^{1 - \frac{1}{\psi}} \right]^{\frac{1}{1 - \frac{1}{\psi}}}$$

where $\mathcal{R}_t(X)$ is the conditional certainty equivalent operator under risk aversion $\gamma$:

$$\mathcal{R}_t(X) = \left( \mathbb{E}_t \left[ X^{1 - \gamma} \right] \right)^{\frac{1}{1 - \gamma}}$$

### 2. Main Theoretical Results (Summarized)

Below are the key mathematical results establishing the convergence, coherence, and boundary conditions of our framework. *(Note: Complete step-by-step proofs are withheld in the private manuscript for intellectual property protection).*

#### Theorem 1 (Wealth-Homotheticity)
Under the dynamic wealth constraint $W_{t+1} = W_t (1 - c_t) R_{p, t+1}$, the value function scales linearly with wealth:
$$V_t(W_t, \mathbf{s}_t) = W_t v_t(\mathbf{s}_t)$$
where $v_t(\mathbf{s}_t)$ satisfies the wealth-normalized recursive Bellman equation. This justifies solving the portfolio optimization in weight space independently of absolute wealth.

#### Lemma 1 (Quantile Representation of Certainty Equivalent)
Let $q_{Y}(\tau) \equiv F_{Y}^{-1}(\tau)$ be the quantile function of the utility-adjusted return $Y_{t+1}$. The certainty equivalent is equivalent to the Lebesgue integral over the quantile space:
$$\mathcal{R}_t(Y_{t+1}) = \left( \int_0^1 \left( q_Y(\tau) \right)^{1 - \gamma} d\tau \right)^{\frac{1}{1 - \gamma}}$$
This guarantees that training the Critic on quantile temporal difference loss (IQN) corresponds directly to the expected recursive utility target.

#### Theorem 2 (Contraction Property of the EZ Bellman Operator)
Let $\mathcal{T}$ be the wealth-normalized Epstein-Zin Bellman operator in the quantile space. If:
$$\kappa \equiv \beta \cdot \sup_{s, \mathbf{w}} \left( \mathbb{E} \left[ R_p(\mathbf{w})^{1-\gamma} \mid s \right] \right)^{\frac{1 - 1/\psi}{1 - \gamma}} < 1$$
then the operator $\mathcal{T}$ is a local contraction mapping on the Banach space of continuous functions under the supremum norm, ensuring the existence of a unique optimal value function $v^*$.

#### Theorem 3 (Transaction-Cost-Induced No-Trade Region)
Given a turnover transaction cost rate $cost > 0$, the optimal policy implements a no-trade region where the investor chooses not to rebalance ($\mathbf{w}_t^* = \mathbf{w}_{t-1}^{\text{adj}}$) if and only if:
$$\left\| \nabla_{\mathbf{w}} v_t(\mathbf{s}_t) \right\|_2 < \text{cost} \cdot (1 - c_t^*)$$
where $c_t^*$ is the optimal consumption rate.

---

## Methodological Architecture

### 1. AssetTransformer (Permutation Invariance)
To process arbitrary dimensions of asset pools without index-order bias, we implement a self-attention transformer network over the asset dimension:
$$\mathbf{f}_i = \text{LayerNorm}\left( \mathbf{e}_i + \text{MultiHeadAttention}(\mathbf{e}_i, \{\mathbf{e}_j\}_{j=1}^N) \right)$$

### 2. Multi-Headed Actor
* **Weight Head ($\mathbf{w}_t$)**: Outputs raw scores per asset, mapped to portfolio weights via a Softmax layer to satisfy the budget constraint ($\sum w_i = 1$).
* **Consumption Head ($c_t$)**: Pools global asset features and projects them via an MLP and a scaled Sigmoid function to map the consumption rate dynamically between $1\%$ and $20\%$ of wealth.

### 3. Numerically Stable Log-Sum-Exp Certainty Equivalent
Under high risk aversion ($\gamma = 8.0$) and low IES ($\psi = 0.5$), $Y^{1-\gamma}$ overflows float32 limits ($> 3.4 \times 10^{38}$). We solve this using a stable Log-Sum-Exp formulation:
$$\mathcal{R}_t(Y_{t+1}) = \exp \left( \frac{1}{1 - \gamma} \left( \text{log\_sum\_exp}(\{(1-\gamma) \ln Y_k\}_{k=1}^K) - \ln K \right) \right)$$

---

## Execution Guide

### 1. Prerequisites
Ensure you have the required packages installed:
```bash
pip install torch numpy pandas yfinance matplotlib
```

### 2. Training the Model
To run the grid search across combinations of risk aversion ($\gamma \in \{2.0, 8.0\}$) and IES ($\psi \in \{0.5, 1.5\}$):
```bash
python run_experiment.py
```
This script downloads asset and macro-financial variables, runs the training iterations, and logs out-of-sample trajectories to local CSV files.

### 3. Plotting Results
To generate publication-quality figures representing the out-of-sample portfolio growth, asset allocation, and consumption dynamics:
```bash
python plot_academic_results.py
```

---

## Repository Structure

* `dynamic_ez_env.py`: Market simulator with Yahoo Finance download and transaction cost calculation.
* `dynamic_ez_agent.py`: Actor, Critic, and Log-Sum-Exp solver implementations.
* `run_experiment.py`: Main script for training loop and grid search.
* `plot_academic_results.py`: Script to generate result plots.
* `.gitignore`: Configured to exclude all private paper drafts (`*.tex`), local data caching (`*.csv`), and result images (`*.png`).
