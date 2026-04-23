# Deep RL Quant Trading Agents for Partially Observable MDPs

A research codebase for **deep reinforcement learning on single-asset stock trading**: custom [Gymnasium](https://gymnasium.farama.org/) environments, multiple agent families, technical-indicator features, and evaluation tooling for train / validation / test splits on real OHLCV data.

> **Disclaimer.** This is an educational project. Past performance of learned policies does not imply future results, and the environment is a simplified model of real markets (discrete daily actions, frictions, no slippage modeling beyond a fixed commission).

---

## What’s inside

| Area | Description |
|------|-------------|
| **Environment** | `TradingEnv` — discrete actions (Hold, Buy, Sell), portfolio accounting, action masking, multiple reward schemes. |
| **Features** | `RawOHLCV` or `OHLCVWithIndicators` (RSI, MACD, Bollinger, ATR, etc.) with a configurable lookback `window_size`. |
| **Value-based** | DQN, Double DQN, **Rainbow** (C51 + PER + n-step + dueling + noisy nets) — see [`agents/value_based/README.md`](agents/value_based/README.md). |
| **Policy gradient** | PPO for discrete actions. |
| **Partial-MDP** | LSTM state predictor + PPO-style actor-critic (“LSTM-PPO”). |
| **Evaluation** | Frozen-policy evaluation, bar charts, trajectories, training curves, comparison utilities — under `evaluation/`. |
| **Notebooks** | `notebooks/` — exploratory DQN, Double DQN, Rainbow. |

Checkpoints are written under `runs/{variant}_{TICKER}_{features}_{reward}/`.

---

## Repository layout

```
agents/          # DQN, Rainbow, PPO, PMDP (LSTM) agents + training CLIs
envs/            # Trading environment + reward schemes
features/        # State construction (raw OHLCV vs indicators)
data/            # Download (yfinance) + chronological train/val/test CSVs
evaluation/      # Metrics, plots, portfolio trajectory, bar charts, etc.
tests/           # Pytest (e.g. environment sanity)
notebooks/       # Jupyter walkthroughs
```

---

## Setup

**Requirements:** Python **3.10+** (see `pyproject.toml`).

```bash
cd DRL-for-Trading
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional dev tools (used in CI):

```bash
pip install ruff pytest pytest-cov
```

Run tests:

```bash
pytest tests/ -v
```

CI (`.github/workflows/ci.yaml`) runs **Ruff** lint/format checks and **pytest** on Python 3.10–3.12.

---

## Data

1. **Download** tickers and **split** chronologically into `data/processed/{TICKER}_{train|val|test}.csv` (avoids look-ahead in time-series evaluation).

```bash
python -m data.download
```

Default tickers are defined in `data/download.py` (e.g. liquid names + `CHWY`). Adjust the list and date range there if needed.

2. For more detail, see [`data/README.md`](data/README.md).

---

## Training

All training entry points read **`data/processed/{TICKER}_train.csv`**.

### Value-based (DQN / Double DQN / Rainbow)

```bash
# Double DQN
python -m agents.value_based.train \
  --agent dqn --double_dqn \
  --ticker AMZN --features indicators --reward portfolio_delta \
  --window_size 20 --episodes 500

# Rainbow
python -m agents.value_based.train \
  --agent rainbow \
  --ticker AMZN --features indicators --reward event_based \
  --window_size 20 --episodes 500 --hidden_rainbow 64
```

- **`--schedule`**: `random` (default) or `sliding` for overlapping window episodes.  
- **`--max_episode_steps`**: if set and smaller than the data span, `reset()` can randomize episode starts; use `None` or a very large value for full-sequence episodes (see `envs/trading.py`).

Saves, e.g. `runs/double_dqn_AMZN_indicators_portfolio_delta/` or `runs/rainbow_AMZN_indicators_event_based/`.

### PPO

```bash
python -m agents.policy_gradient.train \
  --ticker AMZN --features indicators --reward event_based --episodes 500
```

Note: PPO’s training script hardcodes `window_size=20` in the feature builder; keep evaluation consistent.

### PMDP (LSTM predictor + PPO-style AC)

```bash
python -m agents.partial_mdp.train \
  --ticker AMZN --features indicators --reward event_based --episodes 500
```

Two phases: LSTM pretraining, then actor-critic training (see `agents/partial_mdp/train.py`).

---

## Evaluation

**Value-based (DQN / Rainbow)** — one-shot rollout metrics + optional behavior plot:

```bash
python -m evaluation.value_based.evaluate \
  --agent rainbow \
  --checkpoint runs/rainbow_AMZN_indicators_event_based \
  --ticker AMZN --split test \
  --features indicators --reward event_based --window_size 20 \
  --n_atoms 51 --v_min -10 --v_max 10 --hidden_rainbow 64
```

**PPO** and **PMDP** have analogous modules under `evaluation/policy_gradient/evaluate.py` and `evaluation/partial_mdp/evaluate.py`.

**Important:** At eval time, **`--window_size`**, **Rainbow** hyperparameters (`n_atoms`, `v_min`, `v_max`, `hidden_rainbow`), and **feature mode** must match training so observation dimension and network shapes align.

---

## Analysis & figures

| Script | Role |
|--------|------|
| `evaluation/plot_portfolio.py` | Multi-agent **portfolio trajectory** (%) vs buy-and-hold; infers `window_size` from each checkpoint when possible. |
| `evaluation/plot_training_curve.py` | Training / learning curves. |
| `evaluation/plot_returns_bar.py` | Cumulative return bar charts. |
| `evaluation/plot_cumulative_rewards.py` | Per-step reward comparison. |
| `evaluation/compare_*.py` | Reward or split comparisons. |

Example (trajectory plot):

```bash
python -m evaluation.plot_portfolio --ticker AMZN --split test
```

Figures are typically written to `runs/analysis_plots/` or `images/` depending on the script.

---

## Hyperparameter consistency (avoid shape errors)

- Use the **same** `--window_size` and **features** (`raw` vs `indicators`) for training and all plots/eval.  
- **Rainbow:** `--hidden_rainbow`, `--n_atoms`, `--v_min`, `--v_max` at eval must match the trained run.  
- **Checkpoint path** encodes `runs/{variant}_{TICKER}_{features}_{reward}/` — the reward name in the path must match the folder you trained.

---

