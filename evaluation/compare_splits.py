"""Evaluate an agent on both Validation and Test sets, then plot the comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from envs.trading import TradingEnv
from evaluation.plots import plot_agent_vs_baselines
from evaluation.policy_gradient.evaluate import buy_and_hold_baseline
from features import OHLCVWithIndicators, RawOHLCV

def run_evaluation(df: pd.DataFrame, env_cls, fb, reward, agent_type, agent, args) -> dict[str, float]:
    env = env_cls(df=df, feature_builder=fb, reward_scheme=reward, max_episode_steps=None)
    
    # Evaluate Agent
    if agent_type == "pmdp":
        from evaluation.partial_mdp.evaluate import evaluate as evaluate_pmdp
        results = evaluate_pmdp(env, agent)
    else:
        from evaluation.policy_gradient.evaluate import evaluate as evaluate_ppo
        results = evaluate_ppo(env, agent)
        
    agent_ret = results[0]["cumulative_return"]
    
    # Evaluate Baselines
    bh = buy_and_hold_baseline(env)
    bh_ret = bh["cumulative_return"]
    
    return {
        "Agent": agent_ret,
        "Buy & Hold": bh_ret
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare agent vs baselines on Val and Test")
    parser.add_argument("--ticker", default="CHWY")
    parser.add_argument("--agent", choices=["ppo", "pmdp"], default="pmdp")
    parser.add_argument("--features", choices=["raw", "indicators"], default="indicators")
    parser.add_argument("--reward", choices=["simple", "sharpe", "sortino", "action_simple", "action_sharpe", "action_sortino", "event_based"], default="event_based")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    args = parser.parse_args()

    # Data
    val_df = pd.read_csv(f"data/processed/{args.ticker}_val.csv", index_col=0, parse_dates=["Date"])
    test_df = pd.read_csv(f"data/processed/{args.ticker}_test.csv", index_col=0, parse_dates=["Date"])
    
    # Feature Builder
    fb = RawOHLCV(window_size=20) if args.features == "raw" else OHLCVWithIndicators(window_size=20)

    # Initialize Environment to get dims
    env_tmp = TradingEnv(df=val_df, feature_builder=fb, reward_scheme=args.reward)
    obs_dim = int(env_tmp.observation_space.shape[0])
    act_dim = int(env_tmp.action_space.n)
    
    # Load Agent
    if args.agent == "pmdp":
        from agents.partial_mdp.agent import PMDPAgent
        config = {"gamma": 0.99, "lr": 3e-4, "hidden": 128, "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64}
        agent = PMDPAgent(obs_dim, act_dim, config)
    else:
        from agents.policy_gradient.ppo import PPOAgent
        config = {"gamma": 0.99, "lr": 3e-4, "hidden": 128, "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64, "entropy_coef": 0.05}
        agent = PPOAgent(obs_dim, act_dim, config)
        
    agent.load(Path(args.checkpoint))
    print(f"Loaded {args.agent.upper()} from {args.checkpoint}")
    
    print("\n--- Evaluating on Validation Set ---")
    val_metrics = run_evaluation(val_df, TradingEnv, fb, args.reward, args.agent, agent, args)
    
    print("\n--- Evaluating on Test Set ---")
    test_metrics = run_evaluation(test_df, TradingEnv, fb, args.reward, args.agent, agent, args)
    
    print("\n--- Summary ---")
    for k in val_metrics.keys():
        print(f"{k:12s} | Val Return: {val_metrics[k]:+8.2%} | Test Return: {test_metrics[k]:+8.2%}")
    
    save_path = Path(args.checkpoint) / "val_vs_test_bar.png"
    plot_agent_vs_baselines(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        metric_name="Cumulative Return",
        title=f"{args.ticker} Returns: Agent vs Baselines",
        save_path=str(save_path)
    )
