import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_reward(monitor_df, logdir):
    rewards = monitor_df["r"].to_numpy()
    steps_ep = monitor_df["l"].to_numpy()
    cum_steps = np.cumsum(steps_ep)

    # 移動平均も併記
    window = 10
    rewards_smooth = (
        pd.Series(rewards).rolling(window=window, min_periods=1).mean().to_numpy()
    )

    plt.figure(figsize=(6, 4))
    plt.plot(cum_steps, rewards, alpha=0.3, label="raw")
    plt.plot(cum_steps, rewards_smooth, linewidth=2, label=f"MA({window})")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward vs Training Step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "reward_curve.png"))
    plt.close()


def _plot_loss(loss_df, logdir):
    plt.figure(figsize=(6, 4))
    for col in ["train/policy_loss", "train/value_loss", "train/manip_loss"]:
        if col in loss_df.columns:
            plt.plot(loss_df[col], label=col.split("/")[-1])
    plt.xlabel("Rollout")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "loss_curve.png"))
    plt.close()


def save_reward_and_loss_plots(logdir: str,
                               monitor_csv: str,
                               loss_csv: str | None = None):
    """monitor.csv と loss_history.csv から図を作成・保存"""
    monitor_df = pd.read_csv(monitor_csv, skiprows=1)
    _plot_reward(monitor_df, logdir)

    if loss_csv and os.path.exists(loss_csv):
        loss_df = pd.read_csv(loss_csv)
        _plot_loss(loss_df, logdir)
