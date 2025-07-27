import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _plot_reward(monitor_df, logdir):
    rewards = monitor_df["r"].to_numpy()
    steps_ep = monitor_df["l"].to_numpy()
    cum_steps = np.cumsum(steps_ep)

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
    for col in ["train/policy_gradient_loss", "train/value_loss", "train/entropy_loss", "train/manip_loss"]:
        if col in loss_df.columns:
            plt.plot(loss_df[col], label=col.split("/")[-1])
    plt.xlabel("Rollout/Update")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, "loss_curve.png"))
    plt.close()


def _plot_manip(manip_df: pd.DataFrame, logdir: str, agg: str = "mean"):
    """
    可操作度ログ(manip_history.csv)を、
    - 環境平均を一切取らず（= 行の追加順 = 学習順）に seq 軸で連結して可視化
      1) 各脚ごとの推移 (manip_curve_legs_seq.png)
      2) 各行で4脚を agg で集約した推移 (manip_curve_{agg}_seq.png)
    """
    leg_cols = [c for c in manip_df.columns if c.startswith("leg_")]
    # 旧形式（mean/min/maxしかない）へのフォールバック
    if len(leg_cols) == 0:
        steps = manip_df["step"].to_numpy()
        mean_v = manip_df["mean"].to_numpy()
        plt.figure(figsize=(6, 4))
        plt.plot(steps, mean_v, label="manip_mean")
        plt.xlabel("Global Step")
        plt.ylabel("Manipulability (mean)")
        plt.title("Manipulability vs Training Step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, "manip_curve.png"))
        plt.close()
        return

    # 並列環境方向の平均は取らず、ログ行の順序で可視化する
    df = manip_df.reset_index(drop=True).copy()
    df["seq"] = np.arange(len(df), dtype=np.int64)

    # 1) 各脚をそのまま seq 軸で描画
    for leg_col in leg_cols:  # 例: leg_0, leg_1, leg_2, leg_3
        plt.figure(figsize=(6, 4))
        plt.plot(df["seq"], df[leg_col], linewidth=1.0)
        plt.xlabel("Logged Step (seq)")
        plt.ylabel("Manipulability")
        plt.title(f"Manipulability ({leg_col})")
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, f"manip_curve_{leg_col}_seq.png"))
        plt.close()

    # 2) 4脚を 'agg' で1行ずつ集約
    if agg not in {"mean", "min", "max", "sum"}:
        raise ValueError(f"Unknown agg: {agg}")

    mat = df[leg_cols].to_numpy()
    if agg == "mean":
        agg_series = mat.mean(axis=1)
    elif agg == "min":
        agg_series = mat.min(axis=1)
    elif agg == "max":
        agg_series = mat.max(axis=1)
    else:  # sum
        agg_series = mat.sum(axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(df["seq"], agg_series, label=f"legs.{agg}", linewidth=1.2)
    plt.xlabel("Logged Step (seq)")
    plt.ylabel(f"Manip ({agg})")
    plt.title(f"Manipulability ({agg} across 4 legs, no env averaging)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f"manip_curve_{agg}_seq.png"))
    plt.close()


def save_reward_and_loss_plots(logdir: str,
                               monitor_csv: str,
                               loss_csv: str | None = None,
                               manip_csv: str | None = None,
                               manip_agg: str = "mean"):
    monitor_df = pd.read_csv(monitor_csv, skiprows=1)
    _plot_reward(monitor_df, logdir)

    if loss_csv and os.path.exists(loss_csv):
        loss_df = pd.read_csv(loss_csv)
        _plot_loss(loss_df, logdir)

    if manip_csv is None:
        manip_csv = os.path.join(logdir, "manip_history.csv")
    if os.path.exists(manip_csv):
        manip_df = pd.read_csv(manip_csv)
        _plot_manip(manip_df, logdir, agg=manip_agg)
