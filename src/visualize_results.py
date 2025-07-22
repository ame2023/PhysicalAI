# visualize_results.py
"""
学習/テストで出力された monitor.csv (VecMonitor/Monitor) を読み込み、
エピソード毎の累積報酬(r) とエピソード長(l) を可視化する簡易スクリプト。

使用例:
    python visualize_results.py --monitor-csv outputs/2025-07-22/00-07-25/monitor.csv
    python visualize_results.py --monitor-csv my_test_results/monitor.csv --save-png

SB3 Monitorフォーマットは先頭行に "#{"..." : ...}" JSON メタデータ、
2行目に "r,l,t" ヘッダ（VecMonitorの場合は追加列）を含むため、
pandas.read_csv(..., comment="#") で読むのが安全。
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--monitor-csv", "-f",
        type=str,
        default=None,
        help="読み込む monitor.csv のパス（省略時は outputs/以下から最新を自動検出）",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="pngファイルとして保存（<monitor.csvと同階層>/reward.png, len.png）。",
    )
    return parser.parse_args()


def find_latest_monitor(outputs_dir: Path) -> Path:
    """outputsディレクトリ下で最新(更新時刻)の monitor.csv を返す。無ければ例外。"""
    candidates = list(outputs_dir.rglob("monitor.csv"))
    if not candidates:
        raise FileNotFoundError(f"monitor.csv が {outputs_dir} 以下に見つかりません。")
    # 最終更新時刻でソート
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_monitor_csv(path: Path) -> pd.DataFrame:
    """
    SB3 Monitor/VecMonitor の CSV を読み込む。
    comment="#" でメタ情報行スキップ。
    列名は自動取得し、必須列 'r','l' をチェック。
    """
    df = pd.read_csv(path, comment="#")
    if "r" not in df.columns or "l" not in df.columns:
        raise ValueError(f"'r' または 'l' 列が見つかりません: {path}")
    return df


def plot_series(y, title, ylabel, save_path=None):
    x = range(len(y))
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("episode")
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    monitor_path = Path(args.monitor_csv) if args.monitor_csv else find_latest_monitor(outputs_dir)
    if not monitor_path.is_absolute():
        monitor_path = repo_root / monitor_path       
    monitor_path = monitor_path.resolve()
    df = load_monitor_csv(monitor_path)

    # 報酬
    plot_series(
        df["r"].astype(float).values,
        title=f"Episode Reward ({monitor_path.parent.name})",
        ylabel="reward",
        save_path=(monitor_path.parent / "reward.png") if args.save_png else None,
    )

    # エピソード長
    plot_series(
        df["l"].astype(float).values,
        title=f"Episode Length ({monitor_path.parent.name})",
        ylabel="episode len",
        save_path=(monitor_path.parent / "len.png") if args.save_png else None,
    )

    print(f"Loaded monitor: {monitor_path}  episodes={len(df)}")


if __name__ == "__main__":
    main()
