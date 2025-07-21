import os
import sys
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from unitree_pybullet.unitree_pybullet import QuadEnv
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", "-m",
        type=str, required=True,
        help="読み込むPPOモデルのパス（例: outputs/.../ppo_a1.zip）"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str, default="./my_test_results",
        help="テスト結果（リワード推移等）を保存するディレクトリ"
    )
    parser.add_argument(
        "--config-dir", "-c",
        type=str, default=None,
        help="Hydra出力ディレクトリ（例: outputs/.../00-40-36）。省略時は --model-path の親ディレクトリを使用"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int, default=1,
        help="テストするエピソード数"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="シミュレーションを等速(壁時計)に同期する"
    )
    parser.add_argument(
        "--speed",
        type=float, default=1.0,
        help="再生速度倍率 (1.0=等速, 0.5=スロー, 2.0=2倍速) ※ --realtime 指定時有効"
    )
    parser.add_argument(
        "--sync-log-interval",
        type=float, default=5.0,
        help="同期状態(ratio)をログする秒間隔（実時間）。0以下で無効。"
    )
    return parser.parse_args()


def make_env(cfg, seed: int = 0):
    env = QuadEnv(
        model=cfg.unitree_model,
        render=cfg.render,
        max_steps_per_episode=cfg.max_steps_per_episode,
        fall_height_th=cfg.fall_height_th,
        fall_angle_th=cfg.fall_angle_th,
        obs_mode=cfg.obs_mode,
        action_scale_deg=cfg.action_scale_deg,
        control_mode=cfg.control_mode,
        torque_scale_Nm=cfg.torque_scale_Nm,
        reward_mode=cfg.reward_mode,
    )
    os.makedirs(cfg.results_dir, exist_ok=True)
    env = VecNormalize(env, 
                       training = False,
                       norm_obs=True,
                       norm_reward=True,
                       clip_obs=10.)
    env = Monitor(env,
                  filename=os.path.join(cfg.results_dir, "monitor.csv"))
    env.reset(seed=seed)
    return env


if __name__ == "__main__":
    args = parse_args()

    # 設定ファイル読み込み (.hydra/config.yaml を参照)
    base_dir = args.config_dir or os.path.dirname(args.model_path)
    hydra_dir = os.path.join(base_dir, ".hydra")
    config_path = os.path.join(hydra_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    # テスト結果出力先だけ上書き
    cfg.results_dir = args.results_dir

    # モデル読み込み
    model = PPO.load(args.model_path)

    # 環境作成
    env = make_env(cfg)
    env = VecNormalize.load("vecnormalize.pkl", env)

    # リアルタイム同期用初期化
    if args.realtime:
        if not hasattr(env, "dt"):
            raise AttributeError("env に dt 属性が必要です（QuadEnv.dt を参照）。")
        wall_start = time.time()
        last_log_wall = wall_start
        sim_time = 0.0
        speed = args.speed if args.speed > 0 else 1.0

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # --- リアルタイム同期制御 ---
            if args.realtime:
                sim_time += env.dt  # 1 step で dt だけ進む前提
                wall_elapsed = time.time() - wall_start
                target_wall = sim_time / speed
                sleep_needed = target_wall - wall_elapsed
                # 過剰コンテキストスイッチを避けるため閾値を設定
                if sleep_needed > 5e-4:
                    time.sleep(sleep_needed)
                if args.sync_log_interval > 0:
                    now = time.time()
                    if now - last_log_wall >= args.sync_log_interval:
                        ratio = (sim_time / speed) / (now - wall_start + 1e-9)
                        print(f"[SYNC] sim_time={sim_time:.2f}s wall={now-wall_start:.2f}s "
                              f"speed={speed:.2f} ratio={ratio:.3f}")
                        last_log_wall = now

        print(f"Episode {ep:02d}: total_reward = {total_reward:.2f}")

    env.close()
