import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from unitree_pybullet.unitree_pybullet import QuadEnv

from omegaconf import OmegaConf

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

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep:02d}: total_reward = {total_reward:.2f}")

    env.close()
