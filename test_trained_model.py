import os
import sys
import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from unitree_pybullet.unitree_pybullet import QuadEnv
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, DummyVecEnv

from src.models import ExtendModel
from src.SkipFrame import SkipFrame


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
        help="テスト結果（報酬の推移等）を保存するディレクトリ"
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
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="True の場合 determinisitc=True でモデルを実行（再現性重視）。",
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
        calculate_manip = (cfg.calculate_manip or cfg.use_manip_loss),
    )
    os.makedirs(cfg.results_dir, exist_ok=True)
    #env = SkipFrame(env, skip=cfg.skip_freq)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


if __name__ == "__main__":
    args = parse_args()

    # 設定ファイル読み込み (.hydra/config.yaml を参照)
    base_dir   = args.config_dir or os.path.dirname(args.model_path)
    hydra_dir  = os.path.join(base_dir, ".hydra")
    config_path = os.path.join(hydra_dir, "config.yaml")
    stats_path  = os.path.join(base_dir, "vecnormalize.pkl")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    # テスト結果出力先だけ上書き
    cfg.results_dir = args.results_dir

    # 環境作成
    env = make_env(cfg)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(stats_path, env)

    # 評価モード（統計更新停止）
    env.training = False
    """
    # rewardの逆正規化を行って生報酬を得たければ以下を False に
    # （学習時と同じ正規化スケールでログするなら True のまま）
    # env.norm_reward = False
    """
    env = VecMonitor(env, filename=os.path.join(cfg.results_dir, "monitor.csv"))

    # モデル読み込み
    # model = PPO.load(args.model_path, env=env)
    model = ExtendModel.load(args.model_path, env= env, device = "cpu", model_name = cfg.algo)


    # ------------------------- リアルタイム同期 -------------------------
    if args.realtime:
        # VecEnv 経由で基礎シミュレーション周期を取得
        base_dt = env.get_attr("dt")[0]          # QuadEnv.dt (=1/400)
        try:
            # SkipFrame ラッパが存在すればその skip を優先
            skip = env.get_attr("skip")[0]
        except Exception:
            # ラッパなし → cfg が持っていれば使用，なければ 1
            skip = getattr(cfg, "skip_freq", 1) if hasattr(cfg, "skip_freq") else 1
        skip   = max(int(skip), 1)
        eff_dt = base_dt * skip

        wall_start   = time.time()
        last_log_wall = wall_start
        sim_time     = 0.0
        speed        = args.speed if args.speed > 0 else 1.0

    # ------------------------- エピソードループ -------------------------
    # VecEnv なので obs.shape = (1, obs_dim)
    for ep in range(1, args.episodes + 1):
        seed = cfg.seed + ep
        env.seed(seed)
        obs  = env.reset()
        ep_ret = 0.0
        ep_len = 0
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            step_out  = env.step(action)
            # Gym / Gymnasium 両対応
            if len(step_out) == 5:  # Gymnasium
                obs, reward, terminated, truncated, infos = step_out
                done   = terminated[0] or truncated[0]
                rew_scalar = reward
            else:                   # Classic Gym
                obs, reward, done, infos = step_out
                rew_scalar = reward if hasattr(reward, "__getitem__") else reward
            ep_ret += float(reward)
            ep_len += 1

            # ------ 等速同期 ------
            if args.realtime:
                sim_time += eff_dt
                wall_elapsed = time.time() - wall_start
                target_wall = sim_time / speed
                sleep_needed = target_wall - wall_elapsed
                if sleep_needed > 5e-4:
                    time.sleep(sleep_needed)
                if args.sync_log_interval > 0:
                    now = time.time()
                    if now - last_log_wall >= args.sync_log_interval:
                        ratio = (sim_time / speed) / (now - wall_start + 1e-9)
                        print(f"[SYNC] sim_time={sim_time:.2f}s wall={now - wall_start:.2f}s "
                              f"speed={speed:.2f} ratio={ratio:.3f}")
                        last_log_wall = now

            if done:
                break

        print(f"Episode {ep:02d}: total_reward={ep_ret:.2f} len={ep_len}")

    env.close()
