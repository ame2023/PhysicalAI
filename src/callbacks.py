# src/callbacks.py
import os
import numpy as np
import pandas as pd
from collections import deque
from typing import Deque, Tuple, Optional, Any
from stable_baselines3.common.callbacks import BaseCallback


class LossLoggerCallback(BaseCallback):
    """
    SB3のloggerから train/ で始まるキーを拾って CSV に保存
    """
    def __init__(self, logdir: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.logdir = logdir
        self.records = []

    def _on_step(self) -> bool:
        try:
            kv = self.logger.get_log_dict()
        except Exception:
            return True
        row = {k: v for k, v in kv.items() if isinstance(k, str) and k.startswith("train/")}
        if row:
            self.records.append(row)
        return True

    def _on_training_end(self) -> None:
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        out = os.path.join(self.logdir, "loss_history.csv")
        df.to_csv(out, index=False)
        if self.verbose > 0:
            print(f"[LossLoggerCallback] wrote {out}")


class ManipLoggerCallback(BaseCallback):
    """
    各 step の info["manip_w"]（4脚の可操作度）をCSVに保存。
    VecEnv を想定。env_id/episode_id/seq なども付与。
    """
    def __init__(self, logdir: str, use_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.logdir = logdir
        self.use_wandb = use_wandb
        self.rows = []
        self.episode_id = None
        self.t_in_episode = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.episode_id = np.zeros(n_envs, dtype=np.int64)
        self.t_in_episode = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        if infos is None:
            return True

        step = self.num_timesteps
        for env_id, info in enumerate(infos):
            if info is None or "manip_w" not in info:
                continue
            w = np.asarray(info["manip_w"], dtype=float).ravel()
            row = {
                "step": step,
                "env": env_id,
                "episode_id": int(self.episode_id[env_id]),
                "t_in_episode": int(self.t_in_episode[env_id]),
                "leg_0": float(w[0]) if w.size > 0 else np.nan,
                "leg_1": float(w[1]) if w.size > 1 else np.nan,
                "leg_2": float(w[2]) if w.size > 2 else np.nan,
                "leg_3": float(w[3]) if w.size > 3 else np.nan,
            }
            self.rows.append(row)

        if dones is not None:
            for env_id, d in enumerate(dones):
                if bool(d):
                    self.episode_id[env_id] += 1
                    self.t_in_episode[env_id] = 0
                else:
                    self.t_in_episode[env_id] += 1

        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        out = os.path.join(self.logdir, "manip_history.csv")
        df.to_csv(out, index=False)
        if self.verbose > 0:
            print(f"[ManipLoggerCallback] wrote {out}")


class TrainEpisodeStatsCallback(BaseCallback):
    """
    学習中のエピソード統計（平均報酬・平均長）を logger に記録。
    → SB3 本体の dump タイミングの表に一緒に出すため、ここでは dump しない。
    Gym/Gymnasium 両対応：info["episode"] または info["final_info"]["episode"] を拾う。
    """
    def __init__(self, window_size: int = 100, log_interval_steps: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.window_size = window_size
        self.log_interval_steps = int(log_interval_steps)
        self.rew_buf: Deque[float] = deque(maxlen=window_size)
        self.len_buf: Deque[int] = deque(maxlen=window_size)

    def _extract_episode(self, info: Any) -> Optional[Tuple[float, int]]:
        if not info:
            return None
        if "episode" in info and isinstance(info["episode"], dict):
            ep = info["episode"]
            if "r" in ep and "l" in ep:
                return float(ep["r"]), int(ep["l"])
        fi = info.get("final_info", None)
        if isinstance(fi, dict) and "episode" in fi:
            ep = fi["episode"]
            if "r" in ep and "l" in ep:
                return float(ep["r"]), int(ep["l"])
        return None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        for inf in infos:
            ep = self._extract_episode(inf)
            if ep is None:
                continue
            r, l = ep
            self.rew_buf.append(r)
            self.len_buf.append(l)
        return True

    def _on_rollout_end(self) -> None:
        if len(self.rew_buf) == 0:
            return
        self.logger.record("rollout/ep_rew_mean", np.mean(self.rew_buf))
        self.logger.record("rollout/ep_len_mean", np.mean(self.len_buf))
        # dump は SB3 本体が直後に行う（表に一緒に出る）

class EvalCallbackWithVec(BaseCallback):
    """
    EvalCallback 相当の機能に加えて、ベスト更新時に VecNormalize を保存する軽量版。
    """
    def __init__(self, eval_env, save_path: str, train_vecnorm, eval_freq: int = 10000,
                 n_eval_episodes: int = 5, deterministic: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.train_vecnorm = train_vecnorm
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.best_mean = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        # 学習側VecNormalizeの統計を評価側に同期
        try:
            self.eval_env.obs_rms = self.train_vecnorm.obs_rms
            self.eval_env.ret_rms = self.train_vecnorm.ret_rms
            self.eval_env.clip_obs = self.train_vecnorm.clip_obs
            self.eval_env.clip_reward = getattr(self.train_vecnorm, "clip_reward", np.inf)
            self.eval_env.norm_obs = self.train_vecnorm.norm_obs
            self.eval_env.norm_reward = self.train_vecnorm.norm_reward
        except Exception as e:
            if self.verbose:
                print(f"[WARN] failed to sync VecNormalize: {e}")

        # 評価
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_r, std_r = evaluate_policy(self.model, self.eval_env,
                                        n_eval_episodes=self.n_eval_episodes,
                                        deterministic=self.deterministic, render=False)
        self.logger.record("eval/mean_reward", mean_r)
        self.logger.record("eval/mean_reward_std", std_r)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)

        # ベスト更新なら保存（既存の通り）
        if mean_r > self.best_mean:
            self.best_mean = mean_r
            print(f"update best_mean_reward:{self.best_mean}, save best_model")
            self.model.save(os.path.join(self.save_path, "best_model.zip"))
            try:
                self.train_vecnorm.save(os.path.join(self.save_path, "vecnormalize_best.pkl"))
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] failed to save vecnormalize_best: {e}")
        return True

