# src/callbacks.py
import os
import numpy as np
import pandas as pd
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
        kv = self.model.logger.name_to_value
        if any(k.startswith("train/") for k in kv):
            self.records.append({k: kv[k] for k in kv if k.startswith("train/")})
        return True

    def _on_training_end(self) -> None:
        if not self.records:
            return
        out = os.path.join(self.logdir, "loss_history.csv")
        pd.DataFrame(self.records).to_csv(out, index=False)
        if self.verbose > 0:
            print(f"[LossLoggerCallback] wrote {out}")


class ManipLoggerCallback(BaseCallback):
    """
    環境×脚の可操作度をそのまま 1 step ごとに CSV に保存する
    step, env, episode_id, t_in_episode, leg_0..3, mean, min, max
    """
    def __init__(self, logdir: str, use_wandb: bool = False, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.logdir = logdir
        self.use_wandb = use_wandb
        self.rows = []
        self.episode_ids = None
        self.t_in_episode = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.episode_ids = np.zeros(n_envs, dtype=np.int64)
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
                "episode_id": int(self.episode_ids[env_id]),
                "t_in_episode": int(self.t_in_episode[env_id]),
                "mean": float(w.mean()),
                "min": float(w.min()),
                "max": float(w.max()),
            }
            for i, v in enumerate(w):
                row[f"leg_{i}"] = float(v)
            self.rows.append(row)

        # episode境界の更新
        if dones is not None:
            for env_id, d in enumerate(dones):
                if d:
                    self.episode_ids[env_id] += 1
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
