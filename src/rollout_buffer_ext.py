# rollout_buffer_ext.py
# ------------------------------------------------------------
# 可操作度（manip_w: 4脚分）を保持して学習側へ正しく渡すための
# ReplayBuffer / RolloutBuffer の拡張。
# SB3==2.6.0 対応。PPO/SAC/TD3 すべてでミニバッチに同期。
# ------------------------------------------------------------
"""
目的: PPO（on-policy）と SAC/TD3（off-policy）両方で、manip_w を “各遷移” と完全同期してミニバッチに渡す仕組みが必要
・SB3 の標準 RolloutBuffer / ReplayBuffer は manip_w を持っていない
・適当に infos から都度再計算・再取得すると、シャッフル後のサンプル順と manip_w がズレる恐れがある（＝学習が壊れる）
・そこで、RolloutBufferWithManip / ReplayBufferWithManip を実装し、
  ・形（(T, N_env, 4) や (buffer_size, N_env, 4)）を明示
  ・SB3 2.6.0 の add() / sample() / get() のシグネチャ＆返り値に準拠
  ・返り値の namedtuple（RolloutBufferSamplesWithManip, ReplayBufferSamplesWithManip）を作って 型も順序も SB3 準拠で安全に
・これにより、アルゴリズム側（models.py）は batch.manip_w をそのまま受け取れば良い設計になり、責務分離＆安全性が担保されます。
"""

from typing import Any, Dict, Optional, Tuple, Generator
from collections import namedtuple

import numpy as np
import torch as th
from stable_baselines3.common.buffers import (
    ReplayBuffer,
    RolloutBuffer,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

# ---------- Off-policy 用（SAC/TD3） ----------
ReplayBufferSamplesWithManip = namedtuple(
    "ReplayBufferSamplesWithManip",
    ReplayBufferSamples._fields + ("manip_w",),
)


class ReplayBufferWithManip(ReplayBuffer):
    """
    SAC/TD3 用。各遷移に 4 脚分の可操作度ベクトル (manip_w) を保存して、
    sample() 時に ReplayBufferSamplesWithManip で返す。
    ※ optimize_memory_usage=True には未対応
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: th.device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        n_legs: int = 4,
    ):
        assert (
            not optimize_memory_usage
        ), "ReplayBufferWithManip は optimize_memory_usage=False を前提にしています。"
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.n_legs = n_legs
        self.manip_w = th.zeros(
            (self.buffer_size, self.n_envs, self.n_legs),
            dtype=th.float32,
            device=self.device,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Tuple[Dict[str, Any], ...],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        pos = (self.pos - 1) % self.buffer_size
        for env_idx, info in enumerate(infos):
            mw = info.get("manip_w", None)
            if mw is not None:
                self.manip_w[pos, env_idx] = th.as_tensor(
                    mw, dtype=th.float32, device=self.device
                )

    def sample(
        self, batch_size: int, env: Optional[th.Tensor] = None
    ) -> ReplayBufferSamplesWithManip:
        batch_indices = np.random.randint(0, self.size(), size=batch_size)
        if env is None:
            env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        else:
            env_indices = env.cpu().numpy()

        obs = th.as_tensor(
            self.observations[batch_indices, env_indices, :], device=self.device
        )
        next_obs = th.as_tensor(
            self.next_observations[batch_indices, env_indices, :], device=self.device
        )
        actions = th.as_tensor(
            self.actions[batch_indices, env_indices, :], device=self.device
        )
        rewards = th.as_tensor(
            self.rewards[batch_indices, env_indices], device=self.device
        ).unsqueeze(1)
        dones = th.as_tensor(
            self.dones[batch_indices, env_indices], device=self.device
        ).unsqueeze(1)
        manip_w = self.manip_w[batch_indices, env_indices, :]

        samples = ReplayBufferSamplesWithManip(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=rewards,
            manip_w=manip_w,
        )
        return samples


# ---------- On-policy 用（PPO） ----------
RolloutBufferSamplesWithManip = namedtuple(
    "RolloutBufferSamplesWithManip",
    RolloutBufferSamples._fields + ("manip_w",),
)


class RolloutBufferWithManip(RolloutBuffer):
    """
    PPO 用。各ステップで 4 脚分の可操作度ベクトル (manip_w) を保存し、
    get() でミニバッチに同期した形で返す。

    collect_rollouts 側で buffer.add(..., infos=infos) と渡す設計。
    """

    def __init__(self, *args, n_legs: int = 4, **kwargs):
        self.n_legs = n_legs
        super().__init__(*args, **kwargs)
        self.manip_w = th.zeros(
            (self.buffer_size, self.n_envs, self.n_legs),
            dtype=th.float32,
            device=self.device,
        )

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "manip_w"):
            self.manip_w.zero_()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        *_,
        infos: Optional[Tuple[Dict[str, Any], ...]] = None,
    ) -> None:
        super().add(obs, action, reward, episode_start, value, log_prob)
        if infos is None:
            return
        pos = (self.pos - 1) % self.buffer_size
        for env_idx, info in enumerate(infos):
            mw = info.get("manip_w", None)
            if mw is not None:
                self.manip_w[pos, env_idx] = th.as_tensor(
                    mw, dtype=th.float32, device=self.device
                )

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamplesWithManip, None, None]:
        assert self.full, "Rollout buffer must be full before sampling."
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        n_steps, n_envs = self.buffer_size, self.n_envs
        obs = self.observations.reshape(-1, *self.observations.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        values = self.values.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        manip_w_flat = self.manip_w.reshape(-1, self.n_legs)

        indices = np.random.permutation(n_steps * n_envs)

        start_idx = 0
        while start_idx < len(indices):
            end_idx = start_idx + batch_size
            batch_inds = indices[start_idx:end_idx]

            data = RolloutBufferSamplesWithManip(
                observations=th.as_tensor(obs[batch_inds], device=self.device),
                actions=th.as_tensor(actions[batch_inds], device=self.device),
                old_values=th.as_tensor(values[batch_inds], device=self.device),
                old_log_prob=th.as_tensor(log_probs[batch_inds], device=self.device),
                advantages=th.as_tensor(advantages[batch_inds], device=self.device),
                returns=th.as_tensor(returns[batch_inds], device=self.device),
                manip_w=th.as_tensor(manip_w_flat[batch_inds], device=self.device),
            )
            yield data
            start_idx = end_idx
