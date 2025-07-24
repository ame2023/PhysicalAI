# rollout_buffer_ext.py
# ------------------------------------------------------------
# 可操作度（manip_w: 4脚分）を保持して学習側へ正しく渡すための
# ReplayBuffer / RolloutBuffer の拡張。
# SB3==2.6.0 対応。PPO/SAC/TD3 すべてでミニバッチに同期。
# ------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple
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
# ReplayBufferSamples を拡張した namedtuple を用意
ReplayBufferSamplesWithManip = namedtuple(
    "ReplayBufferSamplesWithManip",
    ReplayBufferSamples._fields + ("manip_w",)
)


class ReplayBufferWithManip(ReplayBuffer):
    """
    SAC/TD3 用。各遷移に 4 脚分の可操作度ベクトル (manip_w) を保存して、
    sample() 時に ReplayBufferSamplesWithManip で返す。
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
        if env is None:
            env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        else:
            env_indices = env.cpu().numpy()
        batch_indices = np.random.randint(0, self.size(), size=batch_size)

        obs = th.as_tensor(
            self.observations[batch_indices, env_indices, :], device=self.device
        )
        next_obs = th.as_tensor(
            self.next_observations[batch_indices, env_indices, :], device=self.device
        )
        actions = th.as_tensor(
            self.actions[batch_indices, env_indices, :], device=self.device
        )
        rewards = (
            th.as_tensor(self.rewards[batch_indices, env_indices], device=self.device)
            .unsqueeze(1)
            .clone()
        )
        dones = (
            th.as_tensor(self.dones[batch_indices, env_indices], device=self.device)
            .unsqueeze(1)
            .clone()
        )
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
    """

    def __init__(self, *args, n_legs: int = 4, **kwargs):
        self.n_legs = n_legs
        super().__init__(*args, **kwargs)
        self.manip_w = th.zeros(
            (self.buffer_size, self.n_envs, self.n_legs),
            dtype=th.float32, device=self.device
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
        infos: Tuple[Dict[str, Any], ...],
    ) -> None:
        super().add(obs, action, reward, episode_start, value, log_prob)
        pos = (self.pos - 1) % self.buffer_size
        for env_idx, info in enumerate(infos):
            mw = info.get("manip_w", None)
            if mw is not None:
                self.manip_w[pos, env_idx] = th.as_tensor(
                    mw, dtype=th.float32, device=self.device
                )

    def get(self, batch_size: Optional[int] = None):
        assert self.full, "Rollout buffer must be full before sampling."
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        obs = self.observations.reshape(-1, *self.observations.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        values = self.values.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        manip_w_flat = self.manip_w.reshape(-1, self.n_legs)

        start_idx = 0
        while start_idx < len(indices):
            end_idx = start_idx + batch_size
            batch_inds = indices[start_idx:end_idx]
            data = RolloutBufferSamplesWithManip(
                observations=th.as_tensor(obs[batch_inds]).to(self.device),
                actions=th.as_tensor(actions[batch_inds]).to(self.device),
                old_values=th.as_tensor(values[batch_inds]).to(self.device),
                old_log_prob=th.as_tensor(log_probs[batch_inds]).to(self.device),
                advantages=th.as_tensor(advantages[batch_inds]).to(self.device),
                returns=th.as_tensor(returns[batch_inds]).to(self.device),
                manip_w=th.as_tensor(manip_w_flat[batch_inds]).to(self.device),
            )
            yield data
            start_idx = end_idx
