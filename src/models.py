#########################################################
# SB3をベースに任意で可操作度をロスに追加できるようにしている
# 可操作度(manip_w)を記録できるようにバッファーを拡張
# 同じくcollect_rollout()に差し替え
# ExtendModel()に統合
#########################################################
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union

import importlib
import numpy as np
import torch as th
from torch import nn
import gymnasium as gym

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, polyak_update 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from .rollout_buffer_ext import (
    RolloutBufferWithManip,
    RolloutBufferSamplesWithManip,  
    ReplayBufferWithManip
)

# =========================================================
# 共通 Mixin
# =========================================================
class ManipulabilityLossMixin:
    def __init__(
        self,
        *args,
        use_manip_loss: bool = False,
        manip_coef: float = 0.0,
        manip_agg: str = "mean",
        custom_agg_py: Optional[str] = None,
        **kwargs
    ):
        self.use_manip_loss = use_manip_loss
        self.manip_coef = manip_coef
        self.manip_agg = manip_agg
        self.custom_agg_py = custom_agg_py
        self._custom_agg_fn = self._load_custom_agg(custom_agg_py)
        super().__init__(*args, **kwargs)

    # ---------- aggregation helpers ----------
    def _load_custom_agg(self, custom_agg_py: Optional[str]):
        if custom_agg_py is None:
            return None
        if ":" not in custom_agg_py:
            raise ValueError("custom_agg_py は 'module.path:func_name' 形式で指定してください")
        mod_path, fn_name = custom_agg_py.split(":")
        mod = importlib.import_module(mod_path)
        return getattr(mod, fn_name)

    def _agg_per_sample(self, w: th.Tensor) -> th.Tensor:
        """
        w: (batch, 4) を想定。
        戻り値: (batch,) の 1次元テンソル（サンプル単位に 4脚を集約した値）
        """
        if w.ndim == 1:
            w = w.unsqueeze(0)  # (1, 4)
        if self.manip_agg == "mean":
            return w.mean(dim=-1)
        elif self.manip_agg == "min":
            return w.min(dim=-1).values
        elif self.manip_agg == "max":
            return w.max(dim=-1).values
        elif self.manip_agg == "sum":
            return w.sum(dim=-1)
        elif self.manip_agg == "custom":
            if self._custom_agg_fn is None:
                raise ValueError("manip_agg='custom' ですが custom_agg_py が設定されていません")
            out = self._custom_agg_fn(w)  # 期待: (batch,)
            if not isinstance(out, th.Tensor):
                raise TypeError("custom_agg_fn は torch.Tensor を返す必要があります")
            if out.ndim != 1 or out.shape[0] != w.shape[0]:
                raise ValueError("custom_agg_fn は shape=[batch] の 1次元 Tensor を返す必要があります")
            return out.to(w.device)
        else:
            raise ValueError(f"Unknown manip_agg: {self.manip_agg}")

    def _manip_loss_from_batch(self, manip_w: Optional[th.Tensor]) -> th.Tensor:
        if (not self.use_manip_loss) or (manip_w is None):
            return th.zeros((), device=self.device)

        # manip_w: (batch, 4) を想定
        per_sample = self._agg_per_sample(manip_w)  # (batch,)
        scalar = per_sample.mean()  # バッチ平均

        # 可操作度（大きくしたい）→ -coef * scalar を policy_loss に足す
        loss_term = - self.manip_coef * scalar

        if hasattr(self, "logger"):
            self.logger.record("train/manip_scalar", float(scalar.item()))
            self.logger.record("train/manip_agg", self.manip_agg)

        return loss_term


# =========================================================
# PPO（On-policy）
# =========================================================
class PPOWithManip(ManipulabilityLossMixin, PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # ★ RolloutBuffer を manip 対応版に差し替え
        self.rollout_buffer = RolloutBufferWithManip(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            n_legs=4,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # 1) いまの観測を退避（add にはコレを渡す）
            obs = self._last_obs
            episode_starts = self._last_episode_starts

            with th.no_grad():
                obs_tensor = th.as_tensor(obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            rewards = rewards.reshape(-1, 1)
            dones = dones.reshape(-1, 1)

            # 2) ★ ここで obs/episode_starts を用いて、infos を渡して add
            rollout_buffer.add(obs, actions, rewards, episode_starts, values, log_probs, infos)

            self._last_obs = new_obs
            self._last_episode_starts = dones
            n_steps += 1

        with th.no_grad():
            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            _, values, _ = self.policy(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining) if callable(self.clip_range) else self.clip_range
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if (self.clip_range_vf is not None and callable(self.clip_range_vf))
            else self.clip_range_vf
        )

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss
                entropy_loss = -th.mean(entropy)

                # 可操作度ロス
                if self.use_manip_loss:
                    m_loss = self._manip_loss_from_batch(rollout_data.manip_w)
                    policy_loss = policy_loss + m_loss
                    self.logger.record("train/manip_loss", float(m_loss.item()))
                    self.logger.record("train/manip_metric_only", float(rollout_data.manip_w.mean().item()))

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/n_updates", int(self._n_updates), exclude="tensorboard")
        self.logger.record("train/policy_gradient_loss", float(th.tensor(pg_losses).mean().item()))
        self.logger.record("train/value_loss", float(th.tensor(value_losses).mean().item()))
        self.logger.record("train/entropy_loss", float(th.tensor(entropy_losses).mean().item()))
        self.logger.record("train/clip_fraction", float(th.tensor(clip_fractions).mean().item()))
        self.logger.record("train/explained_variance", float(explained_var))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", float(th.exp(self.policy.log_std).mean().item()))

        self.logger.record("train/clip_range", float(clip_range))
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", float(clip_range_vf))


# =========================================================
# SAC（Off-policy）
# =========================================================
class SACWithManip(ManipulabilityLossMixin, SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # ReplayBuffer を manip 対応に差し替え
        self.replay_buffer = ReplayBufferWithManip(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=getattr(self, "optimize_memory_usage", False),
            handle_timeout_termination=getattr(self, "handle_timeout_termination", True),
            n_legs=4,
        )

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            self._n_updates += 1

            replay_data = self.replay_buffer.sample(batch_size)

            # ★ ent_coef を毎ステップローカルで決定
            if self.ent_coef_optimizer is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = th.as_tensor(self.ent_coef, device=self.device, dtype=th.float32)

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum([th.nn.functional.mse_loss(cur_q, target_q_values) for cur_q in current_q_values])
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            if self.use_manip_loss and hasattr(replay_data, "manip_w"):
                m_loss = self._manip_loss_from_batch(replay_data.manip_w)
                actor_loss = actor_loss + m_loss
                self.logger.record("train/manip_loss", float(m_loss.item()))
                self.logger.record("train/manip_metric_only", float(replay_data.manip_w.mean().item()))

            actor_losses.append(float(actor_loss.item()))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_losses.append(float(ent_coef_loss.item()))
                ent_coefs.append(float(ent_coef.item()))

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self.logger.record("train/n_updates", int(self._n_updates), exclude="tensorboard")
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))
            self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))


# =========================================================
# TD3（Off-policy）
# =========================================================
class TD3WithManip(ManipulabilityLossMixin, TD3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        #  ReplayBuffer を manip 対応に差し替え
        self.replay_buffer = ReplayBufferWithManip(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=getattr(self, "optimize_memory_usage", False),  # TD3 default
            handle_timeout_termination=getattr(self, "handle_timeout_termination", True),
            n_legs=4,
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # ほぼ SB3 の TD3.train() を踏襲し、actor_loss に manip_loss を追加
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            self._n_updates += 1

            replay_data = self.replay_buffer.sample(batch_size)

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                target_q1, target_q2 = self.critic_target(replay_data.next_observations, next_actions)
                target_q = th.min(target_q1, target_q2)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = th.nn.functional.mse_loss(current_q1, target_q) + th.nn.functional.mse_loss(current_q2, target_q)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(
                    replay_data.observations, self.actor(replay_data.observations)
                ).mean()

                #  可操作度ロス
                if self.use_manip_loss and hasattr(replay_data, "manip_w"):
                    m_loss = self._manip_loss_from_batch(replay_data.manip_w)
                    actor_loss = actor_loss + m_loss
                    self.logger.record("train/manip_loss", float(m_loss.item()))
                    self.logger.record("train/manip_metric_only", float(replay_data.manip_w.mean().item()))

                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self.logger.record("train/n_updates", int(self._n_updates), exclude="tensorboard")
        self.logger.record("train/actor_loss", float(th.tensor(actor_losses).mean().item()))
        self.logger.record("train/critic_loss", float(th.tensor(critic_losses).mean().item()))


# =========================================================
# Factory / Wrapper
# =========================================================
class ExtendModel:
    """
    既存 main.py / test_trained_model.py からの呼び出しを保つためのラッパ
    create(...) でインスタンスを返し、load(...) で読み込む
    """
    @classmethod
    def create(
        cls,
        model_name: str,
        policy: Union[str, nn.Module],
        env,
        use_manip_loss: bool,
        manip_coef: float,
        seed: int,
        device: str,
        batch_size: int,
        policy_kwargs: dict,
        manip_agg: str = "mean",
        custom_agg_py: Optional[str] = None,
        **kwargs,
    ):
        algo = model_name.upper()
        common = dict(
            verbose=1,
            seed=seed,
            device=device,
            batch_size=batch_size,
            use_manip_loss=use_manip_loss,
            manip_coef=manip_coef,
            manip_agg=manip_agg,
            custom_agg_py=custom_agg_py,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )
        if algo == "PPO":
            return PPOWithManip(policy, env, **common)
        elif algo == "SAC":
            return SACWithManip(policy, env, **common)
        elif algo == "TD3":
            return TD3WithManip(policy, env, **common)
        else:
            raise ValueError(f"Unsupported algorithm: {model_name}")

    @staticmethod
    def load(path: str, env=None, device: str = "auto", model_name: str = "PPO", **kwargs):
        algo = model_name.upper()
        if algo == "PPO":
            return PPOWithManip.load(path, env=env, device=device, **kwargs)
        elif algo == "SAC":
            return SACWithManip.load(path, env=env, device=device, **kwargs)
        elif algo == "TD3":
            return TD3WithManip.load(path, env=env, device=device, **kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {model_name}")
