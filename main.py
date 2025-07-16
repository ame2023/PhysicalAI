import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

import hydra
from omegaconf import DictConfig
import wandb
from tqdm import tqdm

from src.utils import set_seed
# from src.models import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv # 非同期処理，並列数(>16)＋並列環境の可視化をしない
from stable_baselines3.common.vec_env import DummyVecEnv # 同期処理
from gymnasium.vector import AsyncVectorEnv # 並列数(<=8)＋並列環境の可視化したい
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from unitree_pybullet.unitree_pybullet import QuadEnv

from src.models import ExtendModel


@hydra.main(version_base = None, 
            config_path = "configs",# config.yamlファイルのパス
            config_name = "config"  # config.yamlファイルの名前
            )


def main(cfg:DictConfig):
    set_seed(cfg.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    monitor_path = os.path.join(logdir, "monitor.csv")
    # --- wandb ------------------------------------------------------
    run = None
    if cfg.use_wandb:
        run = wandb.init(mode="online",
                         dir=logdir,
                         project=f"{cfg.unitree_model}_ppo_test",
                         config=dict(cfg)
                         )

    # --- 並列環境の作成 ------------------------------------------------------
    # def make_env(rank:int):
    #     def _init():
    #         env = QuadEnv(model=cfg.unitree_model, 
    #                       render=False,
    #                       max_steps_per_episode=cfg.max_steps_per_episode,
    #                       fall_height_th = cfg.fall_height_th,  
    #                       fall_angle_th = cfg.fall_angle_th,
    #                       )
    #         env.reset(seed = cfg.seed + rank)
    #         env.action_space.seed(cfg.seed + rank)
    #         return env
    #     return _init

    def make_env_subproc(rank: int):
        def _init():
            env = QuadEnv(model=cfg.unitree_model, 
                      render=False,
                      max_steps_per_episode=cfg.max_steps_per_episode,
                      fall_height_th = cfg.fall_height_th,  
                      fall_angle_th = cfg.fall_angle_th,
                      )
            env.reset(seed=cfg.seed+rank)
            env.action_space.seed(cfg.seed+rank)
            return env
        return _init
    
    if cfg.multi_env:
        env = SubprocVecEnv([make_env_subproc(i) for i in range(cfg.num_envs)], start_method = "spawn") 
        #env = AsyncVectorEnv([make_env(i) for i in range(cfg.num_envs)])
        env = VecMonitor(env,
                         filename = monitor_path, # 報酬などの保存先
                         #info_keywords=("episode",)
                         ) # 報酬などを計測

    else:
        env = QuadEnv(model=cfg.unitree_model, 
                      render=cfg.render,
                      max_steps_per_episode=cfg.max_steps_per_episode,
                      fall_height_th = cfg.fall_height_th,  
                      fall_angle_th = cfg.fall_angle_th,
                      obs_mode = cfg.obs_mode,        # 観測データの種類を指定
                      action_scale_deg = cfg.action_scale_deg, # [deg] アクションのスケールを指定
                      control_mode = cfg.control_mode, # 制御方法を指定 position or torque
                      torque_scale_Nm = cfg.torque_scale_Nm,  # [Nm] トルクのスケールを指定
                      reward_mode = cfg.reward_mode,
                      )
        env.reset(seed=cfg.seed)
        env = Monitor(env,
                      filename = monitor_path, # 報酬などの保存先
                      #info_keywords=("episode",)
                      ) # 報酬などを計測

    # --- Agent作成 ------------------------------------------------------
    agent = PPO(cfg.policy, env, 
                verbose=1, # 学習ログの詳細表示，0で表示しない
                device = "auto"
                 )

    # --- 学習 ------------------------------------------------------
    for _ in tqdm(range(cfg.total_steps // 2048)):
        agent.learn(2048, reset_num_timesteps=False)

    # --- モデル保存 ------------------------------------------------------
    agent.save(os.path.join(logdir, f"ppo_{cfg.unitree_model}.zip"))



    if cfg.use_wandb:
        wandb.log({
            "total_steps": cfg.total_steps,
            "unitree_model": cfg.unitree_model,
            "policy": cfg.policy, 
            "seed": cfg.seed,
            # "max_reward": max_reward,                     
            # "mean_reward": mean_reward,
                               
        })

    if run is not None: 
        run.finish()
    env.close()

    

if __name__ == '__main__':
    main()
    # env = QuadEnv(model="a1", render=False)
    # print(env.action_space)
    # print(env.observation_space)
  