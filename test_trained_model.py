import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

import hydra
from omegaconf import DictConfig
import wandb

from src.utils import set_seed
# from src.models import PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv # 並列数(>16)＋並列環境の可視化をしない
from gymnasium.vector import AsyncVectorEnv # 並列数(<=8)＋並列環境の可視化したい
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor

from unitree_pybullet import QuadEnv


@hydra.main(version_base = None, 
            config_path = "configs",# config.yamlファイルのパス
            config_name = "config"  # config.yamlファイルの名前
            )


def main(cfg:DictConfig):
    set_seed(cfg.test_seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # --- wandb ------------------------------------------------------
    run = None
    if cfg.use_wandb:
        run = wandb.init(mode="online",
                         dir=logdir,
                         project=f"{cfg.unitree_model}_ppo_test",
                         config=dict(cfg)
                         )

    # --- 並列環境の作成 ------------------------------------------------------
    #def make_env(rank:int):
    #    def _init():
    #        env = QuadEnv(model=cfg.unitree_model, render=False)
    #        env.reset(seed = cfg.seed + rank)
    #        env.action_space.seed(cfg.seed + rank)
    #        return env
    #    return _init

    def make_env_subproc(rank: int):
        def _init():
            env = QuadEnv(model=cfg.unitree_model, 
                          render=False,
                      max_episode_steps=cfg.max_episode_steps,
                      fall_height_th = cfg.fall_height_th,  
                      fall_angle_th = cfg.fall_angle_th,
                      )
            env.reset(seed=cfg.test_seed+rank)
            env.action_space.seed(cfg.test_seed+rank)
            return env
        return _init
    
    if cfg.multi_env:
        env = SubprocVecEnv([make_env_subproc(i) for i in range(cfg.num_envs)], start_method = "spawn") 
        #env = AsyncVectorEnv([make_env(i) for i in range(cfg.num_envs)])
        env = VecMonitor(env) # 報酬などを計測

    else:
        env = QuadEnv(model=cfg.unitree_model, render=False)
        env.reset(seed=cfg.test_seed)
        env = Monitor(env, 
                      # 動画保存に関する設定
                      # video_callable = lambda episode_id: True, #ビデオを保存するかどうか
                      # video_folder = logdir+'/video', #ビデオを保存するフォルダの指定
                      # video_format = 'mp4', #ビデオの形式
                      ) # 報酬などを計測

    # --- 学習済みAgentの読み込み ------------------------------------------------------
    agent = PPO(cfg.policy, env, verbose=0, device = "auto")
    agent.load(os.path.join(logdir, f"ppo_{cfg.unitree_model}.zip"))


    # --- テスト ------------------------------------------------------
    obs, info = env.reset()
    for i in range(1000):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

    if cfg.use_wandb:
        wandb.log({
            "total_steps": cfg.total_steps,
            "unitree_model": cfg.unitree_model,
            "policy": cfg.policy, 
            "seed": cfg.test_seed,
            # "max_reward": max_reward,                     
            # "mean_reward": mean_reward,
                               
        })

    if run is not None: 
        run.finish()
    env.close()

    

if __name__ == '__main__':
    env = QuadEnv(model="a1", render=False)
    print(env.action_space)
    print(env.observation_space)
  