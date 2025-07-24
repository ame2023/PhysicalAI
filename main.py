import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
import pandas as pd                 
import matplotlib.pyplot as plt

from src.utils import set_seed
# from src.models import PPO
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv # 非同期処理，並列数(>16)＋並列環境の可視化をしない
from stable_baselines3.common.vec_env import DummyVecEnv # 同期処理
from gymnasium.vector import AsyncVectorEnv # 並列数(<=8)＋並列環境の可視化したい
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback,  BaseCallback

from unitree_pybullet.unitree_pybullet import QuadEnv

from src.models import ExtendModel
from src.SkipFrame import SkipFrame
from src.make_figure import save_reward_and_loss_plots



# ---- ロスのロガー -------------------------------------------------------------
class LossLoggerCallback(BaseCallback):
    def __init__(self, logdir, verbose=0):
        super().__init__(verbose=verbose)
        self.logdir, self.records = logdir, []

    def _on_step(self) -> bool:
        # logger.dump() 直後なので train/xxx_loss が揃っている
        kv = self.model.logger.name_to_value
        if any(k.startswith("train/") for k in kv):
            self.records.append({k: kv[k] for k in kv if k.startswith("train/")})
        return True

    def _on_training_end(self):
        if self.records:
            pd.DataFrame(self.records).to_csv(
                os.path.join(self.logdir, "loss_history.csv"), index=False
            )

# ------------------------------------------- #

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
                         project=f"{cfg.unitree_model}_train",
                         config=dict(cfg)
                         )
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)# DictConfig -> dict
    # ---- 環境作成----------------------------------------------------
    def make_env_subproc(rank: int, cfg_dict: dict):
        def _init():
            env = QuadEnv(model=cfg_dict["unitree_model"], 
                      render=False,
                      max_steps_per_episode=cfg_dict["max_steps_per_episode"],
                      fall_height_th = cfg_dict["fall_height_th"],  
                      fall_angle_th = cfg_dict["fall_angle_th"],
                      obs_mode=cfg_dict["obs_mode"],
                      action_scale_deg=cfg_dict["action_scale_deg"],
                      control_mode=cfg_dict["control_mode"],
                      torque_scale_Nm=cfg_dict["torque_scale_Nm"],
                      reward_mode=cfg_dict["reward_mode"],
                      )
            #env = SkipFrame(env, skip = cfg.skip_freq)
            env.reset(seed=cfg_dict["seed"]+rank)
            env.action_space.seed(cfg_dict["seed"]+rank)
            return env
        return _init
    
    if cfg.multi_env:
        env = SubprocVecEnv([make_env_subproc(i, cfg_dict) for i in range(cfg.num_envs)], start_method = "spawn") 
        #env = AsyncVectorEnv([make_env(i) for i in range(cfg.num_envs)])
        env = VecNormalize(env,
                            norm_obs=True,
                            norm_reward=cfg.norm_reward,#報酬の正規化はPPOのみで実施
                            clip_obs=10.
                            )
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
                      control_mode = cfg.control_mode, # 制御方法を指定 PDcontrol or torque
                      torque_scale_Nm = cfg.torque_scale_Nm,  # [Nm] トルクのスケールを指定
                      reward_mode = cfg.reward_mode,
                      )
        #env = SkipFrame(env, skip = cfg.skip_freq)
        # 乱数シード固定
        env.reset(seed=cfg.seed)
        env.action_space.seed(cfg.seed)
        env.observation_space.seed(cfg.seed)

        env = DummyVecEnv([lambda:env])
        env = VecNormalize(env,
                           training = True,
                            norm_obs=True,
                            norm_reward=cfg.norm_reward, 
                            clip_obs=10.)
        env = VecMonitor(env,
                      filename = monitor_path, # 報酬などの保存先
                      #info_keywords=("episode",)
                      ) # 報酬などを計測
    
    # --- Agent作成 ------------------------------------------------------
    if cfg.algo == "PPO":
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]) # pi:policy network, vf:value network
            )
    else:
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    """
    PPO以外の学習では報酬の正規化をoffにしておく
    学習環境と評価（callback）環境の両方
    env = VecNormalize(env, norm_obs=True, 
    norm_reward=True,  ＜ー False
    clip_obs=10.)
    """
    #agent = PPO(cfg.policy, env, 
    #            n_steps = cfg.n_steps,
    #            verbose=1, # 学習ログの詳細表示，0で表示しない
    #            seed = cfg.seed,
    #            device = cfg.device,
    #            batch_size = cfg.minibatch_size,
    #            policy_kwargs = policy_kwargs
    #             )
    agent = ExtendModel.create(
        model_name=cfg.algo,          # 'PPO' / 'SAC' / 'TD3'
        policy = cfg.policy,
        env=env,
        use_manip_loss=cfg.use_manip_loss,
        manip_coef=cfg.manip_coef,
        seed=cfg.seed,
        device=cfg.device,
        batch_size=cfg.minibatch_size,
        policy_kwargs=policy_kwargs,
        )

    # --- 評価環境とコールバックの設定 --------------------------------
    eval_env = SubprocVecEnv([make_env_subproc(i+100, cfg_dict) for i in range(cfg.num_eval_envs)], start_method = "spawn") 
    eval_env = VecNormalize(eval_env, training = False, norm_obs=True, norm_reward=cfg.norm_reward, clip_obs=10.)
    eval_env = VecMonitor(eval_env, filename=os.path.join(logdir, "eval_monitor.csv"))
    eval_env.reset()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=logdir,
        log_path=logdir,
        eval_freq=2048*5,        # 2048*5ステップごとに評価（エージェントの更新が2048stepに1回のため）
        n_eval_episodes=10,     # 10エピソードで平均をとる
        deterministic=False,
        render=False,
        )

    # --- 学習 ------------------------------------------------------
    loss_logger = LossLoggerCallback(logdir)
    agent.learn(total_timesteps = cfg.total_steps,
                 reset_num_timesteps=False,
                 callback = [eval_callback, loss_logger],
                 progress_bar = True,
                 )


    # --- 報酬推移プロット --------------------------------------------------
    # --- 図の作成・保存 --------------------------------------------------
    save_reward_and_loss_plots(
        logdir=logdir,
        monitor_csv=monitor_path,
        loss_csv=os.path.join(logdir, "loss_history.csv"),
    )



    # --- モデル保存 ------------------------------------------------------
    agent.save(os.path.join(logdir, f"{cfg.algo.lower()}_{cfg.unitree_model}.zip"))
    # 統計情報の保存（正規化用） #
    env.save(os.path.join(logdir, "vecnormalize.pkl"))


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
    eval_env.close()

    

if __name__ == '__main__':
    main()
    # env = QuadEnv(model="a1", render=False)
    # print(env.action_space)
    # print(env.observation_space)
  