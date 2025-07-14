import os
from typing import Tuple, Dict, Optional

import numpy as np

try:
    import gymnasium as gym  # Gym >=0.26 (Gymnasium fork)
except ImportError:  # fallback to classic gym
    import gym
    gymnasium_available = False
else:
    gymnasium_available = True

import pybullet as p
from importlib.resources import files

###########################################################################
# QuadEnv: Gym/Gymnasium‑compatible PyBullet environment                   #
# ----------------------------------------------------------------------- #
# * Supports Unitree **A1**, **Laikago**, **Aliengo** models shipped with  #
#   `unitree_pybullet` (data/<model>/urdf/<model>.urdf).                  #
# * No modification of the original demo scripts is required.            #
# * Usage:                                                                #
#       env = QuadEnv(model="a1", render=False)                         #
#       obs, _ = env.reset()                                              #
#       ...                                                               #
###########################################################################

__all__ = ["QuadEnv"]


class QuadEnv(gym.Env):
    """Minimal RL environment for Unitree quadrupeds (A1, Laikago, Aliengo).

    Parameters
    ----------
    model : str, default "a1"
        Which robot to load. One of {"a1", "laikago", "aliengo"}.
    render : bool, default False
        If True, connect in GUI mode; otherwise DIRECT.

    obs_mode : {"joint", "joint+base", "full"}, default "joint"
        "full" は base 状態 + 直前トルク τ_{t-1} も含む（49 次元）
    
    control_mode : {"position", "torque"}, default "position"
        "torque" を選ぶと action は正規化トルク指令（±torque_scale_Nm）
    action_scale_deg : float, default 45
        "position" 制御時に action ∈ [-1,1] を ±action_scale_deg [deg] へ線形変換
    torque_scale_Nm : float, default 60
        "torque" 制御時の最大絶対トルク [Nm]

    Notes
    -----
    * Observation  :    dim vector 
    * Action       : 12‑dim vector (target joint pos or torque)
    * Reward (demo): −‖q‖²   ← replace `_reward()` for real tasks
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    _SUPPORTED_MODELS = {
        "a1": "a1/urdf/a1.urdf",
        "laikago": "laikago/urdf/laikago.urdf",
        "aliengo": "aliengo/urdf/aliengo.urdf",
    }

    def __init__(self, model: str = "a1", 
                 render: bool = False, 
                 max_steps_per_episode: int = 1000, # １エピソードあたりの最大ステップ数
                 fall_height_th: float = 0.25,  # [cm] 胴体の高さが25cm以下で転倒判定
                 fall_angle_th: float = 0.6,    # [rad] 胴体の姿勢が0.6radよりも傾くと転倒判定
                 obs_mode: str = "full",         # 観測データの種類を指定
                 action_scale_deg: float = 45.0, # [deg] アクションのスケールを指定
                 control_mode: str = "position", # 制御方法を指定 position or torque
                 torque_scale_Nm: float = 60.0,  # [Nm] トルクのスケールを指定
                 reward_mode: str = 'progress',
                 ):
        
        if model not in self._SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(self._SUPPORTED_MODELS)}")
        if obs_mode not in {"joint", "joint+base", "full"}:
            raise ValueError(f"Unknown obs_mode '{obs_mode}'")
        if control_mode not in {"position", "torque"}:
            raise ValueError(f"Unknown control_mode '{control_mode}'")

        super().__init__()
        self.model_name = model
        self.urdf_relpath = self._SUPPORTED_MODELS[model]
        self.render = render
        self.num_joint = 12

        # 終了条件の設定
        self.max_steps_per_episode = max_steps_per_episode
        self.fall_height_th = fall_height_th
        self.fall_angle_th = fall_angle_th
        self._ep_step = 0
        
        # 観測データの指定
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        # 行動出力の指定
        self._action_scale_rad = np.deg2rad(action_scale_deg)
        self.torque_scale = torque_scale_Nm
        # 報酬関数の指定
        self.reward_mode = reward_mode 

        self._cid: Optional[int] = None
        self._robot: Optional[int] = None
        self.dt = 1.0 / 500.0
        self._prev_tau = np.zeros(self.num_joint, dtype=np.float32)
        
        # data/ path (shared)
        self._data_path = str(
        files("unitree_pybullet").joinpath("data").joinpath("")  # pathlib→str
        )
        # Gym spaces
        self._build_spaces()






    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._connect_if_needed()
        p.resetSimulation(physicsClientId=self._cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self._cid)
        p.setTimeStep(self.dt, physicsClientId=self._cid)
                
        # data/　の絶対パス
        from unitree_pybullet.unitree_pybullet import get_data_path
        data_dir = get_data_path()

        # ② URDF を絶対パスで読み込む
        plane_path = os.path.join(data_dir, "plane.urdf")
        robot_path = os.path.join(data_dir, self.urdf_relpath)
        
        p.loadURDF(plane_path, physicsClientId=self._cid)
        
        self._robot = p.loadURDF(
            robot_path,
            [0, 0, 0.48],
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self._cid,
        )
        self._prev_tau.fill(0.0)   # ← 直前トルクの初期化
        self._ep_step = 0          # ← エピソードステップをリセット
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._ep_step += 1 

        if self.control_mode == "position":
            targets = self._action_scale_rad * action
            for j, tgt in enumerate(targets):
                p.setJointMotorControl2(
                    self._robot,
                    j,
                    p.POSITION_CONTROL,
                    targetPosition=float(tgt),
                    force=self.torque_scale,
                    physicsClientId=self._cid,
                )
            # 実際に発生したトルクを取得（次ステップ用）
            self._prev_tau = np.array(
                [p.getJointState(self._robot, j, physicsClientId=self._cid)[3] for j in range(self.num_joint)],
                dtype=np.float32,
            )

        elif self.control_mode == "torque":
            taus = self.torque_scale * action
            for j, tau in enumerate(taus):
                p.setJointMotorControl2(
                    self._robot,
                    j,
                    p.TORQUE_CONTROL,
                    force=float(tau),
                    physicsClientId=self._cid,
                )
            self._prev_tau = taus.astype(np.float32)

        p.stepSimulation(physicsClientId=self._cid)

        obs = self._get_obs()
        reward = self._reward(obs, action)

        # エピソード修了条件 #
        terminated = self._robot_fallen()
        truncated = self._ep_step >= self.max_steps_per_episode

        if terminated or truncated:
            self._ep_step = 0


        info = {}
        if gymnasium_available:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, info

    def close(self):
        if self._cid is not None:
            p.disconnect(self._cid)
            self._cid = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_spaces(self):
        # 行動空間
        high_act = np.ones(self.num_joint, dtype=np.float32)
        self.action_space = gym.spaces.Box(-high_act, high_act, dtype=np.float32)

        # 観測空間
        if self.obs_mode == "joint": # 関節角度(12)＋関節角速度(12) = 24次元
            high_obs = np.array([np.pi] * self.num_joint + [np.inf] * self.num_joint, dtype=np.float32)
        elif self.obs_mode == "joint+base": # 関節角度(12)＋関節角速度(12)＋ベース位置(3)＋姿勢：クォータニオン(4)＋移動速度(3)+姿勢の角速度(3)　＝　37次元
            high_obs = np.array([np.pi] * self.num_joint + [np.inf] * 25, dtype=np.float32)
        elif self.obs_mode == "full": # 関節角度(12)＋関節角速度(12)＋ベース位置(3)＋姿勢：クォータニオン(4)＋移動速度(3)+姿勢の角速度(3) + １ステップ前のトルク(12)　＝　49次元
            high_obs = np.array([np.pi] * self.num_joint + [np.inf] * 37, dtype=np.float32)  # 49 dims
        self.observation_space = gym.spaces.Box(-high_obs, high_obs, dtype=np.float32)

    def _connect_if_needed(self):
        if self._cid is not None and p.isConnected(self._cid):
            return
        mode = p.GUI if self.render else p.DIRECT
        self._cid = p.connect(mode)
        p.setAdditionalSearchPath(self._data_path, physicsClientId=self._cid)

    def _get_obs(self) -> np.ndarray:
        qs = [p.getJointState(self._robot, j, physicsClientId=self._cid)[0] for j in range(self.num_joint)]
        qdots = [p.getJointState(self._robot, j, physicsClientId=self._cid)[1] for j in range(self.num_joint)]
        obs = qs + qdots

        if self.obs_mode in {"joint+base", "full"}:
            pos, orn = p.getBasePositionAndOrientation(self._robot, self._cid)
            lin, ang = p.getBaseVelocity(self._robot, self._cid)
            obs += list(pos) + list(orn) + list(lin) + list(ang)

        if self.obs_mode == "full":
            obs += list(self._prev_tau)

        return np.array(obs, dtype=np.float32)

    def _reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        if self.reward_mode == "progress" and self.obs_mode in {"joint+base", "full"}:
            vx, vy = obs[31], obs[32]
            target_vx, target_vy = 1.0, 0.0
            sigma_v = 0.25
            v_err_sq = (target_vx - vx) ** 2 + (target_vy - vy) ** 2
            return float(np.exp(-v_err_sq / (2 * sigma_v ** 2)))

        elif self.reward_mode == "EnergeticProgress":
            """
            報酬関数の設計
            """
            return 0.0  # TODO: 実装
        

        # fallback: joint-angle penalty
        q = obs[:12]
        return -float(np.sum(q ** 2))



    def _robot_fallen(self) -> bool:
        """胴体の高さと姿勢から転倒を判定する

        - 胴体(ベースリンク)の z 座標 < height_th → 転倒
        - roll, pitch の絶対値 > angle_th → 転倒
        - しきい値はコンストラクタ引数で変更可
        """
        height_th = getattr(self, "fall_height_th", 0.25)   # 25 cm
        angle_th  = getattr(self, "fall_angle_th", 0.6)     # 約34°

        pos, orn = p.getBasePositionAndOrientation(self._robot,
                                                   physicsClientId=self._cid)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        fall_height = pos[2] < height_th
        fall_angle  = abs(roll) > angle_th or abs(pitch) > angle_th
        return fall_height or fall_angle


# Quick test -----------------------------------------------------------------
# if __name__ == "__main__":
#     env = QuadEnv(
#         model="a1",
#         render=False,
#         obs_mode="full",
#         control_mode="torque",
#         torque_scale_Nm=40,
#     )
#     obs, _ = env.reset()
#     print("obs dim =", obs.shape)
#     for _ in range(1000):
#         a = env.action_space.sample()
#         env.step(a)
#     env.close()

