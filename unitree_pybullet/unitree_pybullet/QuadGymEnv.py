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

    Notes
    -----
    * Observation  : 24‑dim vector (12 joint pos + 12 joint vel)
    * Action       : 12‑dim vector (target joint pos, scaled to ±45°)
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
                 fall_angle_th: float = 0.6,    # [rad]胴体の姿勢が0.6radよりも傾くと転倒判定

                 ):
        if model not in self._SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(self._SUPPORTED_MODELS)}")
        super().__init__()
        self.model_name = model
        self.urdf_relpath = self._SUPPORTED_MODELS[model]
        self.render = render
        self.max_steps_per_episode = max_steps_per_episode
        self.fall_height_th = fall_height_th
        self.fall_angle_th = fall_angle_th
        self._ep_step = 0


        self._cid: Optional[int] = None
        self._robot: Optional[int] = None
        self.dt = 1.0 / 500.0

        # Action/obs space: assuming 12 actuated joints for all three models
        high_act = np.ones(12, dtype=np.float32)
        self.action_space = gym.spaces.Box(-high_act, high_act, dtype=np.float32)

        high_obs = np.array([np.pi] * 12 + [np.inf] * 12, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high_obs, high_obs, dtype=np.float32)

        # data/ path (shared)
        self._data_path = str(
        files("unitree_pybullet").joinpath("data").joinpath("")  # pathlib→str
        )






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
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._ep_step += 1
        action = np.clip(action, -1.0, 1.0)
        targets = np.deg2rad(45.0) * action  # scale to ±45°
        for j, tgt in enumerate(targets):
            p.setJointMotorControl2(
                self._robot,
                j,
                p.POSITION_CONTROL,
                tgt,
                force=60.0,
                physicsClientId=self._cid,
            )
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

    def _connect_if_needed(self):
        if self._cid is not None and p.isConnected(self._cid):
            return
        mode = p.GUI if self.render else p.DIRECT
        self._cid = p.connect(mode)
        p.setAdditionalSearchPath(self._data_path, physicsClientId=self._cid)

    def _get_obs(self) -> np.ndarray:
        qs = [p.getJointState(self._robot, j, physicsClientId=self._cid)[0] for j in range(12)]
        qdots = [p.getJointState(self._robot, j, physicsClientId=self._cid)[1] for j in range(12)]
        return np.array(qs + qdots, dtype=np.float32)

    def _reward(self, obs: np.ndarray, action: np.ndarray) -> float:
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
if __name__ == "__main__":
    env = QuadEnv(model="aliengo", render=True)
    obs, _ = env.reset()
    for _ in range(600):  # 1.2 s
        act = env.action_space.sample()
        obs, rew, done, *_ = env.step(act)
    env.close()

