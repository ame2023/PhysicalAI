import os
from typing import Tuple, Dict, Optional, List
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
from . import manip_utils as mu

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
    
    control_mode : {"PDcontrol", "torque"}, default "PDcontrol"
        "torque" を選ぶと action は正規化トルク指令（±torque_scale_Nm）
    action_scale_deg : float, default 45
        "PDcontrol" 制御時に action ∈ [-1,1] を ±action_scale_deg [deg] へ線形変換
    torque_scale_Nm : float, default 60
        "torque" 制御時の最大絶対トルク [Nm]

    Notes
    -----
    * Observation  :    dim vector 
    * Action       : 12dim vector (target joint pos or torque)
    * Reward       :
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
                 fall_height_th: float = 0.25,  # [m] 胴体の高さが25cm以下で転倒判定
                 fall_angle_th: float = 0.6,    # [rad] 胴体の姿勢が0.6radよりも傾くと転倒判定
                 obs_mode: str = "full",         # 観測データの種類を指定
                 action_scale_deg: float = 45.0, # [deg] アクションのスケールを指定
                 control_mode: str = "PDcontrol", # 制御方法を指定 PDcontrol or torque
                 torque_scale_Nm: float = 60.0,  # [Nm] トルクのスケールを指定
                 reward_mode: str = 'Progress',
                 calculate_manip: bool = True,
                 ):
        
        if model not in self._SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(self._SUPPORTED_MODELS)}")
        if obs_mode not in {"joint", "joint+base","nonManip" ,"full"}:
            raise ValueError(f"Unknown obs_mode '{obs_mode}'")
        if control_mode not in {"PDcontrol", "torque"}:
            raise ValueError(f"Unknown control_mode '{control_mode}'")

        super().__init__()
        self.model_name = model
        self.urdf_relpath = self._SUPPORTED_MODELS[model]
        self.render = render
        self.num_joint = 12
        self.actuated: List[int] = []

        # 終了条件の設定
        self.max_steps_per_episode = max_steps_per_episode
        self.fall_height_th = fall_height_th
        self.fall_angle_th = fall_angle_th
        self._ep_step = 0
        
        # 観測データの指定
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        # 行動出力の指定
        self._action_scale_rad = np.deg2rad(action_scale_deg) # deg -> rad
        self.torque_scale = torque_scale_Nm
        # 報酬関数の指定
        self.reward_mode = reward_mode 
        # 可操作度計算のフラグ
        self.calculate_manip = bool(calculate_manip)
        if self.obs_mode == "full" and not self.calculate_manip:
            raise ValueError("obs_mode='full' を使う場合は calculate_manip=True にしてください。")

        self._cid: Optional[int] = None
        self._robot: Optional[int] = None

        # 各脚ごとの関節とリンク情報
        self.leg_joints: Dict[str, list[int]] | None = None
        self.leg_ee_link: Dict[str, int]      | None = None

        self.dt = 1.0 / 400.0 # シミュレーションの1stepあたりの経過時間
        self._prev_action = np.zeros(self.num_joint, dtype=np.float32)
        # PDゲイン
        self.Kp = 40.0
        self.Kd = 1

        
        self.fall_penalty = 100 # 転倒時のペナルティ

        # 初期姿勢（ホームポジション）
        self.initial_joint = [0.0, 0.8, -1.5] #[hip, thigh, calf] の順に角度(rad)設定

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
        #print(f"[DEBUG] cid={self._cid}  isConnected={p.isConnected(self._cid)}")
        # --- シミュレーション初期化 -----------------------------------
        p.resetSimulation(physicsClientId=self._cid)
        p.setAdditionalSearchPath(self._data_path, physicsClientId=self._cid)# 再度パスを接続
        p.setGravity(0, 0, -9.8, physicsClientId=self._cid)
        p.setTimeStep(self.dt, physicsClientId=self._cid)
        

        # data/　の絶対パス
        from unitree_pybullet.unitree_pybullet import get_data_path
        data_dir = get_data_path()

        # URDF を絶対パスで読み込む
        plane_path = os.path.join(data_dir, "plane.urdf")
        robot_path = os.path.join(data_dir, self.urdf_relpath)
        #print(f"[DEBUG] plane  = {plane_path}")
        #print(f"[DEBUG] robot  = {robot_path}")
        #print(f"[DEBUG] exists = {os.path.isfile(robot_path)}")
        # 実際に読もうとしているファイルを表示
        #print("robot_path =", robot_path)
        #print("exists?    =", os.path.isfile(robot_path))

        # ファイル頭 200 文字だけ確認
        # with open(robot_path, "r", encoding="utf-8") as f:
        #     print(f.read(200))

        # Bullet 詳細ログを出す (一時的に)
        # p.connect(p.DIRECT, options="--verbose")
        # ---  VERBOSE ログを一時的に有効化 ------------------------------
        # p.connect(p.DIRECT, options="--verbose")   # ←別接続になるので注意
        # 代わりに以下:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)   # GUI無効なら無視される
        # ---------------------------------------------------------------


        p.loadURDF(plane_path, physicsClientId=self._cid)
        if self.model_name == "a1":
            self._robot = p.loadURDF(
                robot_path,
                [0, 0, 0.31],
                useFixedBase=False,
                flags=p.URDF_USE_SELF_COLLISION,
                physicsClientId=self._cid,
            )
        elif self.model_name == "aliengo":
            self._robot = p.loadURDF(
                robot_path,
                [0, 0, 0.39],
                useFixedBase=False,
                flags=p.URDF_USE_SELF_COLLISION,
                physicsClientId=self._cid,
            )

        #####################################
        # ↓ デバッグに使用
        #for j in range(p.getNumJoints(self._robot, physicsClientId=self._cid)):
        #    name = p.getJointInfo(self._robot, j, physicsClientId=self._cid)[1].decode()
        #    print(j, name)

        ## link・jointの取得確認 ###
        #if p.getNumJoints(self._robot, physicsClientId=self._cid) == 0:
        #    raise RuntimeError(
        #        f"{robot_path} has 0 joints. "
        #        "Use the SDK-supplied a1/urdf/a1.urdf or remove "
        #        "<transmission> and floating_base from the current file.")
        
        #n_joints = p.getNumJoints(self._robot, physicsClientId=self._cid)
        #print(f"[DEBUG] after load: uid={self._robot}  nJoints={n_joints}")
        #if n_joints == 0:
        #    raise RuntimeError(
        #        f"{robot_path} has 0 joints. "
        #        "Check that this is the SDK-supplied file and contains no "
        #        "<transmission>/<floating_base> tags."
        #    )
        ########################################################################
        
        self.leg_joints, self.leg_ee_link = mu.build_leg_maps(self._robot, self._cid) # 脚の関節とリンクのindexを取得
        self.actuated = (self.leg_joints["FR"] + self.leg_joints["FL"] + self.leg_joints["RR"] + self.leg_joints["RL"])
        self.num_joint = len(self.actuated) # = 12
        # 初期姿勢の設定
        self._set_initial_pose()

        #print("Num Joints", p.getNumJoints(self._robot, physicsClientId=self._cid))
        #print("DEBUG_A: after load local n_joints =", p.getNumJoints(self._robot, physicsClientId=self._cid))
        #print("DEBUG_B: hasattr(self,'num_joint')?", hasattr(self,'num_joint'), " value:", getattr(self,'num_joint','<NA>'))
        #print("DEBUG_C: id(self) =", id(self))
        #import inspect, sys
        #print("DEBUG_D: QuadEnv defined in:", inspect.getsourcefile(self.__class__))

        if self.control_mode == "torque"or self.control_mode =="PDcontrol":
            for j in self.actuated:
                p.setJointMotorControl2(self._robot, j, p.VELOCITY_CONTROL, force=0, physicsClientId=self._cid)

        self._prev_action.fill(0.0)   # ← 直前トルクの初期化
        self._ep_step = 0          # ← エピソードステップをリセット
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        
        self._ep_step += 1 

        if self.control_mode == "PDcontrol":
            # 初期姿勢をホームポジションに設定
            self.q0 = np.array( self.initial_joint * 4, dtype=np.float32)   # 12 次元
            # PDcontrol
            # targets = self._action_scale_rad * action
            targets = self.q0 + self._action_scale_rad * action

            # 現在角度と角速度を取得
            qs = np.array([p.getJointState(self._robot, j, physicsClientId=self._cid)[0] for j in self.actuated], dtype=np.float32)
            qds = np.array([p.getJointState(self._robot, j, physicsClientId=self._cid)[1] for j in self.actuated], dtype=np.float32)
            ## PD制御でトルクを計算 ##
            #print("qs= ", qs )
            #print("qds = ", qds)
            #print("target qs = ", targets)
            torques = self.Kp * (targets - qs) + self.Kd * (0.0 - qds)
            # トルクのクリッピング 
            torques = np.clip(torques,  -self.torque_scale, self.torque_scale)
           
            #print("torques = ", torques)
            # トルク制御として適用
            for j, tau in zip(self.actuated,torques):
                p.setJointMotorControl2(
                    self._robot,
                    j,
                    p.TORQUE_CONTROL,
                    force=float(tau),
                    physicsClientId=self._cid,
                )
            # 実際に発生したトルクを取得（次ステップ用）
            self._prev_action = torques.astype(np.float32)

        elif self.control_mode == "torque":
            torques = self.torque_scale * action
            torques = np.clip(torques,  -self.torque_scale, self.torque_scale)
            #print("taus = ", taus)
            for j, tau in zip(self.actuated, torques):
                p.setJointMotorControl2(
                    self._robot,
                    j,
                    p.TORQUE_CONTROL,
                    force=float(tau),
                    physicsClientId=self._cid,
                )
            self._prev_action = torques.astype(np.float32)
            
        p.stepSimulation(physicsClientId=self._cid)

        
        # 可操作度の計算
        """
        可操作度をロスや観測に入れたくはないが、比較のために計算したい場合があるため
        ここで計算して、観測に追加したい場合は再利用する
        """
        m4 = None
        if self.calculate_manip:
            m4 = mu.compute_leg_manipulability(
                self._robot, self.leg_joints,
                self.leg_ee_link,
                self._cid)

        obs = self._get_obs(manip_cached=m4)
        reward = self._reward(obs, action)

        # 転倒時のペナルティ        
        fallen = self._robot_fallen()
        if fallen:
            reward -= self.fall_penalty
        # エピソード修了条件 #
        terminated = fallen
        truncated = self._ep_step >= self.max_steps_per_episode

        if terminated or truncated:
            self._ep_step = 0

        

        info = {}
        if m4 is not None:
            info["manip_w"] = np.asarray(m4, dtype=np.float32)

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
        elif self.obs_mode == "nonManip": # 関節角度(12)＋関節角速度(12)＋ベース位置(3)＋姿勢：クォータニオン(4)＋移動速度(3)+姿勢の角速度(3) + １ステップ前のトルク(12)　＝　49次元
            high_obs = np.array([np.pi] * self.num_joint + [np.inf] * 37, dtype=np.float32)  
        elif self.obs_mode == "full": # 関節角度(12)＋関節角速度(12)＋ベース位置(3)＋姿勢：クォータニオン(4)＋移動速度(3)+姿勢の角速度(3) + １ステップ前のトルク(12) + 可操作度(4)　＝　53次元
            high_obs = np.array([np.pi] * self.num_joint + [np.inf] * 41, dtype=np.float32) 
        self.observation_space = gym.spaces.Box(-high_obs, high_obs, dtype=np.float32)

    def _connect_if_needed(self):
        if self._cid is not None and p.isConnected(self._cid):
            return
        mode = p.GUI if self.render else p.DIRECT
        self._cid = p.connect(mode)
        #self._cid = p.connect(p.GUI,    options="--start_demo_name=Physics Server")
        #self._cid = p.connect(mode, options="--search-path="+self._data_path)
        p.setAdditionalSearchPath(self._data_path, physicsClientId=self._cid)


    # ---- 観測データの作成 ---------------------------------------------------------------------------------
    def _get_obs(self, manip_cached: Optional[np.ndarray] = None) -> np.ndarray:
        # obs_mode in {"joint","joint+base","nonManip", "full"}
        qs = [p.getJointState(self._robot, j, physicsClientId=self._cid)[0] for j in self.actuated]
        qdots = [p.getJointState(self._robot, j, physicsClientId=self._cid)[1] for j in self.actuated]
        obs = qs + qdots

        if self.obs_mode in {"joint+base","nonManip", "full"}:
            pos, orn = p.getBasePositionAndOrientation(self._robot, self._cid)
            lin, ang = p.getBaseVelocity(self._robot, self._cid)
            obs += list(pos) + list(orn) + list(lin) + list(ang)

        if self.obs_mode in {"nonManip", "full"}:
            obs += list(self._prev_action)

        if self.obs_mode == "full":
            if not self.calculate_manip:
                raise RuntimeError("obs_mode='full' で calculate_manip=False は許可されていません。")
            if manip_cached is None:# すでに計算されている場合は再利用（キャッシュに無ければ計算）
                m4 = mu.compute_leg_manipulability(
                        self._robot, self.leg_joints,
                        self.leg_ee_link,
                        self._cid)
            else:
                m4 = manip_cached
            obs += list(m4)

        return np.array(obs, dtype=np.float32)
    
    # ------ 報酬定義 ---------------------------------------------------------
    def _reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        if self.reward_mode == "Progress":
            vx, vy = obs[31], obs[32]
            target_vx, target_vy = 0.8, 0.0 #[m/s]
            sigma_v = 0.25
            v_err_sq = (vx - target_vx) ** 2 + (vy - target_vy) ** 2
            return float(np.exp(-v_err_sq / sigma_v))

        elif self.reward_mode == "EnergeticProgress":
            """
            報酬関数の設計
            """
            # 前進速度を正とし、関節トルク²をエネルギ損失として差し引く
            vx = obs[31]
            progress = vx                      # 1 [m/s] ≒ 1 点
            qdot   = obs[12:24]                  # 各関節角速度 [rad/s]
            torque = self._prev_action           # 1 ステップ前に実際に発生したトルク [Nm]
            power  = np.dot(torque, qdot)        # ∑ τ·ω = 機械的仕事率
            beta   = 0.001                        # エネルギ重み (要チューニング)

            return float(progress - beta * power)

        # fallback: joint-angle penalty
        q = obs[:12]
        return -float(np.sum(q ** 2))


    # ---- 転倒の定義 ----------------------------------------------------
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
    
    # ---- 初期姿勢の設定 -----------------------------------------------
    def _set_initial_pose(self):
        """ 各脚関節をself.initial_jointの角度に設定 """
        # [hip, thigh, calf] の順に角度(rad)設定
        default_angles = np.array(self.initial_joint, dtype=np.float32)
        for leg in ("FL", "FR", "RL", "RR"):
            for k, jid in enumerate(self.leg_joints[leg]):
                p.resetJointState(
                    self._robot,
                    jid,
                    default_angles[k],
                    targetVelocity=0.0,
                    physicsClientId=self._cid,
                )

        