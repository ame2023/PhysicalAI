# src/make_video.py
import os
from typing import Optional, Sequence, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

from unitree_pybullet.unitree_pybullet.QuadGymEnv import QuadEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from src.models import ExtendModel


class VideoRecorder:
    """PyBullet の mp4 録画ユーティリティ（ffmpeg 必須・GUI）。"""
    def __init__(self, env: Optional[Any] = None, out_path: str = "episode.mp4",
                 client_id: Optional[int] = None, camera: Optional[dict] = None) -> None:
        self.env = env
        self.out_path = out_path
        self.client_id = client_id
        self.camera_cfg = camera or {}
        self._log_id: Optional[int] = None

    def start(self):
        self._resolve_client_id()
        if self.camera_cfg:
            self.set_camera(**self.camera_cfg)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self._log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4, self.out_path, physicsClientId=self.client_id
        )
        if self._log_id < 0:
            print(f"[WARN] Failed to start MP4 logging: {self.out_path}")
        return self

    def stop(self):
        if self._log_id is not None and self._log_id >= 0:
            p.stopStateLogging(self._log_id, physicsClientId=self.client_id)
            self._log_id = None

    def set_camera(self, distance: Optional[float] = None, yaw: Optional[float] = None,
                   pitch: Optional[float] = None, target: Optional[Sequence[float]] = None):
        self._resolve_client_id()
        try:
            cam = p.getDebugVisualizerCamera(physicsClientId=self.client_id)
            current_dist = float(cam[10]); current_yaw = float(cam[8]); current_pitch = float(cam[9])
            current_target = tuple(cam[11]) if cam[11] is not None else (0.0, 0.0, 0.0)
        except Exception:
            current_dist, current_yaw, current_pitch = 1.0, 0.0, -30.0
            current_target = (0.0, 0.0, 0.0)
        d = current_dist if distance is None else float(distance)
        y = current_yaw if yaw is None else float(yaw)
        pi = current_pitch if pitch is None else float(pitch)
        t = current_target if target is None else tuple(target)
        p.resetDebugVisualizerCamera(d, y, pi, t, physicsClientId=self.client_id)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def _resolve_client_id(self):
        if self.client_id is not None:
            return
        cid = None
        if self.env is not None:
            try:
                if hasattr(self.env, "get_attr"):
                    cids = self.env.get_attr("_cid")
                    if isinstance(cids, (list, tuple)) and len(cids) > 0:
                        cid = cids[0]
            except Exception:
                pass
            if cid is None:
                try:
                    base = getattr(self.env, "envs", [self.env])[0]
                    while hasattr(base, "env"):
                        base = base.env
                    cid = getattr(base, "_cid", None)
                except Exception:
                    pass
        if cid is None:
            cid = p.connect(p.GUI)
        self.client_id = cid


class SoftwareVideoRecorder:
    """
    ffmpeg PATH 不要のソフトウェア録画：
    - computeView/Projection で view/proj を自前算出
    - OpenGL → 失敗/真っ黒なら TinyRenderer にフォールバック
    - ターゲットfpsで間引き、最後は最低2フレーム保証
    """
    def __init__(self, out_path: str, client_id: Optional[int] = None, env: Optional[Any] = None,
                 fps: int = 60, size: Optional[Sequence[int]] = None, camera: Optional[dict] = None,
                 fov: float = 60.0, renderer: str = "auto") -> None:
        self.out_path = out_path
        self.client_id = client_id
        self.env = env
        self.fps = int(max(1, fps))
        self.size = tuple(size) if size is not None else None
        self.camera_cfg = camera or {}
        self.fov = float(fov)
        self.renderer_mode = renderer.lower()  # "auto"|"opengl"|"tiny"

        self._writer = None
        self._use_cv2 = False
        self._frame_count = 0
        self._last_rgb = None

    def start(self):
        self._resolve_client_id()
        if self.camera_cfg:
            self._set_camera(**self.camera_cfg)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        return self

    def stop(self):
        # 最低2フレーム保証
        if self._writer is not None and self._frame_count < 2 and self._last_rgb is not None:
            if self._use_cv2:
                import cv2
                self._writer.write(cv2.cvtColor(self._last_rgb, cv2.COLOR_RGB2BGR))
            else:
                self._writer.append_data(self._last_rgb)
            self._frame_count += 1
        if self._writer is not None:
            try:
                self._writer.close() if not self._use_cv2 else self._writer.release()
            except Exception:
                pass
            self._writer = None

    def capture_frame(self):
        # 1) 画面サイズ
        w0, h0, view_dbg, proj_dbg = self._get_dbg_camera()
        W = self.size[0] if self.size else w0
        H = self.size[1] if self.size else h0
        aspect = float(W) / float(max(1, H))

        # 2) カメラパラメータ（cfg優先/なければDebug）
        d, y, pi, t = self._get_camera_params_from_cfg_or_debug()

        # 3) 自前 view/proj
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=t, distance=d, yaw=y, pitch=pi, roll=0.0, upAxisIndex=2
        )
        proj = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=aspect, nearVal=0.01, farVal=max(5.0, d*10.0)
        )

        # 4) レンダラ選択・フォールバック
        if self.renderer_mode == "opengl":
            renderer_list = [p.ER_BULLET_HARDWARE_OPENGL]
        elif self.renderer_mode == "tiny":
            renderer_list = [p.ER_TINY_RENDERER]
        else:
            renderer_list = [p.ER_BULLET_HARDWARE_OPENGL, p.ER_TINY_RENDERER]

        rgb = None
        cand = None
        for rend in renderer_list:
            img = p.getCameraImage(
                width=W, height=H, viewMatrix=view, projectionMatrix=proj,
                renderer=rend, physicsClientId=self.client_id
            )
            rgba = np.reshape(np.array(img[2], dtype=np.uint8), (H, W, 4))
            cand = rgba[..., :3]
            # 真っ黒回避の簡易判定（平均輝度 > 0.5）
            if cand.mean() > 0.5:
                rgb = cand
                break
        if rgb is None:
            rgb = cand  # フォールバック失敗時でも保存は継続

        self._last_rgb = rgb

        if self._writer is None:
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (W, H))
                self._use_cv2 = True
            except Exception:
                import imageio
                self._writer = imageio.get_writer(
                    self.out_path, fps=self.fps,
                    codec='libx264', pixelformat='yuv420p',
                    macro_block_size=None
                )
                self._use_cv2 = False

        if self._use_cv2:
            import cv2
            self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            self._writer.append_data(rgb)
        self._frame_count += 1

    # ---- helpers ----
    def _resolve_client_id(self):
        if self.client_id is not None:
            return
        cid = None
        if self.env is not None:
            try:
                if hasattr(self.env, "get_attr"):
                    cids = self.env.get_attr("_cid")
                    if isinstance(cids, (list, tuple)) and len(cids) > 0:
                        cid = cids[0]
            except Exception:
                pass
            if cid is None:
                try:
                    base = getattr(self.env, "envs", [self.env])[0]
                    while hasattr(base, "env"):
                        base = base.env
                    cid = getattr(base, "_cid", None)
                except Exception:
                    pass
        if cid is None:
            cid = p.connect(p.GUI)
        self.client_id = cid

    def _set_camera(self, distance: Optional[float] = None, yaw: Optional[float] = None,
                    pitch: Optional[float] = None, target: Optional[Sequence[float]] = None):
        try:
            cam = p.getDebugVisualizerCamera(physicsClientId=self.client_id)
            current_dist = float(cam[10]); current_yaw = float(cam[8]); current_pitch = float(cam[9])
            current_target = tuple(cam[11]) if cam[11] is not None else (0.0, 0.0, 0.0)
        except Exception:
            current_dist, current_yaw, current_pitch = 1.2, 0.0, -30.0
            current_target = (0.0, 0.0, 0.0)
        d = current_dist if distance is None else float(distance)
        y = current_yaw if yaw is None else float(yaw)
        pi = current_pitch if pitch is None else float(pitch)
        t = current_target if target is None else tuple(target)
        p.resetDebugVisualizerCamera(d, y, pi, t, physicsClientId=self.client_id)

    def _get_dbg_camera(self) -> Tuple[int, int, Any, Any]:
        cam = p.getDebugVisualizerCamera(physicsClientId=self.client_id)
        w, h = int(cam[0]), int(cam[1])
        view, proj = cam[2], cam[3]
        return w, h, view, proj

    def _get_camera_params_from_cfg_or_debug(self) -> Tuple[float, float, float, Tuple[float,float,float]]:
        d = y = pi = None
        t = None
        if self.camera_cfg:
            d = self.camera_cfg.get("distance", None)
            y = self.camera_cfg.get("yaw", None)
            pi = self.camera_cfg.get("pitch", None)
            t = self.camera_cfg.get("target", None)
        try:
            cam = p.getDebugVisualizerCamera(physicsClientId=self.client_id)
            d0 = float(cam[10]); y0 = float(cam[8]); pi0 = float(cam[9])
            t0 = tuple(cam[11]) if cam[11] is not None else (0.0, 0.0, 0.0)
        except Exception:
            d0, y0, pi0, t0 = 1.2, 0.0, -30.0, (0.0, 0.0, 0.0)
        d = d0 if d is None else float(d)
        y = y0 if y is None else float(y)
        pi = pi0 if pi is None else float(pi)
        t = t0 if t is None else tuple(t)
        return d, y, pi, t


def run_best_model_test(logdir: str, cfg, video_enable: bool | None = None) -> None:
    """
    best_model で 1 エピソードだけ検証し、以下を logdir/best_model_test/ に保存する:
      - mp4 動画（episode.mp4）※ video_enable=True のとき
      - 各脚の可操作度推移図（manip_FL.png, manip_FR.png, manip_RL.png, manip_RR.png）
      - 可操作度の時系列（manip_series.npy）

    仕様:
      - VecNormalize は vecnormalize_best.pkl を優先（無ければ vecnormalize.pkl）
      - テスト時は calculate_manip=True を強制
      - deterministic は cfg.test_deterministic を参照（既定 False）
      - 録画は "auto|pybullet|software"（cfg.video_record_mode）。auto は ffmpeg 無ければ software
      - 等速化はソフト録画時に「小数累積」で厳密化（video_fps, 既定 60）
      - video_enable=False の場合は録画せず、検証と可操作度図のみ保存
    """
    import os, shutil, time
    from typing import Any
    import numpy as np
    import matplotlib.pyplot as plt

    from unitree_pybullet.unitree_pybullet.QuadGymEnv import QuadEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
    from src.models import ExtendModel

    # ------------- 出力ディレクトリ -------------
    test_dir = os.path.join(logdir, "best_model_test")
    os.makedirs(test_dir, exist_ok=True)

    # ------------- モデルと正規化統計の特定 -------------
    best_model_path = os.path.join(logdir, "best_model.zip")
    if not os.path.isfile(best_model_path):
        fallback = os.path.join(logdir, f"{cfg.algo.lower()}_{cfg.unitree_model}.zip")
        best_model_path = fallback if os.path.isfile(fallback) else None

    stats_best = os.path.join(logdir, "vecnormalize_best.pkl")
    stats_path = stats_best if (best_model_path and os.path.isfile(stats_best)) else os.path.join(logdir, "vecnormalize.pkl")

    if best_model_path is None or not os.path.isfile(best_model_path):
        print("[WARN] best model が見つかりません。検証をスキップします。")
        return
    if not os.path.isfile(stats_path):
        print(f"[WARN] 正規化統計が見つかりません: {stats_path}")
        return

    # ------------- 評価用環境（検証時のみ可操作度を強制計算） -------------
    test_env_base = QuadEnv(
        model=cfg.unitree_model,
        render=True,  # 録画のため GUI
        max_steps_per_episode=cfg.max_steps_per_episode,
        fall_height_th=cfg.fall_height_th,
        fall_angle_th=cfg.fall_angle_th,
        obs_mode=cfg.obs_mode,
        action_scale_deg=cfg.action_scale_deg,
        control_mode=cfg.control_mode,
        torque_scale_Nm=cfg.torque_scale_Nm,
        reward_mode=cfg.reward_mode,
        calculate_manip=True,        # ★ config に依らず可操作度を計算
    )
    test_env = DummyVecEnv([lambda: test_env_base])
    test_env = VecNormalize.load(stats_path, test_env)
    test_env.training = False       # 観測正規化の更新を止める
    test_env.norm_reward = False    # 報酬正規化は学習時のみ
    test_env = VecMonitor(test_env, filename=os.path.join(test_dir, "monitor.csv"))

    # ------------- 推論設定 -------------
    test_deterministic = bool(getattr(cfg, "test_deterministic", True))

    # ------------- モデル読込 & 初期化 -------------
    model_best = ExtendModel.load(best_model_path, env=test_env, device=cfg.device, model_name=cfg.algo)
    obs = test_env.reset()

    # ------------- 録画関連の設定（動画ON/OFF, モード, FPS, 実時間） -------------
    # video_enable: 引数優先、未指定なら cfg.video_enable（無ければ True）
    if video_enable is None:
        video_enable = bool(getattr(cfg, "video_enable", True))

    # 実時間スリープは動画ONのときのみ適用（OFFなら高速化）
    target_fps = int(getattr(cfg, "video_fps", 60))
    target_fps = max(1, min(120, target_fps))
    realtime_effective = (True if not hasattr(cfg, "video_realtime") else bool(cfg.video_realtime)) and video_enable

    # env.dt の取得（unwrap）
    base_env: Any = getattr(test_env, "envs", [test_env])[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env
    dt = float(getattr(base_env, "dt", 1.0 / 60.0))
    sim_fps = 1.0 / dt

    # 録画モード決定（video_enable=False の場合は強制スキップ）
    mode = getattr(cfg, "video_record_mode", "auto")  # "auto"|"pybullet"|"software"
    use_software = False
    if video_enable:
        if mode == "software":
            use_software = True
        elif mode == "auto":
            use_software = (shutil.which("ffmpeg") is None)

    video_path = os.path.join(test_dir, "episode.mp4")

    # ------------- レコーダ初期化 -------------
    # VideoRecorder / SoftwareVideoRecorder は make_video.py 内のクラスを想定
    rec = None
    sw = None
    if video_enable and not use_software:
        try:
            rec = VideoRecorder(env=test_env, out_path=video_path,
                                camera=getattr(cfg, "video_camera", None)).start()
            if rec._log_id is None or rec._log_id < 0:
                print("[WARN] PyBullet MP4 ロギング開始に失敗。ソフトウェア録画へ切替します。")
                rec.stop()
                rec = None
                use_software = True
        except Exception as e:
            print(f"[WARN] PyBullet ロギング初期化で例外。ソフトウェア録画へ切替: {e}")
            rec = None
            use_software = True

    if video_enable and use_software:
        sw = SoftwareVideoRecorder(
            out_path=video_path,
            env=test_env,
            fps=target_fps,
            size=getattr(cfg, "video_size", None),
            camera=getattr(cfg, "video_camera", None),
        ).start()
        # 初期フレーム（真っ黒や0枚回避）
        sw.capture_frame()

    # ------------- ループ本体（厳密等速: 小数累積/Bresenham風） -------------
    steps_per_frame_f = sim_fps / float(target_fps)  # 例: dt=1/400, fps=60 → 6.666...
    acc = 0.0

    manip_seq = []        # [T, 4]
    ep_steps, ep_ret = 0, 0.0
    done_flag = False

    try:
        while not done_flag:
            action, _ = model_best.predict(obs, deterministic=test_deterministic)
            step_out = test_env.step(action)

            # Gymnasium と Gym 互換の done 取り出し
            if len(step_out) == 5:
                obs, rewards, terminated, truncated, infos = step_out
                done_flag = bool(terminated[0] or truncated[0])
                info0 = infos[0]
                r0 = float(rewards[0]) if np.ndim(rewards) else float(rewards)
            else:
                obs, rewards, dones, infos = step_out
                done_flag = bool(dones[0])
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
                r0 = float(rewards[0]) if np.ndim(rewards) else float(rewards)

            ep_steps += 1
            ep_ret += r0

            # 可操作度の収集
            w = info0.get("manip_w", None)
            if w is not None:
                manip_seq.append(np.asarray(w, dtype=float))

            # 録画（video_enable=True のときのみ）
            if video_enable and use_software:
                acc += 1.0
                while acc >= steps_per_frame_f:
                    sw.capture_frame()
                    acc -= steps_per_frame_f

            # 実時間同期（video_enable=True のときのみ）
            if realtime_effective:
                time.sleep(max(0.0, (1.0 / target_fps) * 0.98))

        # 終了時に保険で1枚
        if video_enable and use_software:
            sw.capture_frame()

    finally:
        # レコーダ停止と保存確認
        if video_enable:
            if use_software and sw is not None:
                sw.stop()
            if (not use_software) and rec is not None:
                rec.stop()

            if os.path.isfile(video_path):
                try:
                    size = os.path.getsize(video_path)
                    print(f"[INFO] Saved video: {video_path} ({size/1024:.1f} KiB)")
                except Exception:
                    print(f"[INFO] Saved video: {video_path}")
            else:
                print(f"[WARN] 動画ファイルが見つかりませんでした: {video_path}")

        # 環境クローズ（VecEnv/Env の順で安全に）
        try:
            test_env.close()
        except Exception:
            pass

    # テスト集計の表示
    print(
        f"[INFO] Test finished: steps={ep_steps}, return={ep_ret:.2f}, "
        f"stats='{os.path.basename(stats_path)}', model='{os.path.basename(best_model_path)}', "
        f"video={'on' if video_enable else 'off'}, mode={'software' if (video_enable and use_software) else ('pybullet' if video_enable else '-')}, "
        f"fps={target_fps}"
    )

    # ------------- 各脚ごとの可操作度プロットと保存（1枚に4脚は重ねない） -------------
    if len(manip_seq) > 0:
        M = np.vstack(manip_seq)  # [T, 4]
        leg_names = ["FL", "FR", "RL", "RR"]
        for li, name in enumerate(leg_names):
            plt.figure(figsize=(6, 4))
            plt.plot(M[:, li])
            plt.xlabel("Step")
            plt.ylabel("Manipulability")
            plt.title(f"Manipulability - {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(test_dir, f"manip_{name}.png"))
            plt.close()
        np.save(os.path.join(test_dir, "manip_series.npy"), M)
