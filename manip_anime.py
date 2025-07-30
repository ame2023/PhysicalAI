import os
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from unitree_pybullet.unitree_pybullet import get_data_path
from unitree_pybullet.unitree_pybullet import manip_utils as mu

# ===== ユーザ設定 =====
model      = "a1"      # {"a1","laikago","aliengo"}
leg        = "FR"      # {"FR","FL","RR","RL"}
render_gui = True      # ← PyBullet GUI を開くと「単脚のみ」が見えます
save_path  = None      # 例: "kunoji_walk.gif"（gif保存）
# 描画解像度
ELL_U = 40
ELL_V = 20
# 判定しきい値
D_PERP_EPS   = 1e-3     # [m] HAA軸からの最短距離（タイプ1）
KNEE_EPS_DEG = 5.0      # [deg] 膝一直線（タイプ2）近傍
# タイムライン
FPS        = 20
CYCLE      = 160        # 1 周期
HOLD_SING  = 2 * FPS    # 2 秒停止
TOTAL_INIT = CYCLE

def hide_everything_but_leg(robot, keep_links, cid):
    """keep_links に含まれないリンクを不可視＆衝突無効化"""
    try:
        p.changeVisualShape(robot, -1, rgbaColor=[1,1,1,0], physicsClientId=cid)
    except Exception:
        pass
    n = p.getNumJoints(robot, physicsClientId=cid)
    keep = set(keep_links)
    for j in range(n):
        if j not in keep:
            try:
                p.changeVisualShape(robot, j, rgbaColor=[1,1,1,0], physicsClientId=cid)
            except Exception:
                pass
            try:
                p.setCollisionFilterGroupMask(robot, j, 0, 0, physicsClientId=cid)
            except Exception:
                pass
            # 任意: 動力学をさらに消したい場合
            # try:
            #     p.changeDynamics(robot, j, mass=0.0, physicsClientId=cid)
            # except Exception:
            #     pass

def haa_axis_world(robot, hip_jid, cid):
    info = p.getJointInfo(robot, hip_jid, physicsClientId=cid)
    axis_local = np.array(info[13])  # joint axis in parent-link frame
    parent = info[16]
    if parent == -1:
        _, parent_orn = p.getBasePositionAndOrientation(robot, physicsClientId=cid)
    else:
        parent_orn = p.getLinkState(robot, parent, physicsClientId=cid)[1]
    R_parent = np.array(p.getMatrixFromQuaternion(parent_orn)).reshape(3,3)
    a_world = R_parent @ axis_local
    return a_world / (np.linalg.norm(a_world) + 1e-12)

def movable_joint_ids(robot, cid):
    return [j for j in range(p.getNumJoints(robot, physicsClientId=cid))
            if p.getJointInfo(robot, j, physicsClientId=cid)[2] != p.JOINT_FIXED]

def ik_freeze_others_setup(robot, movable_ids, free_ids, cid, rest_bias=None):
    """対象脚free_idsのみ可動。他は現在値で固定。rest_biasでくの字バイアス。"""
    q_now = [p.getJointState(robot, j, physicsClientId=cid)[0] for j in movable_ids]
    lower, upper, ranges, rest = [], [], [], []
    for j, q in zip(movable_ids, q_now):
        info = p.getJointInfo(robot, j, physicsClientId=cid)
        lo, hi = info[8], info[9]
        if j in free_ids and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            lower.append(lo); upper.append(hi); ranges.append(hi - lo)
        else:
            lower.append(q);  upper.append(q);  ranges.append(0.1)  # 実質固定
        rest.append(q)
    if rest_bias is not None:
        for k, j in enumerate(movable_ids):
            if j in free_ids and j in rest_bias:
                rest[k] = rest_bias[j]
    return lower, upper, ranges, rest

def knee_angle_deg(joints_w, toe_w):
    v1 = joints_w[2] - joints_w[1]  # thigh->calf
    v2 = toe_w      - joints_w[2]   # calf->toe
    den = (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12)
    cosang = np.clip(np.dot(v1, v2)/den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def ellipsoid_mesh_from_J(J, center):
    U, s, _ = np.linalg.svd(J)
    s = np.maximum(s, 1e-12)
    u = np.linspace(0, 2*np.pi, ELL_U)
    v = np.linspace(0, np.pi,   ELL_V)
    x = s[0] * np.outer(np.cos(u), np.sin(v))
    y = s[1] * np.outer(np.sin(u), np.sin(v))
    z = s[2] * np.outer(np.ones_like(u), np.cos(v))
    E = U @ np.vstack([x.ravel(), y.ravel(), z.ravel()])
    X, Y, Z = E.reshape(3, x.shape[0], x.shape[1])
    return X + center[0], Y + center[1], Z + center[2]

def main():
    cid = p.connect(p.GUI if render_gui else p.DIRECT)

    # --- URDF ---
    data_dir = get_data_path()
    rel = {"a1":"a1/urdf/a1.urdf",
           "laikago":"laikago/urdf/laikago.urdf",
           "aliengo":"aliengo/urdf/aliengo.urdf"}[model]
    robot_path = os.path.join(data_dir, rel)
    if not os.path.isfile(robot_path):
        raise FileNotFoundError(f"URDF not found: {robot_path}")
    p.setAdditionalSearchPath(os.path.dirname(robot_path), physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(),          physicsClientId=cid)

    base_z = {"a1":0.31, "laikago":0.39, "aliengo":0.39}[model]
    _ = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"), physicsClientId=cid)
    robot = p.loadURDF(robot_path, [0,0,base_z], useFixedBase=False, physicsClientId=cid)

    # --- マップ・可動関節 ---
    leg_joints, leg_ee = mu.build_leg_maps(robot, pclient=cid)
    jids = leg_joints[leg]  # [HAA, HFE, KFE]
    ee   = leg_ee[leg]
    mids = movable_joint_ids(robot, cid)

    # === 対象脚以外を不可視/衝突無効化 ===
    keep_links = set(jids + [ee])
    hide_everything_but_leg(robot, keep_links, cid)

    # --- 全脚ホーム（くの字寄り初期） ---
    home = [0.0, 0.8, -1.5]
    for lp in ("FR","FL","RR","RL"):
        for k, jid in enumerate(leg_joints[lp]):
            p.resetJointState(robot, jid, home[k], physicsClientId=cid)

    # HAA軸（世界）と基底
    a_w   = haa_axis_world(robot, jids[0], cid)         # 単位ベクトル
    hip_w = np.array(p.getLinkState(robot, jids[0], physicsClientId=cid)[0])
    toe_w0 = np.array(p.getLinkState(robot, ee,        physicsClientId=cid)[0])

    tmp   = np.array([0,0,1]) if abs(np.dot(a_w,[0,0,1]))<0.9 else np.array([0,1,0])
    n_lat = np.cross(a_w, tmp);  n_lat /= (np.linalg.norm(n_lat)+1e-12)  # 横
    n_ver = np.cross(a_w, n_lat); n_ver /= (np.linalg.norm(n_ver)+1e-12) # 上下

    # 軸方向中心・振幅（前後 6cm）、横振幅 3cm、上下 4cm
    r0 = toe_w0 - hip_w
    s_center = float(np.dot(r0, a_w))
    s_amp    = 0.06
    y_amp    = 0.03
    z_amp    = 0.04

    def foot_traj(t):
        """t∈[0,1): くの字歩行の足先軌道（HAA軸基底）"""
        if t < 0.5:  # stance: 後方へ
            ph = t / 0.5
            s   = s_center - s_amp * ph
            z   = 0.0
        else:       # swing: 前方へ + 上げ
            ph = (t - 0.5) / 0.5
            s   = (s_center - s_amp) + 2*s_amp*ph
            z   = z_amp * np.sin(np.pi*ph)
        y = y_amp * np.sin(2*np.pi*t)  # 必ず 0 を横切る → タイプ1発生
        return hip_w + s*a_w + y*n_lat + z*n_ver

    # IK：対象脚3関節のみ可動、休止姿勢を「くの字」寄りに
    rest_bias = { jids[1]: 0.9,  jids[2]: -1.2 }  # HFE/KFE を屈曲側に
    low, upp, rng, rest = ik_freeze_others_setup(robot, mids, set(jids), cid, rest_bias=rest_bias)

    # Matplotlib 描画
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    # 停止制御
    hold_count = 0
    held_target = None
    type1_seen, type2_seen = False, False   # 1 周回あたり 1 回だけホールド

    def knee_angle_deg_local(joints, toe_r):
        return knee_angle_deg(joints, toe_r)

    def ellipsoid_mesh_from_J_local(J, center):
        return ellipsoid_mesh_from_J(J, center)

    def draw_one(target_toe_world):
        # IK
        sol = p.calculateInverseKinematics(
            robot, ee, target_toe_world,
            lowerLimits=low, upperLimits=upp, jointRanges=rng, restPoses=rest,
            maxNumIterations=200, residualThreshold=1e-6,
            physicsClientId=cid
        )
        for q, j in zip(sol, mids):
            p.resetJointState(robot, j, q, physicsClientId=cid)

        # 位置
        hip = np.array(p.getLinkState(robot, jids[0], physicsClientId=cid)[0])
        th  = np.array(p.getLinkState(robot, jids[1], physicsClientId=cid)[0])
        cf  = np.array(p.getLinkState(robot, jids[2], physicsClientId=cid)[0])
        toe = np.array(p.getLinkState(robot, ee,      physicsClientId=cid)[0])

        # 原点=ヒップ座標
        joints = np.vstack([hip, th, cf]) - hip
        toe_r  = toe - hip

        # HAA 軸の直線
        L = 0.25
        axis_pts = np.vstack([-L*a_w, L*a_w])

        # 計測
        d_perp   = float(np.linalg.norm(np.cross(toe_r, a_w)))       # タイプ1近傍
        knee_deg = knee_angle_deg_local(joints, toe_r)                # タイプ2近傍

        zeros = [0.0]*len(mids)
        q_full = [p.getJointState(robot, j, physicsClientId=cid)[0] for j in mids]
        jrel = [mids.index(j) for j in jids]
        J_lin, _ = p.calculateJacobian(robot, ee, [0,0,0], q_full, zeros, zeros, physicsClientId=cid)
        J = np.asarray(J_lin)[:, jrel]
        svals = np.linalg.svd(J, compute_uv=False)
        smin  = svals.min()
        s_show = 0.0 if smin < 1e-8 else smin
        w = float(np.sqrt(np.linalg.det(J @ J.T) + 1e-24))
        w_show = 0.0 if w < 1e-12 else w

        # Matplotlib 描画（楕円体＋棒線）
        ax.cla()
        X,Y,Z = ellipsoid_mesh_from_J_local(J, toe_r)
        ax.plot_surface(X, Y, Z, alpha=0.5, linewidth=0)
        ax.plot(axis_pts[:,0], axis_pts[:,1], axis_pts[:,2], '-', linewidth=2, label="HAA axis")
        ls  = '-.' if (abs(180.0 - knee_deg) < KNEE_EPS_DEG) else '-'
        lw  = 3.0 if (abs(180.0 - knee_deg) < KNEE_EPS_DEG) else 2.0
        ax.plot(joints[:,0], joints[:,1], joints[:,2], ls, linewidth=lw, label="hip→thigh→calf")
        ax.plot([joints[2,0], toe_r[0]], [joints[2,1], toe_r[1]], [joints[2,2], toe_r[2]],
                '-', linewidth=3.0, label="calf→toe")
        ax.scatter(0,0,0, s=70, marker='o', label="hip (origin)")
        ax.scatter(joints[1,0], joints[1,1], joints[1,2], s=55, marker='o', label="thigh")
        ax.scatter(joints[2,0], joints[2,1], joints[2,2], s=55, marker='o', label="calf")
        ax.scatter(toe_r[0], toe_r[1], toe_r[2], s=90, marker='*',
                   label=("toe (near HAA axis)" if d_perp < D_PERP_EPS else "toe (EE)"))
        ax.set_title(
            f"{model.upper()}-{leg}  Kunoji-walk (origin=HIP, centered at TOE)\n"
            f"d_perp={d_perp*1e3:.1f} mm | knee={knee_deg:.1f}° | σ_min={s_show:.3e} | w={w_show:.3e}"
        )
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        lim = 0.25
        ax.set_xlim(-lim, +lim); ax.set_ylim(-lim, +lim); ax.set_zlim(-lim, +lim)
        ax.legend(loc="upper right")

        return toe, d_perp, knee_deg

    hold_count = 0
    held_target = None
    type1_seen, type2_seen = False, False

    def update(fi):
        nonlocal hold_count, held_target, type1_seen, type2_seen
        if hold_count > 0:
            target = held_target
            hold_count -= 1
        else:
            t = (fi % CYCLE) / CYCLE
            target = foot_traj(t)

        toe, d_perp, knee_deg = draw_one(target)

        # タイプ1：HAA軸上で 2 秒停止
        if (d_perp < D_PERP_EPS) and (hold_count == 0) and (not type1_seen):
            hip = np.array(p.getLinkState(robot, jids[0], physicsClientId=cid)[0])
            toe_r = toe - hip
            s = float(np.dot(toe_r, a_w))
            held_target = hip_w + s*a_w
            hold_count  = HOLD_SING
            type1_seen  = True

        # タイプ2：膝一直線に到達したら 2 秒停止（くの字軌道では通常避ける）
        if (abs(180.0 - knee_deg) < KNEE_EPS_DEG) and (hold_count == 0) and (not type2_seen):
            held_target = np.array(p.getLinkState(robot, ee, physicsClientId=cid)[0])
            hold_count  = HOLD_SING
            type2_seen  = True

    ani = FuncAnimation(fig, update, frames=TOTAL_INIT + HOLD_SING*2, interval=int(1000/FPS), repeat=True)

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == ".gif":
            ani.save(save_path, writer="pillow", fps=FPS)
        else:
            ani.save(save_path, writer="ffmpeg", fps=FPS)
        print(f"saved: {save_path}")
    else:
        plt.show()

    p.disconnect(cid)

if __name__ == "__main__":
    main()
