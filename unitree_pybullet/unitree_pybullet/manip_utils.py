"""
manip_utils.py – Unitree A1/Laikago/Aliengo 用ユーティリティ
--------------------------------------------------------------
1) build_leg_maps(robot_id, pclient)
   ・各脚の [hip, upper/thigh, lower/calf] joint index
   ・各脚の toe(foot) link index
   を辞書で返す

2) compute_leg_manipulability(...)
   Yoshikawa 指標 M(q)=√det(JJᵀ) を 4脚ぶん計算し ndarray(4,) で返す
"""
from __future__ import annotations
import re
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p

# -----------------------------------------------------------------------------
# joint / link マッピングを自動生成
# -----------------------------------------------------------------------------
_LEG_PREFIX = ("FL", "FR", "RL", "RR")


def build_leg_maps(robot_id: int, pclient: int = 0
                   ) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    各脚ごとの joint index と toe/foot link index を辞書で返す
      leg_joints = { "FR": [hip, upper, lower], ... }
      leg_ee     = { "FR": toe_link_index, ... }
    """
    # --- ① Joint 名一覧 ------------------------------------------------------
    joint_name_to_idx = {
        p.getJointInfo(robot_id, j, physicsClientId=pclient)[1].decode(): j
        for j in range(p.getNumJoints(robot_id, physicsClientId=pclient))
    }

    leg_joints, leg_ee = {}, {}
    for leg in _LEG_PREFIX:
        # hip / upper(thigh) / lower(calf)
        try:
            # --- 3 DoF joint indices (lazy lookup に変更) --------------------------
            hip_key = f"{leg}_hip_joint"
            upp_key = f"{leg}_upper_joint"
            thi_key = f"{leg}_thigh_joint"
            low_key = f"{leg}_lower_joint"
            cal_key = f"{leg}_calf_joint"

            hip_idx = joint_name_to_idx[hip_key]
            if upp_key in joint_name_to_idx:
                upper_idx = joint_name_to_idx[upp_key]
            else:
                upper_idx = joint_name_to_idx[thi_key]

            if low_key in joint_name_to_idx:
                lower_idx = joint_name_to_idx[low_key]
            else:
                lower_idx = joint_name_to_idx[cal_key]
        except KeyError as e:
            raise KeyError(f"[build_leg_maps] '{leg}' の主要ジョイントが見つかりません: {e}")

        leg_joints[leg] = [hip_idx, upper_idx, lower_idx]

        # toe / foot link（joint ではなく *子リンク* の index）
        regex = re.compile(fr"{leg}_(toe|foot)", re.I)
        ee_idx = None
        for j in range(p.getNumJoints(robot_id, physicsClientId=pclient)):
            link_name = p.getJointInfo(robot_id, j, physicsClientId=pclient)[12].decode()
            if regex.fullmatch(link_name):
                ee_idx = j
                break
        if ee_idx is None:
            raise KeyError(f"[build_leg_maps] '{leg}' の toe/foot link が見つかりません")
        leg_ee[leg] = ee_idx

    return leg_joints, leg_ee


# -----------------------------------------------------------------------------
#  可操作度 Yoshikawa 指標
# -----------------------------------------------------------------------------
def compute_leg_manipulability(
    robot_id: int,
    leg_joints: Dict[str, List[int]],
    leg_ee: Dict[str, int],
    pclient: int = 0,
) -> np.ndarray:
    """
    各脚の可操作度 M(q)=√det(J Jᵀ) を計算して ndarray(4,) を返す
      戻り順序は (_LEG_PREFIX) = (FL, FR, RL, RR)
    """
    # --- 可動 DOF の joint index リストを取得 ------------------------------
    movable_ids = [
        j for j in range(p.getNumJoints(robot_id, physicsClientId=pclient))
        if p.getJointInfo(robot_id, j, physicsClientId=pclient)[2] != p.JOINT_FIXED
    ]
    n_dof = len(movable_ids)

    # 各脚 3 関節が movable_ids 内で何番目か → 絶対 index → 可動 DOF の相対 index
    leg_joints_rel = {
        leg: [movable_ids.index(jid) for jid in idxs]
        for leg, idxs in leg_joints.items()
    }

    # --- joint positions/vel/acc 配列（長さ = n_dof） -----------------------
    q_full = [p.getJointState(robot_id, j, physicsClientId=pclient)[0] for j in movable_ids]
    zeros  = [0.0] * n_dof

    # --- 各脚のヤコビ行列 → M(q) ------------------------------------------
    Ms = []
    for leg in _LEG_PREFIX:
        jj = leg_joints_rel[leg]        # 3 ints
        link = leg_ee[leg]

        J_lin, _ = p.calculateJacobian(
            robot_id,
            link,
            [0, 0, 0],  # local position
            q_full,
            zeros,
            zeros,
            physicsClientId=pclient,
        )                           # J_lin shape = (3, n_dof)

        J_leg = np.asarray(J_lin)[:, jj]   # (3,3)
        Ms.append(np.sqrt(np.linalg.det(J_leg @ J_leg.T) + 1e-12))

    return np.asarray(Ms, dtype=np.float32)

# -----------------------------------------------------------------------------
# 可操作性楕円体の主軸計算 （可操作性楕円体を可視化するために使用）
# -----------------------------------------------------------------------------
def get_leg_ellipsoid_axes(
    robot_id: int,
    leg_joints: Dict[str, List[int]],
    leg_ee: Dict[str, int],
    pclient: int = 0
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    各脚ごとの可操作性楕円体の情報を返す:
      半軸長 sqrt(eigenvalues) と主軸方向 eigenvectors, 及び接地点の3D座標
    戻り値: { leg: (radii[3], axes[3x3], position[3]) }
    """
    # movable DOF と状態取得
    movable_ids = [
        j for j in range(p.getNumJoints(robot_id, physicsClientId=pclient))
        if p.getJointInfo(robot_id, j, physicsClientId=pclient)[2] != p.JOINT_FIXED
    ]
    n_dof = len(movable_ids)
    q_full = [p.getJointState(robot_id, j, physicsClientId=pclient)[0] for j in movable_ids]
    zeros = [0.0] * n_dof

    radii_dict = {}
    for leg in _LEG_PREFIX:
        jj = [movable_ids.index(jid) for jid in leg_joints[leg]]
        link = leg_ee[leg]
        # Jacobian
        J_lin, _ = p.calculateJacobian(
            robot_id, link, [0, 0, 0], q_full, zeros, zeros, physicsClientId=pclient
        )
        J_leg = np.asarray(J_lin)[:, jj]
        # 楕円体行列 M = J J^T
        M = J_leg @ J_leg.T
        vals, vecs = np.linalg.eigh(M)
        # 半軸長 = sqrt(eigenvalues)
        radii = np.sqrt(np.clip(vals, 0, None))
        # 接地点座標
        state = p.getLinkState(robot_id, link, physicsClientId=pclient)
        pos = np.array(state[0])  # ワールド座標位置
        radii_dict[leg] = (radii, vecs, pos)
    return radii_dict

