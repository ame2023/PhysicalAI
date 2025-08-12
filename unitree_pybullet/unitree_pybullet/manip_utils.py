"""
manip_utils.py – Unitree A1/Laikago/Aliengo 用ユーティリティ
--------------------------------------------------------------
1) build_leg_maps(robot_id, pclient)
   ・各脚の [hip, upper/thigh, lower/calf] joint index
   ・各脚の toe(foot) link index
   を辞書で返す

2) compute_leg_manipulability(...)
   Yoshikawa 指標 M(q)=√det(JJᵀ) を 4脚ぶん計算し ndarray(4,) で返す
   PyBullet のヤコビアン列順は環境により
   - "qindex"（base6列 + 各関節, qIndex 準拠） base6列：ベースの回転3自由度 + 並進3自由度（の6自由度が関節のように効き、胴体のベース座標が変化）
   - "movable"（可動関節のみ）
   の2通りがあるため、実行時に自動判別し、かつベース並進 I(3) を検出して
   列を正しく抜き出す。
   
   dotP = dotX + dotR@r(q) + R@dr/dq * dotq
        = [I_3 , R[r(q)]_x , R@dr/dq]@[dotX ; omega ; dotq]
        

3) get_leg_ellipsoid_axes(...)
   可操作性楕円体の半軸と主軸方向、接地点座標を返す
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p

# -----------------------------------------------------------------------------
# 定数
# -----------------------------------------------------------------------------
_LEG_PREFIX = ("FL", "FR", "RL", "RR")


# -----------------------------------------------------------------------------
# 内部ヘルパ
# -----------------------------------------------------------------------------
def _infer_base_offset(J_lin: np.ndarray) -> int:
    #PyBullet のビルド・環境によっては、calculateJacobian の結果の列の先頭に
    #「ベース：角3＋並3 = 6列」が入る場合と、入らない場合があり
    #関節の列開始位置が毎回（= 環境ごと／プロセスごと）違い得るので、毎回自動で起点を検出しておく。
    """
    3x(total_cols) の線速度ヤコビアン J_lin の中から、
    連続する 3x3 の「ほぼ単位行列」（= ベース並進）を探し、
    その直後の列インデックスをオフセットとして返す。
    見つからなければ 0 を返す。

    3x3 の「ほぼ単位行列」：ベース並進に対する微分はほぼ単位行列になる(理論値は単位行列)
    """
    ncols = J_lin.shape[1]
    for k in range(max(0, ncols - 2)):
        block = J_lin[:, k:k+3]
        if block.shape == (3, 3) and np.allclose(block, np.eye(3), atol=1e-9):
            return k + 3
    return 0


def _warn_if_identity(J_leg: np.ndarray, leg: str):
    """デバッグ用：脚の 3x3 サブヤコビアンがほぼ単位なら警告"""
    if J_leg.shape == (3, 3) and np.allclose(J_leg, np.eye(3), atol=1e-9):
        print(f"[WARN] {leg}: J_leg is ~Identity. Likely picking base-translation columns. "
              f"Check base-offset detection.", flush=True)


def _get_movable_joint_ids(robot_id: int, pclient: int = 0) -> List[int]:
    """固定関節を除く jointId リスト（PyBulletのジョイント順）"""
    # p.JOINT_FIXED: 固定関節
    # p.getJointInfo()[2]：ジョイントの種別（JOINT_REVOLUTE/JOINT_PRISMATIC/JPINT_FIXED...）
    return [
        j for j in range(p.getNumJoints(robot_id, physicsClientId=pclient))
        if p.getJointInfo(robot_id, j, physicsClientId=pclient)[2] != p.JOINT_FIXED 
    ]


def _get_qindex_map(robot_id: int, pclient: int = 0) -> Dict[int, int]:
    """
    { jointId -> qIndex (>=0) } を返す。固定関節は含めない。
    注意: 多くのビルドでは qIndex は「関節のみで0始まり」。
          一方、calculateJacobian の列は先頭に base6（角3+並3）が乗ることがある。
    """
    qindex_map: Dict[int, int] = {}
    for j in range(p.getNumJoints(robot_id, physicsClientId=pclient)):
        ji = p.getJointInfo(robot_id, j, physicsClientId=pclient)
        joint_type = ji[2]
        q_index = ji[3]
        if joint_type != p.JOINT_FIXED and q_index >= 0:
            qindex_map[j] = q_index
    return qindex_map


def _build_state_qindex(robot_id: int, pclient: int = 0
                        ) -> Tuple[List[float], List[float], List[float], Dict[int, int]]:
    """
    qIndex（関節のみ）順で q/qd/qdd を作る。
    戻り: (q_full, qd_full, qdd_full, qindex_map)
    """
    qindex_map = _get_qindex_map(robot_id, pclient)
    if not qindex_map:
        raise RuntimeError("[_build_state_qindex] 位置DOFをもつ関節が見つかりません")

    total_dofs = max(qindex_map.values()) + 1  # qIndex は 0 始まり
    q_full = [0.0] * total_dofs
    qd_full = [0.0] * total_dofs
    qdd_full = [0.0] * total_dofs

    for jid, qidx in qindex_map.items():
        q_full[qidx] = p.getJointState(robot_id, jid, physicsClientId=pclient)[0]

    return q_full, qd_full, qdd_full, qindex_map


def _build_state_movable(robot_id: int, pclient: int = 0
                         ) -> Tuple[List[float], List[float], List[float], Dict[int, int]]:
    """
    可動関節のみの配列を構成。戻りの dict は {jointId -> 列インデックス(0..)}。
    """
    movable_ids = _get_movable_joint_ids(robot_id, pclient)
    if not movable_ids:
        raise RuntimeError("[_build_state_movable] 可動関節が見つかりません")

    q = [p.getJointState(robot_id, j, physicsClientId=pclient)[0] for j in movable_ids]
    zeros = [0.0] * len(movable_ids)
    col_map = {jid: i for i, jid in enumerate(movable_ids)}  # jointId -> 列
    return q, zeros, zeros, col_map


def _choose_jacobian_mode(robot_id: int, pclient: int, link_index: int) -> str:
    """
    ヤコビアンの列順モードを判別して 'qindex' or 'movable' を返す。
    既に p._mu_jac_mode があればそれを使う。
    """
    if hasattr(p, "_mu_jac_mode"):
        return p._mu_jac_mode  # type: ignore[attr-defined]

    toe_offset = [0.0, 0.0, -0.04]

    # 1) qindex を試す
    try:
        q_full, qd_full, qdd_full, _ = _build_state_qindex(robot_id, pclient)
        J_lin, _ = p.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=link_index,
            localPosition=toe_offset,
            objPositions=q_full,
            objVelocities=qd_full,
            objAccelerations=qdd_full,
            physicsClientId=pclient,
        )
        J = np.asarray(J_lin, dtype=float)
        # ベース I(3) ブロックの位置を検出してキャッシュ（列抽出で使う）
        base_off = _infer_base_offset(J)
        p._mu_jac_mode = "qindex"                 # type: ignore[attr-defined]
        p._mu_num_base_cols = int(base_off)       # type: ignore[attr-defined]
        print(f"[MU] Jacobian mode = qindex (base_off={base_off})")
        return "qindex"
    except Exception:
        pass

    # 2) movable を試す
    try:
        q, qd, qdd, _ = _build_state_movable(robot_id, pclient)
        J_lin, _ = p.calculateJacobian(
            robot_id, link_index, toe_offset, q, qd, qdd, physicsClientId=pclient
        )
        J = np.asarray(J_lin, dtype=float)
        base_off = _infer_base_offset(J)  # movable でも環境によっては base6 が先頭に来る
        p._mu_jac_mode = "movable"              # type: ignore[attr-defined]
        p._mu_num_base_cols = int(base_off)     # type: ignore[attr-defined]
        print(f"[MU] Jacobian mode = movable (base_off={base_off})")
        return "movable"
    except Exception as e:
        raise RuntimeError(f"[MU] ヤコビアンモード判別に失敗: {e}")


# -----------------------------------------------------------------------------
# joint / link マッピングを自動生成
# -----------------------------------------------------------------------------
def build_leg_maps(robot_id: int, pclient: int = 0
                   ) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """
    各脚ごとの joint index と toe/foot link index を辞書で返す
      leg_joints = { "FR": [hip, upper, lower], ... }
      leg_ee     = { "FR": toe_link_index, ... }
    """
    # Joint 名 -> index
    joint_name_to_idx = {
        p.getJointInfo(robot_id, j, physicsClientId=pclient)[1].decode(): j
        for j in range(p.getNumJoints(robot_id, physicsClientId=pclient))
    }

    leg_joints: Dict[str, List[int]] = {}
    leg_ee: Dict[str, int] = {}

    for leg in _LEG_PREFIX:
        # hip / upper(thigh) / lower(calf) を名前で解決（URDF差異に対応）
        try:
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

        # toe / foot の *リンク* index（joint ではなく子リンク）
        toe_idx = foot_idx = None
        for j in range(p.getNumJoints(robot_id, physicsClientId=pclient)):
            link_name = p.getJointInfo(robot_id, j, physicsClientId=pclient)[12].decode()
            if re.fullmatch(fr"{leg}_toe", link_name, re.I):
                toe_idx = j
            elif re.fullmatch(fr"{leg}_foot", link_name, re.I):
                foot_idx = j
        ee_idx = toe_idx if toe_idx is not None else foot_idx
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
    各脚の可操作度 M(q)=√(det(J Jᵀ)) を計算して ndarray(4,) を返す
      戻り順序は (_LEG_PREFIX) = (FL, FR, RL, RR)
    """

    # オフセット（足裏がちょうど接触点とならない場合があるため）
    """
    toe原点からさらに4cm下にオフセット
    URDF上ではtoeは
    A1:(0,0,-0.2)
    Aliengo:(0,0,-0.25)
    だけオフセットがされた固定関節をみているため必要が無ければ[0,0,0]にしても良い
    """
    #toe_offset = [0.0, 0.0, -0.04] # 接触パッド（感圧センサなど）の分だけオフセット（足裏の少し下を参照したい場合）
    toe_offset = [0.0, 0.0, 0.0] # 足裏を参照

    # 初回のみモード自動判別
    any_leg = next(iter(leg_ee.values()))
    mode = _choose_jacobian_mode(robot_id, pclient, any_leg)
    base_off_cached = int(getattr(p, "_mu_num_base_cols", 0))

    Ms: List[float] = []

    # 各脚でヤコビアンを取り、3列抜き出して可操作度を計算
    for leg in _LEG_PREFIX:
        link = leg_ee[leg]

        # --- q/qd/qdd & 列マップ（関節だけの長さにする！） ---
        if mode == "qindex":
            q, qd, qdd, col_map = _build_state_qindex(robot_id, pclient)
        else:
            q, qd, qdd, col_map = _build_state_movable(robot_id, pclient)

        toe_local = np.asarray(toe_offset, dtype=np.float64).reshape(3,)

        # --- ヤコビアン取得（3 x total_cols） ---
        J_lin, _ = p.calculateJacobian(
            bodyUniqueId=robot_id,
            linkIndex=link,
            localPosition=toe_local,
            objPositions=q,               # ← 可動DoF数に一致
            objVelocities=qd,
            objAccelerations=qdd,
            physicsClientId=pclient
        )
        J_lin = np.asarray(J_lin, dtype=np.float64)

        # ベース並進 I(3) の直後をオフセットとして使う
        base_off = _infer_base_offset(J_lin)# ヤコビアンの関節列が始まる列番号(回転3+並進3が先に現れる場合があるため)
        if base_off == 0:
            base_off = base_off_cached  # うまく検出できない環境向け保険

        # --- この脚の3関節の列を抽出（必ず base_off を加える！） ---
        try:
            jj_cols = [base_off + col_map[jid] for jid in leg_joints[leg]]
        except KeyError as e:
            raise RuntimeError(f"[{leg}] joint id not in column map: {e}. "
                               "URDFのjoint種別/並びを確認してください。")


        """
        J_lin は (3 × N) の行列（3行は x,y,z の線速度、N列はベース+関節）。
        欲しいのは「この脚の3関節だけ」に対応する 3列。
        """
        J_leg = J_lin[:, jj_cols]  # (3,3)
        _warn_if_identity(J_leg, leg)  # 誤抽出の早期検知

        # --- 可操作度 Yoshikawa: sqrt(det(J J^T)) ---
        JJt = J_leg @ J_leg.T
        detJJt = float(np.linalg.det(JJt))
        M = np.sqrt(detJJt) if detJJt > 0 else 0.0
        if M < 1e-12:
            M = 0.0

        Ms.append(M)

    return np.asarray(Ms, dtype=np.float32)


# -----------------------------------------------------------------------------
# 可操作性楕円体の主軸計算（可視化用）
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
    toe_offset = [0.0, 0.0, -0.04]

    # モードを判別
    any_leg = next(iter(leg_ee.values()))
    mode = _choose_jacobian_mode(robot_id, pclient, any_leg)
    base_off_cached = int(getattr(p, "_mu_num_base_cols", 0))

    result: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for leg in _LEG_PREFIX:
        link = leg_ee[leg]

        # q/qd/qdd と列マップ
        if mode == "qindex":
            q, qd, qdd, col_map = _build_state_qindex(robot_id, pclient)# col_map = JointID
        else:
            q, qd, qdd, col_map = _build_state_movable(robot_id, pclient)

        J_lin, _ = p.calculateJacobian(
            robot_id, link, toe_offset, q, qd, qdd, physicsClientId=pclient
        )
        J_lin = np.asarray(J_lin, dtype=np.float64)

        base_off = _infer_base_offset(J_lin)
        if base_off == 0:
            base_off = base_off_cached

        jj_cols = [base_off + col_map[jid] for jid in leg_joints[leg]]
        J_leg = J_lin[:, jj_cols]  # (3,3)

        # 楕円体行列 M = J J^T
        M = J_leg @ J_leg.T
        vals, vecs = np.linalg.eigh(M)# 固有値と固有ベクトルを取得
        radii = np.sqrt(np.clip(vals, 0.0, None))# 半径はsqrt(vals)

        # 接地点座標（ワールド座標）
        # 脚先の座標を取得して可操作性楕円体を描画
        state = p.getLinkState(robot_id, link, physicsClientId=pclient)# 指定リンクのワールド座標系での位置と姿勢を取得
        pos = np.array(state[0])# state[0]：ワールド座標系の位置ベクトル(x,y,z)

        result[leg] = (radii, vecs, pos)

    return result
