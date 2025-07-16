import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from unitree_pybullet.unitree_pybullet import QuadEnv

# ── 環境の初期化 ──
env = QuadEnv(model="aliengo", render=True)

# 最初に reset() を呼んでおく
out = env.reset()
if len(out) == 2:
    obs, info = out
else:
    obs, info = out  # 古い Gym バージョン

# ログ用リスト
manip_log = []
height_log = []
roll_log = []
pitch_log = []

# ── ランダムアクションでステップ ──
for step in range(15000):
    action = 30 * np.random.uniform(-1, 1, size=12)
    out = env.step(action)
    # Gymnasium vs Gym 判定
    if len(out) == 5:
        obs, rew, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, rew, done, info = out

    # manipulability は obs の末尾4要素
    manip = obs[-4:]
    manip_log.append(manip)

    # ベースの位置・姿勢を pybullet から取得
    pos, orn = p.getBasePositionAndOrientation(env._robot, physicsClientId=env._cid)
    rpy = p.getEulerFromQuaternion(orn)
    height_log.append(pos[2])
    roll_log.append(rpy[0])
    pitch_log.append(rpy[1])

    print(f"[step {step:03d}] reward={rew:.3f}, manip={manip}")
    time.sleep(1/240)

    if done:
        out = env.reset()
        if len(out) == 2:
            obs, info = out
        else:
            obs, info = out

# ── ログの統計と可視化 ──
manip_array = np.stack(manip_log, axis=0)
print("manip range:", np.min(manip_array, axis=0), "〜", np.max(manip_array, axis=0))

# 脚ごとの manipulability 推移
plt.figure()
for i in range(4):
    plt.plot(manip_array[:, i], label=f"leg{i}")
plt.legend()
plt.xlabel("step")
plt.ylabel("manipulability")
plt.title("Leg Manipulability over Time")
plt.show()

# ベースの高さ・ロール・ピッチ 推移
plt.figure()
plt.plot(height_log, label="height")
plt.plot(roll_log, label="roll")
plt.plot(pitch_log, label="pitch")
plt.legend()
plt.xlabel("step")
plt.title("Base Height, Roll, and Pitch over Time")
plt.show()
