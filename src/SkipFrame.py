# src/wrappers/action_repeat.py
"""
skipの回数だけエージェントは同じアクションを取り続ける
"""


import gymnasium as gym

class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False
        info = {}
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated |= term
            truncated |= trunc
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    # Optional: ラッパー経由で reset を透過的に呼びたい場合
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
