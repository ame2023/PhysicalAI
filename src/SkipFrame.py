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
        """
        skipした場合、その間のinfoを最後のステップのinfoに追加
        """
        total_reward = 0.0
        terminated, truncated = False, False
        infos = []
        for _ in range(self.skip):
            obs, reward, term, trunc, step_info = self.env.step(action)
            total_reward += reward
            infos.append(step_info)
            terminated |= term
            truncated |= trunc
            if terminated or truncated:
                break
        info = infos[-1]  # 最後のステップのinfo
        info["skip_infos"] = infos  # すべてのinfoを格納
        return obs, total_reward, terminated, truncated, info

    # Optional: ラッパー経由で reset を透過的に呼びたい場合
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class RenderEveryNSteps(gym.Wrapper):
    def __init__(self, env, render_every=4):
        super().__init__(env)
        self.render_every = render_every
        self._cnt = 0
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._cnt += 1
        if self._cnt % self.render_every == 0 and hasattr(self.env, "render"):
            self.env.render()
        return obs, r, term, trunc, info
