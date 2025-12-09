import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LastActionWrapper(gym.Wrapper):
    """
    [MultiDiscrete 版本]
    将上一帧的组合动作添加到观测空间。
    输出形状: (Action_Dim,)
    """

    def __init__(self, env):
        super().__init__(env)

        # 确认是 MultiDiscrete
        assert isinstance(env.action_space, spaces.MultiDiscrete), "此 Wrapper 仅支持 MultiDiscrete 动作空间"

        self.action_nvec = env.action_space.nvec
        self.action_dim = len(self.action_nvec)

        # 更新观测空间
        new_spaces = env.observation_space.spaces.copy()

        # last_action 是一个向量 (Action_Dim,)
        # 在 FrameStack 后会变成 (T, Action_Dim)
        new_spaces['last_action'] = spaces.Box(
            low=0,
            high=max(self.action_nvec),
            shape=(self.action_dim,),
            dtype=np.int64
        )
        self.observation_space = spaces.Dict(new_spaces)

        # 初始动作 (全0)
        self.current_last_action = np.zeros(self.action_dim, dtype=np.int64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_last_action = np.zeros(self.action_dim, dtype=np.int64)
        obs['last_action'] = self.current_last_action
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.current_last_action = np.array(action, dtype=np.int64)
        obs['last_action'] = self.current_last_action

        return obs, reward, terminated, truncated, info