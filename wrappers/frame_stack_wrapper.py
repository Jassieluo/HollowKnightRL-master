import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque


class FrameStackWrapper(gym.Wrapper):
    """
    [时序堆叠版] (Time-Channel-Height-Width)

    不同于传统的通道堆叠，此 Wrapper 会增加一个时间维度 T。
    并调整图像维度顺序以适配 PyTorch (T, C, H, W)。

    输出形状变换:
    1. Image/Mask (3D): (H, W, C) -> (T, C, H, W)
    2. State/Vector (1D): (N,)    -> (T, N)
    """

    def __init__(self, env, frame_stack=8):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frames = {}
        new_obs_spaces = {}

        for key, space in env.observation_space.items():
            self.frames[key] = deque(maxlen=frame_stack)

            # === A类：图像或掩码 (H, W, C) ===
            if key == 'image' or key.startswith('mask_'):
                # 原始 Gym 形状通常是 (H, W, C)
                h, w, c = space.shape

                # 目标形状: (T, C, H, W)
                new_shape = (frame_stack, c, h, w)

                new_obs_spaces[key] = spaces.Box(
                    low=0,
                    high=255 if key == 'image' else 1,
                    shape=new_shape,
                    dtype=space.dtype  # 通常是 uint8
                )
                print(f"  - [3D -> 4D] {key}: {space.shape} -> {new_shape} (T, C, H, W)")

            # === B类：向量状态 (N,) ===
            else:
                shape = space.shape
                # 目标形状: (T, N)
                new_shape = (frame_stack,) + shape

                new_obs_spaces[key] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=new_shape,
                    dtype=np.float32  # 即使原先是int (如last_action)，堆叠后通常转float进网络，或保持原样
                )
                # 如果是 last_action 且需要保持 int，可以在这里特殊判断，但 float32 更通用
                if key == 'last_action':
                    new_obs_spaces[key].dtype = np.int64

                print(f"  - [1D -> 2D] {key}: {shape} -> {new_shape} (T, N)")

        self.observation_space = spaces.Dict(new_obs_spaces)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        for key in self.frames:
            self.frames[key].clear()

        # 填充初始帧 T 次
        for _ in range(self.frame_stack):
            self._append_obs(obs)

        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._append_obs(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _append_obs(self, obs):
        for key, value in obs.items():
            if key in self.frames:
                # 存入队列时保持原始单帧格式
                self.frames[key].append(value)

    def _get_stacked_obs(self):
        stacked_obs = {}
        for key, queue in self.frames.items():
            frames_list = list(queue)

            # === A类处理: (T, C, H, W) ===
            if key == 'image' or key.startswith('mask_'):
                # 1. 列表中的帧是 (H, W, C)
                # 2. 先转置每个帧变成 (C, H, W)
                transposed_frames = [np.transpose(f, (2, 0, 1)) for f in frames_list]

                # 3. 在第0维堆叠 -> (T, C, H, W)
                stacked_obs[key] = np.stack(transposed_frames, axis=0)

            # === B类处理: (T, N) ===
            else:
                # 直接在第0维堆叠 -> (T, N)
                stacked_obs[key] = np.stack(frames_list, axis=0)

        return stacked_obs