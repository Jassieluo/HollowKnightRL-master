import json
import os
from typing import Optional, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from sb3_contrib.common.maskable.distributions import MaskableMultiCategoricalDistribution
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.type_aliases import RNNStates

# 假设 expert_logic 在同级目录下
from tools.expert_logic import SpatialExpertLogic


class ReferenceNetExtractor(BaseFeaturesExtractor):
    """
    [适配版] ReferenceNetExtractor
    适配重构后的 FrameStackWrapper (T, C, H, W)
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # === 1. 视觉流 (3D CNN) ===
        img_space = observation_space['image']
        # img_space.shape 是 (T, C, H, W)
        t, c, h, w = img_space.shape

        self.n_frames = t
        self.img_c = c

        print(f"[RefNet] Visual Input: (T={t}, C={c}, H={h}, W={w}) -> Conv3d")

        self.cnn = nn.Sequential(
            # Conv3d 输入: (B, C, T, H, W)
            nn.Conv3d(self.img_c, 32, kernel_size=(2, 3, 3), stride=(1, 2, 2)),
            # nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.Conv3d(32, 48, kernel_size=(2, 3, 3), stride=(1, 1, 1)),
            # nn.BatchNorm3d(48),
            nn.SiLU(),
            nn.Conv3d(48, 64, kernel_size=(2, 3, 3), stride=(1, 1, 1)),
            # nn.BatchNorm3d(64),
            nn.SiLU(),
            nn.Conv3d(64, 64, kernel_size=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )

        # === 2. 向量流 (Vector) ===
        self.vec_keys = [
            k for k in observation_space.keys()
            if k not in ['image', 'expert_action'] and not k.startswith('mask_')
        ]

        self.target_vec_dim = 13

        self.vec_net = nn.Sequential(
            nn.Linear(self.target_vec_dim, 128),
            # nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 64)
        )

        # self.final_proj = nn.Identity()
        self._features_dim = 64 + 64
        # self._features_dim = 64

    def forward(self, observations):
        # --- A. 向量流处理 ---
        # observations['hero_state']: (B, T, N)
        # 取时间维度 T 的最后一个索引 (-1) -> (B, N)
        curr_hero = observations['hero_state'][:, -1, :]
        curr_boss = observations['boss_state'][:, -1, :]

        vec_features = torch.cat([
            # curr_hero[:, 0:1] / (curr_hero[:, 1:2] + 1e-6),  # HP Ratio
            curr_hero[:, 2:3] / 100.0,  # Soul
            curr_hero[:, 10:11], # Can Dash
            curr_hero[:, 3:5] / 50.0,  # Hero Pos
            curr_hero[:, 7:8],  # Hero Facing
            curr_hero[:, 5:7] / 15.0,  # Hero Vel
            # curr_boss[:, 1:2] / 1000.0,  # Boss HP
            curr_boss[:, 2:4] / 50.0,  # Boss Pos
            curr_boss[:, 4:6] / 15.0, # Boss vel
            # curr_boss[:, 0:1],  # Boss Exists
            # (curr_boss[:, 4:5] > 0).float(),  # Boss Facing
            curr_boss[:, 6:8] / 50.

        ], dim=1)

        vec_out = self.vec_net(vec_features.float())
        # --- B. 视觉流处理 ---
        # observations['image']: (B, T, C, H, W)
        img = observations['image'].float() / 255.0

        # Conv3d 需要: (B, C, T, H, W)
        # permute: 0->B, 2->C, 1->T, 3->H, 4->W
        img_in = img.permute(0, 2, 1, 3, 4)

        cnn_out = self.cnn(img_in)
        return torch.cat([cnn_out, vec_out], dim=1)
        # return torch.cat([vec_out], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class RecurrentMaskableExpertPolicy(RecurrentActorCriticPolicy):
    """
    结合 LSTM、动作掩码和专家系统的策略

    修复的主要问题：
    1. 正确的 LSTM 状态处理
    2. 动作掩码支持
    3. 专家系统集成
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule,
            expert_epsilon: float = 0.8,
            min_epsilon: float = 0.05,
            epsilon_decay: float = 0.99995,
            env_interface=None,
            features_extractor_class=ReferenceNetExtractor,
            features_extractor_kwargs=None,
            **kwargs
    ):
        # 保存专家相关参数
        self.expert_epsilon = expert_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.env_interface = env_interface

        # 初始化特征提取器参数
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        # 确保 net_arch 存在，否则设置为默认值
        if 'net_arch' not in kwargs:
            kwargs['net_arch'] = dict(pi=[256], vf=[256])

        # 确保 LSTM 参数存在
        if 'lstm_hidden_size' not in kwargs:
            kwargs['lstm_hidden_size'] = 512
        if 'n_lstm_layers' not in kwargs:
            kwargs['n_lstm_layers'] = 1
        if 'enable_critic_lstm' not in kwargs:
            kwargs['enable_critic_lstm'] = True
        if 'shared_lstm' not in kwargs:
            kwargs['shared_lstm'] = False

        # 初始化父类
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )

        self.move_action = 6
        self.attack_action = 8

        self.action_net = nn.Sequential(
            nn.Linear(self.net_arch['pi'][-1], 256),
            # nn.BatchNorm1d(128),
            # nn.SiLU(),
            # nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(256, 14)
        )

        # self.action_net_backbone = nn.Sequential(
        #     nn.Linear(self.features_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU()
        # )
        #
        # self.action_net_head = nn.ModuleList([nn.Linear(128, self.move_action),
        #                                       nn.Linear(128, self.attack_action)])

        self.value_net = nn.Sequential(
            nn.Linear(self.net_arch['vf'][-1], 256),
            # nn.BatchNorm1d(256),
            # nn.SiLU(),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            # nn.SiLU(),
            # nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

        # 创建掩码感知的动作分布
        self.action_dims = action_space.nvec.tolist() if hasattr(action_space, 'nvec') else [action_space.n]
        self.maskable_dist = MaskableMultiCategoricalDistribution(self.action_dims)

        # 专家逻辑（如果需要的话）
        self.expert_logic = SpatialExpertLogic()

        print(f"[INFO] Policy initialized with action dims: {self.action_dims}")
        print(f"[INFO] Features dim: {self.features_dim}")
        print(f"[INFO] LSTM hidden size: {kwargs['lstm_hidden_size']}")

    def set_env_interface(self, env):
        """设置环境接口（用于专家系统获取状态）"""
        self.env_interface = env

    def forward(
            self,
            obs: torch.Tensor,
            lstm_states: RNNStates,
            episode_starts: torch.Tensor,
            deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        """
        前向传播（训练和评估时使用）

        修复：正确处理 LSTM 状态，支持动作掩码
        """
        # 1. 提取特征
        features = self.extract_features(obs)

        # 应用动作掩码
        # action_masks = obs['action_masks'][:, -1, :]
        action_masks = None

        # 2. 分离 actor 和 critic 特征（如果需要）
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        # 3. 处理 LSTM 序列
        # Actor LSTM
        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features.unsqueeze(0),  # 增加序列维度
            lstm_states.pi,
            episode_starts,
            self.lstm_actor
        )

        # Critic LSTM（如果启用）
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features.unsqueeze(0),
                lstm_states.vf,
                episode_starts,
                self.lstm_critic
            )
        elif self.shared_lstm:
            # 共享 LSTM，但不反向传播 critic 的梯度
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic 使用前馈网络
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states.vf

        # 4. 确保维度正确
        # latent_pi 和 latent_vf 现在是 (seq_len, batch, hidden_dim)
        # 我们需要取最后一个时间步，并确保是 2D (batch, hidden_dim)
        if latent_pi.dim() == 3:
            latent_pi = latent_pi[-1]  # 取最后一个时间步
        if latent_vf.dim() == 3:
            latent_vf = latent_vf[-1]

        # 进一步确保维度
        if latent_pi.dim() == 1:
            latent_pi = latent_pi.unsqueeze(0)
        if latent_vf.dim() == 1:
            latent_vf = latent_vf.unsqueeze(0)

        # 5. 通过 MLP 提取器
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # 6. 价值估计
        values = self.value_net(latent_vf)

        # 7. 动作分布（支持掩码）
        action_logits = self.action_net(latent_pi)
        # action_logits = torch.cat([head(self.action_net_backbone(latent_pi)) for head in self.action_net_head], dim=-1)
        distribution = self.maskable_dist.proba_distribution(action_logits)

        # 应用动作掩码
        # action_masks = self.env_interface._get_action_mask(self.env_interface.pre_state) if self.env_interface is not None else None

        if action_masks is not None:
            distribution.apply_masking(action_masks)

        # 8. 采样动作
        actions = distribution.get_actions(deterministic=deterministic)

        # 9. 专家系统注入（仅在推理时）
        final_actions = actions
        if not self.training and self.env_interface is not None:
            batch_size = actions.shape[0]
            rand_vals = torch.rand(batch_size, device=actions.device)
            should_use_expert = (rand_vals < self.expert_epsilon)

            if should_use_expert.any():
                if batch_size == 1 and should_use_expert[0]:
                    try:
                        # 从环境获取原始状态
                        raw_state = self.env_interface.pre_state
                        if raw_state:
                            # 使用专家逻辑计算动作
                            exp_act_np = self.expert_logic.compute_action(raw_state)
                            exp_act_tensor = torch.from_numpy(exp_act_np).to(actions.device).long()
                            final_actions = exp_act_tensor.unsqueeze(0)
                    except Exception as e:
                        print(f"[WARNING] Expert system failed: {e}")

                # 衰减 epsilon
                self.expert_epsilon = max(self.min_epsilon, self.expert_epsilon * self.epsilon_decay)

        # 10. 计算对数概率
        log_prob = distribution.log_prob(final_actions)

        # 11. 返回新的 LSTM 状态
        lstm_states = RNNStates(lstm_states_pi, lstm_states_vf)

        return final_actions, values, log_prob, lstm_states

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        """
        [必须重写] 在 PPO 训练更新阶段被调用。
        我们需要在这里应用掩码，否则训练时的 LogProb 和 Entropy 计算是错的。
        """
        # 1. 提取特征
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        # 2. LSTM 处理 (注意：这里必须使用传入的 lstm_states，不能用 get_initial_states)
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        else:
            latent_vf = vf_features

        # 3. MLP 头
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # 4. 计算价值和 Logits
        values = self.value_net(latent_vf)
        action_logits = self.action_net(latent_pi)

        # 5. [关键] 生成分布并应用掩码
        distribution = self.action_dist.proba_distribution(action_logits)

        # action_masks = self._get_action_masks(obs)
        # if action_masks is not None:
        #     distribution.apply_masking(action_masks)

        # 6. 计算 Log Prob 和 Entropy
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def __getstate__(self):
        """序列化时剔除环境接口"""
        state = super().__getstate__()
        state['env_interface'] = None
        return state

    def __setstate__(self, state):
        """反序列化"""
        super().__setstate__(state)
        if 'env_interface' not in self.__dict__:
            self.env_interface = None

    def get_initial_states(self, batch_size=1):
        """获取LSTM的初始状态"""
        device = next(self.parameters()).device

        # 创建初始LSTM状态
        if self.lstm_actor is not None:
            # 对于LSTM，状态包含 (h, c)
            h_actor = torch.zeros(
                self.lstm_actor.num_layers,
                batch_size,
                self.lstm_actor.hidden_size
            ).to(device)
            c_actor = torch.zeros_like(h_actor).to(device)

            if self.lstm_critic is not None and not self.shared_lstm:
                h_critic = torch.zeros(
                    self.lstm_critic.num_layers,
                    batch_size,
                    self.lstm_critic.hidden_size
                ).to(device)
                c_critic = torch.zeros_like(h_critic).to(device)
            else:
                h_critic, c_critic = h_actor, c_actor
        else:
            # 如果没有LSTM，返回None
            return None

        return ((h_actor, c_actor), (h_critic, c_critic))

class RewardAdjustmentCallback(BaseCallback):
    def __init__(self, env, reward_config_path=None, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.reward_config_path = reward_config_path
        self.last_adjustment_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_adjustment_step >= 10000:
            self._check_reward_update()
            self.last_adjustment_step = self.num_timesteps
        return True

    def _check_reward_update(self):
        if self.reward_config_path and os.path.exists(self.reward_config_path):
            try:
                with open(self.reward_config_path, 'r') as f:
                    new_rewards = json.load(f)

                # 寻找最底层的 env
                current_env = self.env.envs[0]
                while hasattr(current_env, 'env'):
                    if hasattr(current_env, 'reward_params'):
                        break
                    current_env = current_env.env

                if hasattr(current_env, 'reward_params'):
                    for key, value in new_rewards.items():
                        if key in current_env.reward_params:
                            current_env.reward_params[key] = value
            except Exception:
                pass