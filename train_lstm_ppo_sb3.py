import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import gc  # 引入垃圾回收

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 你的自定义模块
from envs.HollowKnightEnv_sb3 import HollowKnightEnv, Scenes
from model.model_sb3 import RecurrentMaskableExpertPolicy, ReferenceNetExtractor
from wrappers.frame_stack_wrapper import FrameStackWrapper
from wrappers.action_wrapper import LastActionWrapper
from wrappers.wrapper_tools import unwrap_to_base
from tools.expert_logic import SpatialExpertLogic

# ==============================================================================
# 1. 超参数配置 (HYPERPARAMS)
# ==============================================================================
HYPERPARAMS = {
    "run_name": "HK_MultiGPU_Training",
    "seed": 42,

    # --- [修改点] 设备分配 ---
    # 你可以在这里指定不同的卡，例如 "cuda:0" 和 "cuda:1"
    "devices": {
        "pretrain": "cuda:1",  # 预训练跑在 0 号卡
        "ppo": "cuda:1"  # PPO 跑在 1 号卡 (如果没有多卡，就都写 cuda:0)
    },

    # --- 环境参数 ---
    "env": {
        "host": "127.0.0.1",
        "port": 5555,
        "scene": Scenes.HORNET_1,
        "image_resolution": (128, 128),
        "mask_resolution": (128, 128),
        "frame_stack": 4,
        "action_space_dims": [3, 3, 2, 2, 2, 2],
    },

    # --- 阶段 1: 专家预训练 ---
    "pretrain": {
        "enabled": True,
        "collect_steps": 5000,
        "epochs": 10,
        "batch_size": 8,
        "seq_len": 32,
        "learning_rate": 3e-4,
        "val_coef": 0.001,
        "gamma": 0.99,
    },

    # --- 阶段 2: PPO 强化学习 ---
    "ppo": {
        "total_timesteps": 2_000_000,
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "lstm_hidden_size": 512,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    },

    "expert_guide": {
        "initial_epsilon": 1.0,
        "final_epsilon": 0.0,
        "decay_steps": 10000,
    },

    "paths": {
        "save_dir": "./checkpoints/",
        "log_dir": "./sb3_tensorboard/",
        "resume_from": None,
    }
}


# ==============================================================================
# 2. 工具函数 (保持不变)
# ==============================================================================

def mask_fn(env):
    return unwrap_to_base(env)._get_action_mask(unwrap_to_base(env).pre_state)


def make_env(config):
    env_conf = config["env"]
    env = HollowKnightEnv(
        host=env_conf["host"],
        port=env_conf["port"],
        scene=env_conf["scene"],
        image_resolution=env_conf["image_resolution"],
        mask_resolution=env_conf["mask_resolution"],
        use_grayscale=True,
        use_visual_attention=True
    )
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    env = LastActionWrapper(env)
    if env_conf["frame_stack"] > 1:
        env = FrameStackWrapper(env, frame_stack=env_conf["frame_stack"])
    return env


class ExpertEpsilonDecayCallback(BaseCallback):
    def __init__(self, initial_eps, final_eps, decay_steps, verbose=0):
        super().__init__(verbose)
        self.initial = initial_eps
        self.final = final_eps
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        current_eps = self.initial + progress * (self.final - self.initial)
        if hasattr(self.model.policy, "expert_epsilon"):
            self.model.policy.expert_epsilon = current_eps
        if self.num_timesteps % 5000 == 0 and self.verbose > 0:
            print(f"[ExpertGuide] Step {self.num_timesteps}: epsilon = {current_eps:.4f}")
        return True


# ==============================================================================
# 3. 预训练逻辑 (小幅修改，适配 device 参数)
# ==============================================================================

class ExpertPretrainer:
    def __init__(self, model, env, config, device):
        self.model = model
        self.env = env
        self.config = config["pretrain"]
        self.device = device  # [修改] 明确传入 device
        self.expert_logic = SpatialExpertLogic()
        self.action_dims = config["env"]["action_space_dims"]

    # ... _collect_data 方法保持不变 ...
    def _collect_data(self):
        # (这里代码与之前一致，省略以节省篇幅，逻辑完全不用变)
        print(f"[Pretrain] 开始收集数据... 目标: {self.config['collect_steps']} 步")
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        return_buffer = []

        base_env = unwrap_to_base(self.env.envs[0])
        obs = self.env.reset()
        self.expert_logic.reset_state()
        current_episode_rewards = []
        steps = 0

        while steps < self.config['collect_steps']:
            raw_state = base_env.pre_state
            if raw_state:
                expert_action = self.expert_logic.compute_action(raw_state)
            else:
                expert_action = np.zeros(6, dtype=int)

            next_obs, reward, done, info = self.env.step([expert_action])

            obs_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy() for k, v in obs.items()}
            obs_buffer.append(obs_copy)
            action_buffer.append(expert_action)
            current_episode_rewards.append(reward[0])
            obs = next_obs
            steps += 1

            if done[0] or steps == self.config['collect_steps']:
                R = 0
                episode_returns = []
                for r in reversed(current_episode_rewards):
                    R = r + self.config['gamma'] * R
                    episode_returns.insert(0, R)
                return_buffer.extend(episode_returns)
                current_episode_rewards = []
                obs = self.env.reset()
                self.expert_logic.reset_state()
                if steps % 1000 == 0:
                    print(f"  已收集 {steps} 步...")

        return obs_buffer, np.array(action_buffer), np.array(return_buffer)

    def _prepare_dataset(self, obs_list, act_arr, ret_arr):
        # ... 保持不变，注意 .to(self.device) 已经使用了传入的 device ...
        seq_len = self.config['seq_len']
        total = len(obs_list)
        num_seqs = total // seq_len
        cutoff = num_seqs * seq_len

        obs_list = obs_list[:cutoff]
        act_arr = act_arr[:cutoff]
        ret_arr = ret_arr[:cutoff]

        print(f"[Pretrain] 整理为 LSTM 序列: {num_seqs} 条数据 (SeqLen={seq_len})")

        tensor_obs = {}
        for k in obs_list[0].keys():
            data = np.array([item[k][0] for item in obs_list])
            t = torch.tensor(data).to(self.device)  # 使用 self.device

            if k == 'image' or k.startswith('mask_'):
                if t.dtype == torch.uint8: t = t.float() / 255.0
            elif k == 'last_action':
                t = t.float()

            shape = t.shape[1:]
            tensor_obs[k] = t.view(num_seqs, seq_len, *shape)

        tensor_act = torch.tensor(act_arr).long().to(self.device).view(num_seqs, seq_len, -1)
        tensor_ret = torch.tensor(ret_arr).float().to(self.device).view(num_seqs, seq_len, 1)

        return tensor_obs, tensor_act, tensor_ret

    def train(self):
        # ... 保持不变 ...
        obs, acts, rets = self._collect_data()
        t_obs, t_acts, t_rets = self._prepare_dataset(obs, acts, rets)

        optimizer = torch.optim.Adam(self.model.policy.parameters(), lr=self.config['learning_rate'])
        batch_size = self.config['batch_size']
        num_seqs = t_acts.shape[0]
        seq_len = self.config['seq_len']

        self.model.policy.train()
        print(f"[Pretrain] 开始监督训练 (Device: {self.device})...")

        for epoch in range(self.config['epochs']):
            epoch_a_loss = 0
            epoch_c_loss = 0
            perm = torch.randperm(num_seqs)

            for i in range(0, num_seqs, batch_size):
                indices = perm[i:i + batch_size]
                if len(indices) == 0: continue
                current_bs = len(indices)

                b_obs = {k: v[indices] for k, v in t_obs.items()}
                b_acts = t_acts[indices]
                b_rets = t_rets[indices]

                # Forward Pass (RecurrentPPO Policy)
                flat_obs = {k: v.flatten(0, 1) for k, v in b_obs.items()}
                features = self.model.policy.extract_features(flat_obs)

                if self.model.policy.share_features_extractor:
                    pi_features = vf_features = features
                else:
                    pi_features, vf_features = features

                pi_features = pi_features.view(current_bs, seq_len, -1)
                vf_features = vf_features.view(current_bs, seq_len, -1)

                lstm_states = self.model.policy.get_initial_states(current_bs)
                # episode_starts = torch.zeros((current_bs, seq_len), dtype=torch.bool).to(self.device)
                episode_starts = torch.zeros((current_bs, seq_len), dtype=torch.float32).to(self.device)
                # episode_starts[:, 0] = True
                episode_starts[:, 0] = 1.0

                latent_pi, _ = self.model.policy._process_sequence(
                    pi_features, lstm_states[0], episode_starts, self.model.policy.lstm_actor
                )

                if self.model.policy.lstm_critic:
                    latent_vf, _ = self.model.policy._process_sequence(
                        vf_features, lstm_states[1], episode_starts, self.model.policy.lstm_critic
                    )
                else:
                    latent_vf = vf_features

                # latent_pi = latent_pi.flatten(0, 1)
                # latent_vf = latent_vf.flatten(0, 1)

                lstm_hidden = self.model.policy.lstm_actor.hidden_size
                latent_pi = latent_pi.reshape(-1, lstm_hidden)

                if self.model.policy.lstm_critic:
                    lstm_hidden_vf = self.model.policy.lstm_critic.hidden_size
                    latent_vf = latent_vf.reshape(-1, lstm_hidden_vf)
                else:
                    latent_vf = latent_vf.reshape(-1, latent_vf.shape[-1])

                latent_pi = self.model.policy.mlp_extractor.forward_actor(latent_pi)
                latent_vf = self.model.policy.mlp_extractor.forward_critic(latent_vf)

                action_logits = self.model.policy.action_net(latent_pi)
                values = self.model.policy.value_net(latent_vf)

                flat_rets = b_rets.flatten(0, 1)
                critic_loss = F.mse_loss(values, flat_rets)

                flat_acts = b_acts.flatten(0, 1)
                actor_loss = 0
                start_idx = 0
                for dim_i, dim_size in enumerate(self.action_dims):
                    logits_slice = action_logits[:, start_idx: start_idx + dim_size]
                    target_slice = flat_acts[:, dim_i]
                    actor_loss += F.cross_entropy(logits_slice, target_slice)
                    start_idx += dim_size

                total_loss = actor_loss + self.config['val_coef'] * critic_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
                optimizer.step()

                epoch_a_loss += actor_loss.item()
                epoch_c_loss += critic_loss.item()

            print(
                f"  Epoch {epoch + 1:02d} | Actor Loss: {epoch_a_loss / num_seqs:.4f} | Critic Loss: {epoch_c_loss / num_seqs:.4f}")

        # 返回 state_dict 供后续加载
        return self.model.policy.state_dict()


# ==============================================================================
# 4. 主程序 (Main Logic)
# ==============================================================================

def main():
    conf = HYPERPARAMS

    # 路径设置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(conf["paths"]["save_dir"], f"{conf['run_name']}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    conf["pretrain"]["save_dir"] = run_dir

    # 获取设备设置
    device_pre = conf["devices"]["pretrain"]
    device_ppo = conf["devices"]["ppo"]

    print(f"设备分配: 预训练 -> {device_pre}, PPO -> {device_ppo}")

    # 环境初始化
    env = DummyVecEnv([lambda: make_env(conf)])

    # Policy 参数 (两个阶段共用)
    policy_kwargs = dict(
        features_extractor_class=ReferenceNetExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        lstm_hidden_size=conf["ppo"]["lstm_hidden_size"],
        n_lstm_layers=conf["ppo"]["n_lstm_layers"],
        shared_lstm=conf["ppo"]["shared_lstm"],
        enable_critic_lstm=conf["ppo"]["enable_critic_lstm"],
        net_arch=dict(pi=[256], vf=[256]),
    )

    pretrained_weights = None

    # ==========================================
    # 阶段 1: 预训练 (在 Device A 上)
    # ==========================================
    if conf["pretrain"]["enabled"] and not conf["paths"]["resume_from"]:
        print(f"\n=== Phase 1: Expert Pretraining on {device_pre} ===")

        # 1. 在 device_pre 上初始化模型
        model_pre = RecurrentPPO(
            RecurrentMaskableExpertPolicy,
            env,
            policy_kwargs=policy_kwargs,
            device=device_pre,
            verbose=1
        )
        if hasattr(model_pre.policy, "set_env_interface"):
            model_pre.policy.set_env_interface(unwrap_to_base(env.envs[0]))

        # 2. 训练
        pretrainer = ExpertPretrainer(model_pre, env, conf, device_pre)
        pretrained_weights = pretrainer.train()  # 获取权重字典

        # 3. 保存临时文件 (可选，防止中间断电)
        temp_path = os.path.join(run_dir, "temp_bc_weights.pth")
        torch.save(pretrained_weights, temp_path)
        print(f"预训练完成，权重已缓存。")

        # 4. 销毁旧模型，释放显存
        del model_pre
        del pretrainer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"已释放 {device_pre} 上的模型资源。")

    # ==========================================
    # 阶段 2: PPO 训练 (在 Device B 上)
    # ==========================================
    print(f"\n=== Phase 2: PPO Reinforcement Learning on {device_ppo} ===")

    # 1. 在 device_ppo 上初始化新模型
    # 如果指定了 resume_from，直接加载，忽略预训练权重
    if conf["paths"]["resume_from"]:
        print(f"正在从检查点恢复: {conf['paths']['resume_from']}")
        model_ppo = RecurrentPPO.load(conf["paths"]["resume_from"], env=env, device=device_ppo)
    else:
        # 全新初始化
        model_ppo = RecurrentPPO(
            RecurrentMaskableExpertPolicy,
            env,
            verbose=1,
            learning_rate=conf["ppo"]["learning_rate"],
            n_steps=conf["ppo"]["n_steps"],
            batch_size=conf["ppo"]["batch_size"],
            n_epochs=conf["ppo"]["n_epochs"],
            gamma=conf["ppo"]["gamma"],
            gae_lambda=conf["ppo"]["gae_lambda"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=conf["paths"]["log_dir"],
            device=device_ppo  # <--- 使用 PPO 专用设备
        )

        # 如果有预训练权重，进行加载
        if pretrained_weights is not None:
            print("正在加载预训练权重到 PPO 模型...")
            # load_state_dict 会自动处理 tensor 的 device 迁移
            model_ppo.policy.load_state_dict(pretrained_weights)

            # 设置初始 expert epsilon
            if hasattr(model_ppo.policy, "expert_epsilon"):
                model_ppo.policy.expert_epsilon = conf["expert_guide"]["initial_epsilon"]
            print("权重加载完成。")

    # 2. 确保环境接口注入 (因为是新实例化的模型)
    if hasattr(model_ppo.policy, "set_env_interface"):
        model_ppo.policy.set_env_interface(unwrap_to_base(env.envs[0]))

    # 3. 回调与训练
    callbacks = CallbackList([
        CheckpointCallback(save_freq=10000, save_path=run_dir, name_prefix="ppo_hk"),
        ExpertEpsilonDecayCallback(
            initial_eps=conf["expert_guide"]["initial_epsilon"],
            final_eps=conf["expert_guide"]["final_epsilon"],
            decay_steps=conf["expert_guide"]["decay_steps"],
            verbose=1
        )
    ])

    model_ppo.learn(
        total_timesteps=conf["ppo"]["total_timesteps"],
        callback=callbacks,
        tb_log_name=conf["run_name"],
        reset_num_timesteps=False
    )

    model_ppo.save(os.path.join(run_dir, "final_model"))
    env.close()
    print("All Training Finished.")


if __name__ == "__main__":
    main()