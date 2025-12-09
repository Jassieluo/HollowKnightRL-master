import math
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Any
from .HollowKnightEnvBase import HKController, Actions, DoneState, Scenes, GameState, ActionKeys, ActionType, CollisionUtils


class HollowKnightEnv(gym.Env):
    """
    空洞骑士强化学习环境 - 支持自定义分辨率
    """

    def __init__(self, host='127.0.0.1', port=5555, scene=Scenes.HORNET_1,
                 device='cpu', image_resolution=(120, 160), mask_resolution=(64, 64), use_grayscale=True, use_visual_attention=False):
        super().__init__()

        self.controller = HKController(host, port)
        self.scene = scene
        self.device = device
        self.image_resolution = image_resolution  # (height, width)
        self.mask_resolution = mask_resolution
        self.use_grayscale = use_grayscale
        self.use_visual_attention = use_visual_attention
        self.visualize = False

        self.skill_soul_cost = 33.

        self.mask_expansion_config = {
            'hero': 0.05,          # 稍微放大以便看清动作
            'enemies': 0.05,       # 包含Boss
            'hero_attacks': 0.00,  # 攻击判定
            'enemy_attacks': 0.00, # 敌人弹幕
            # 'terrain': 0.0,        # 地形原样保留
            'destructibles': 0.00,
            'traps': 0.00
        }

        self.frames_since_last_damage = 0

        print(f"初始化环境，图像分辨率: {image_resolution}")

        self.initialized = False

        # 动作空间: 14个离散动作
        # self.action_space = spaces.Discrete(15)
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 2, 2])
        self.action_nvec = [3, 3, 2, 2, 2, 2]

        self.channels = 1 if self.use_grayscale else 3

        self.mask_channels = 7

        self.mask_categories = [
            'terrain',
            'hero',
            'enemies',
            'hero_attacks',
            'enemy_attacks',
            'destructibles',
            'traps'
        ]

        # === 构建观察空间 ===
        obs_spaces = {
            'hero_state': spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
            'boss_state': spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            'lidar': spaces.Box(low=0, high=5.0, shape=(8,), dtype=np.float32),
            'image': spaces.Box(low=0, high=255,
                                shape=(image_resolution[0], image_resolution[1], self.channels),
                                dtype=np.uint8),
            'action_masks' : spaces.Box(low=0, high=1, shape=(14,), dtype=np.int32)
        }

        # [新增] 为每个掩码类别添加独立的观测空间
        # 我们使用 (H, W, 1) 的形状，方便 CNN 处理，但不合并
        for cat in self.mask_categories:
            key_name = f"mask_{cat}"  # 例如 mask_terrain
            obs_spaces[key_name] = spaces.Box(
                low=0, high=1,
                shape=(mask_resolution[0], mask_resolution[1], 1),  # 单通道
                dtype=np.uint8
            )

        self.observation_space = spaces.Dict(obs_spaces)

        # 暂时无用
        self.reward_params = {
            'damage_to_boss': 1.0,  # 提高造成伤害的权重
            'damage_taken': -2.0,   # 提高受伤惩罚
            'boss_kill': 50.0,
            'death_penalty': -10.0,
            'survival_reward': 0.001,
        }

        self.current_step = 0
        self.max_steps = 10000
        self.last_hero_hp = 5
        self.last_boss_hp = 0
        self.last_soul = 0

        self.pre_state = None

    def _get_action_mask(self, state: GameState):
        """生成动作掩码
            0-5 move
            6-13 jump act skill
        """
        action_mask = np.array(
            [1, 1, 1, # 不动 左 右
                 1, 1, 1, # 不动 上 下
                 1, 1,  # 不动 跳
                 1, 1, # 不动 攻击
                 1, 1, # 不动 Dash
                 1, 1 # 不动 Skill
             ], dtype=bool)

        if state is None:
            return action_mask

        # 规则1: 魂量不足时禁用技能
        if state.hero.soul < self.skill_soul_cost:
            action_mask[13] = 0

        # 规则2: Boss不在上方时禁用向上
        if state.boss.pos[1] <= (state.hero.pos[1] + 0.1):
            action_mask[4] = 0

        if not state.hero.can_dash:
            action_mask[11] = 0

        return action_mask

    def env_initialize(self, load_delay=2.25, visualize=False):
        self.controller.set_mode_training()
        self.controller.set_scene(self.scene)
        self.controller.set_response_mode(True)
        self.controller.set_hard_mode(False)
        self.controller.set_load_delay(load_delay)
        self.controller.set_fps(25)
        self.controller.set_frame_skip(2)
        self.controller.set_timescale(1.)

        if visualize:
            self.visualize = True
            cv2.namedWindow("Hollow Knight Env", cv2.WINDOW_NORMAL)

    def reset(self, seed=None, options=None, visualize=True):
        """重置环境"""
        # super().reset(seed=seed)

        self.frames_since_last_damage = 0

        if not self.controller.connected:
            if not self.controller.connect():
                raise ConnectionError("无法连接到空洞骑士Mod")

        if not self.initialized:
            self.env_initialize(load_delay=2.25, visualize=visualize)
            self.initialized = True
        elif self.controller.hard_mode:
            # time.sleep(1.2)
            pass
        else:
           self.controller.reset_game()

        state, img = self._wait_for_hero_ready()

        self.current_step = 0
        self.last_hero_hp = state.hero.hp if state else 5
        self.last_boss_hp = state.boss.hp if state and state.boss.exists else 0
        self.last_soul = state.hero.soul if state else 0

        obs = self._get_observation(state, img)
        info = self._get_info(state)

        self.pre_state = state

        return obs, info

    def step(self, action):
        """
        action: numpy array, e.g., [1, 0, 1, 0, 0, 0] (Left, None, Jump, No Attack...)
        """
        # === MultiDiscrete -> Bitmask 转换逻辑 ===
        # 将 [3, 3, 2, 2, 2, 2] 的动作转为 Mod 能识别的位掩码
        action_mask = 0

        # Horizontal: 0=None, 1=Left, 2=Right
        if action[0] == 1:
            action_mask |= ActionKeys.LEFT
        elif action[0] == 2:
            action_mask |= ActionKeys.RIGHT

        # Vertical: 0=None, 1=Up, 2=Down
        if action[1] == 1:
            action_mask |= ActionKeys.UP
        elif action[1] == 2:
            action_mask |= ActionKeys.DOWN

        # Actions: 0=No, 1=Yes
        if action[2] == 1: action_mask |= ActionKeys.JUMP
        if action[3] == 1: action_mask |= ActionKeys.ATTACK
        if action[4] == 1: action_mask |= ActionKeys.DASH
        if action[5] == 1: action_mask |= ActionKeys.SPELL

        # 发送动作 (注意：这里直接传 mask 整数)
        state, img = self.controller.step(action_mask, action_type=ActionType.MULTI_BINARY, visualize=True)

        if state is None:
            # 异常处理...
            obs = self._get_default_observation()
            return obs, 0, True, True, {}

        # 计算奖励
        reward, terminated = self._calculate_reward(state, self.pre_state, action)
        self.pre_state = state

        truncated = self.current_step >= self.max_steps
        obs = self._get_observation(state, img)
        info = self._get_info(state)

        if self.visualize:
            cv2.imshow("Hollow Knight Env", obs['image'])
            cv2.waitKey(1)

        return obs, reward, terminated, truncated, info

    def _wait_for_hero_ready(self):
        """
        阻塞等待，直到 Mod 返回 ready=true
        """
        print("Waiting for Reset...", end="", flush=True)
        max_retries = 200  # 5秒超时

        for _ in range(max_retries):
            # 发送 NO_OP 动作，获取当前状态
            # 注意：Mod 即使在 isBusy 状态，也应该能返回 JSON（哪怕是旧的或者空的），
            # 只要包含 ready 字段即可。
            state, img = self.controller.step(Actions.NO_OP, action_type=ActionType.MULTI_BINARY)

            if state is None:
                time.sleep(0.02)
                continue

            # [NEW] 检查 Mod 回传的 ready 标志
            # 我们在 C# 的 GetCurrentStateJSON 里加了 "ready" 字段
            # 同时也检查一下 HP，双重保险
            is_ready = getattr(state, 'ready', True)  # 假设你在 GameState 解析里加了这个字段

            # 如果没有解析到 ready 字段，回退到旧逻辑 (hp > 0 and not paused)
            if not hasattr(state, 'ready'):
                is_ready = state.hero.hp > 0 and not state.paused

            # is_ready = state.hero.hp > 0 and not state.paused

            if is_ready:
                # 额外等待几帧以确保渲染稳定（可选）
                # time.sleep(0.1)

                # 重新获取一帧干净的 Observation 作为 s_0
                state, img = self.controller.step(Actions.NO_OP, action_type=ActionType.MULTI_BINARY)
                return state, img

            # 如果不 ready，休眠一小会儿，避免疯狂发包占满带宽
            # time.sleep(0.05)

        raise TimeoutError("Reset timeout: Game did not become ready.")

    # def _calculate_reward(self, state: GameState, pre_state: GameState, action) -> Tuple[float, bool]:
    #     """
    #     [对齐版] 严格复刻参考代码的奖励函数
    #     核心逻辑：
    #     1. 伤害权重: 造成伤害给予正反馈，受到伤害给予巨额负反馈 (11倍)。
    #     2. 空间逻辑: 强制 Agent 保持在"甜点位" (2.5 - 4.8 单位距离)。
    #     3. 移动逻辑: 距离太近时鼓励远离，距离太远时鼓励靠近。
    #     """
    #     reward = 0.0
    #     done = False
    #
    #     # 0. 空值与结束检查
    #     if pre_state is None or state is None:
    #         return 0.0, False
    #
    #     hero = state.hero
    #     boss = state.boss
    #     prev_hero = pre_state.hero
    #     prev_boss = pre_state.boss
    #
    #     # 基础数据计算
    #     # 注意：参考代码主要使用 X 轴距离 (abs(player_x - hornet_x))
    #     # 但为了更鲁棒，建议使用欧几里得距离，这里为了对齐逻辑，我们主要参考 X 轴，但在判定距离时用欧氏距离会更准
    #     # 这里折中：计算欧氏距离 dist 用于区间判定，计算 dx 用于方向判定
    #     dx_val = hero.pos[0] - boss.pos[0]
    #     dist = math.sqrt(dx_val ** 2 + (hero.pos[1] - boss.pos[1]) ** 2)
    #
    #     # 动作解析 (MultiDiscrete -> 意图)
    #     # action[0]: 0=None, 1=Left, 2=Right
    #     move_action = action[0]
    #     # action[5]: 1=Cast
    #     is_cast_action = (action[5] == 1)
    #
    #     # =======================================================
    #     # 1. 核心生存与伤害 (HP Rewards)
    #     # =======================================================
    #
    #     # [Self HP]: 受到伤害给予巨额惩罚 (参考代码系数 11)
    #     # 参考: 11 * (next - curr) -> 也就是 11 * 负数
    #     hp_change = hero.hp - prev_hero.hp
    #     if hp_change < 0:
    #         reward += 11.0 * hp_change  # 例如掉1血 -> -11分
    #
    #     # [Boss HP]: 造成伤害给予奖励 (参考代码系数 1/8)
    #     # 参考: (prev - next) / 8
    #     boss_dmg = prev_boss.hp - boss.hp
    #     if boss_dmg > 0:
    #         reward += boss_dmg / 8.0  # 例如打20血 -> +2.5分
    #
    #         # [法术奖励]: 参考代码鼓励用法术攻击
    #         # if prev["boss_hp"] - curr["boss_hp"] > 0 and attack in [5,6]: reward += 2
    #         if is_cast_action:
    #             reward += 2.0
    #
    #     # =======================================================
    #     # 2. 方向奖励 (Direction Reward)
    #     # =======================================================
    #     # 参考代码逻辑: dire * s * dis * base(3)
    #     # 解析：
    #     # - 如果距离 < 2.5 (太近): 只有"远离Boss"的移动给正分，"靠近"给负分
    #     # - 如果距离 >= 2.5 (安全): 只有"靠近Boss"的移动给正分，"远离"给负分
    #
    #     base_dir_reward = 3.0
    #
    #     # 判定当前是否处于"危险距离" (太近)
    #     is_too_close = (dist < 2.5)
    #
    #     # 判定玩家相对于 Boss 的方位 (s)
    #     # dx_val > 0: 玩家在右 (需要向右逃，向左追)
    #     # dx_val < 0: 玩家在左 (需要向左逃，向右追)
    #     player_on_right = (dx_val > 0)
    #
    #     # 判定移动意图 (dire)
    #     is_moving_left = (move_action == 1)
    #     is_moving_right = (move_action == 2)
    #
    #     dir_reward = 0.0
    #
    #     if is_moving_left or is_moving_right:
    #         # 逻辑拆解：
    #         if is_too_close:
    #             # === 危险距离：鼓励远离 ===
    #             if player_on_right:
    #                 # 在右边，应该向右跑
    #                 if is_moving_right:
    #                     dir_reward = base_dir_reward
    #                 else:
    #                     dir_reward = -base_dir_reward
    #             else:
    #                 # 在左边，应该向左跑
    #                 if is_moving_left:
    #                     dir_reward = base_dir_reward
    #                 else:
    #                     dir_reward = -base_dir_reward
    #         else:
    #             # === 安全距离：鼓励靠近 ===
    #             if player_on_right:
    #                 # 在右边，应该向左追
    #                 if is_moving_left:
    #                     dir_reward = base_dir_reward
    #                 else:
    #                     dir_reward = -base_dir_reward
    #             else:
    #                 # 在左边，应该向右追
    #                 if is_moving_right:
    #                     dir_reward = base_dir_reward
    #                 else:
    #                     dir_reward = -base_dir_reward
    #
    #     reward += dir_reward
    #
    #     # =======================================================
    #     # 3. 距离保持奖励 (Distance Reward / Sweet Spot)
    #     # =======================================================
    #     # 参考代码逻辑：
    #     # < 2.5: -3 (太近)
    #     # 2.5 ~ 4.8: +3 (甜点位)
    #     # > 4.8:
    #     #    如果是移动操作(move < 2): +2 (鼓励动起来追)
    #     #    如果站着不动: -2 (参考代码 else return -2) -> 其实是鼓励远距离要跑动
    #
    #     dist_reward = 0.0
    #
    #     if dist < 1.5:
    #         dist_reward = -3.0
    #     elif dist < 4.8:
    #         dist_reward = 3.0
    #     else:
    #         # 距离远的时候
    #         if move_action in [1, 2]:  # 正在移动
    #             dist_reward = 2.0
    #         else:
    #             dist_reward = -2.0  # 站着发呆惩罚
    #
    #     reward += dist_reward
    #
    #     # =======================================================
    #     # 4. 特殊技能/状态修正 (Skill & Judge)
    #     # =======================================================
    #     # 参考代码: act_skill_reward
    #     # 如果 Boss 放技能1 (空中悬停技能)，且玩家跳跃，给予惩罚 (防止撞技能)
    #     # 我们用简单的 Y 轴判定来模拟 Boss 技能状态
    #     is_boss_using_skill = (boss.pos[1] > 24.0)  # 假设的技能高度阈值
    #
    #     # if is_boss_using_skill:
    #     #     # 玩家正在跳 (action[2] == 1)
    #     #     if action[2] == 1:
    #     #         reward -= 3.0
    #
    #     # 参考代码: act_distance_reward
    #     # 远距离乱放攻击惩罚
    #     if dist > 12.0:
    #         # 如果按了攻击键
    #         if action[3] == 1:
    #             reward -= 3.0
    #         # 如果只是站着 (action 0)
    #         elif move_action == 0:
    #             reward += 1.0
    #
    #     if action[4] == 1:
    #         reward -= 1.0
    #
    #     # =======================================================
    #     # 5. 终局状态 (Terminal States)
    #     # =======================================================
    #     if state.done == DoneState.VICTORY:
    #         # 参考: 50 + 10 * hp
    #         reward += 50.0 + 10.0 * hero.hp
    #         done = True
    #         print(f"=== VICTORY! Reward: {reward:.2f} ===")
    #
    #     elif state.done == DoneState.DEAD:
    #         # 参考: -(20 + boss_hp/10)
    #         reward -= (20.0 + boss.hp / 10.0)
    #         done = True
    #         print(f"--- DIED. Reward: {reward:.2f} ---")
    #
    #     elif self.current_step >= self.max_steps:
    #         # 超时截断
    #         done = True
    #
    #     return reward, done

    def _calculate_reward(self, state: GameState, pre_state: GameState, action) -> Tuple[float, bool]:
        """
        [终极版] 疯狗流 + 严厉教官 (Berserker & Drill Sergeant)
        1. 鼓励换血：攻击收益 > 受伤代价
        2. 禁止乱按：没蓝放法、CD没好冲刺、瞎闪避都会扣分
        3. 空间引导：强迫贴脸
        """
        reward = 0.0
        done = False

        if pre_state is None or state is None:
            return 0.0, False

        # --- 1. 解包数据 ---
        hero = state.hero
        prev_hero = pre_state.hero
        boss = state.boss
        prev_boss = pre_state.boss

        # 动作解析 (MultiDiscrete: [Hor, Ver, Jump, Attack, Dash, Spell])
        # action[4] == 1 是冲刺, action[5] == 1 是法术
        act_dash = (action[4] == 1)
        act_spell = (action[5] == 1)

        # 基础距离
        dx = hero.pos[0] - boss.pos[0]
        dy = hero.pos[1] - boss.pos[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # =======================================================
        # 2. “严厉教官”惩罚 (Anti-Spam Logic) - 必须放在最前面
        # =======================================================

        # [A. 闪避惩罚]
        if act_dash:
            # 1. 基础过路费 (Cost of Living)
            # 只要按了闪避，就扣一点点。这会教它：除非是为了躲那个 -0.5 的伤害，否则别乱动。
            reward -= 0.02

            # 2. 违规操作：CD没好还要硬按
            # 这是极其愚蠢的行为，必须重罚，让它记住 CD 的概念。
            if not prev_hero.can_dash:
                reward -= 0.1

                # 3. 懦夫行为：离得老远还往外润
            # 如果距离 > 6.0 且 正在远离 Boss，还按冲刺，判定为消极避战。
            prev_dist = math.sqrt(
                (prev_hero.pos[0] - prev_boss.pos[0]) ** 2 + (prev_hero.pos[1] - prev_boss.pos[1]) ** 2)
            is_moving_away = dist > prev_dist
            if dist > 6.0 and is_moving_away:
                reward -= 0.1

        # [B. 法术惩罚]
        if act_spell:
            # 1. 违规操作：没蓝还要硬放
            # 空洞骑士里没蓝按法术会有一个“放屁”的硬直动作，非常致命。
            if prev_hero.soul < 33:
                reward -= 0.08  # 罚重一点，因为这个硬直很危险

            # (可选) 如果你想更严格：距离太远不要空放法术 (针对白波)
            # if dist > 10.0: reward -= 0.1

        # =======================================================
        # 3. “疯狗流”经济系统 (Berserker Economy)
        # =======================================================

        # [C. 伤害收益 - 核心驱动力]
        # 假设平A一下 15血。
        boss_dmg = prev_boss.hp - boss.hp
        if boss_dmg > 0:
            # 伤害分：15 / 10 = +1.5分
            reward += boss_dmg / 10.0
            # 命中分：只要打中就是好样的
            reward += 0.5
            # 只要造成伤害，重置“消极比赛”计时器
            self.frames_since_last_hit = 0
        else:
            self.frames_since_last_hit = getattr(self, 'frames_since_last_hit', 0) + 1

        # [D. 受伤代价 - 鼓励换血]
        # 掉1血扣 0.5。
        # 逻辑：只要我砍中一刀(+2.0)，哪怕被打两下(-1.0)，我还是赚的。
        hero_dmg = prev_hero.hp - hero.hp
        if hero_dmg > 0:
            reward -= 0.5 * hero_dmg

        # =======================================================
        # 4. 空间与时间引导
        # =======================================================

        # [E. 距离控制]
        # 简单粗暴：不许离 Boss 太远。
        if dist > 6.0:
            reward -= 0.05  # 只要太远，每帧扣分，逼它靠近

        # [F. 面向纠正]
        # 没打中往往是因为背对 Boss。
        boss_on_right = (boss.pos[0] > hero.pos[0])
        hero_face_right = (hero.facing == 1)
        if boss_on_right != hero_face_right:
            reward -= 0.01

        # [G. 甚至更狠的：消极比赛判负]
        # 如果 8 秒 (约 240 帧) 没摸到 Boss，直接扣大分。
        if self.frames_since_last_hit > 20:
            reward -= 0.1  # 每帧 -0.1 非常痛，很快就会把累计奖励扣光

        if hero.lidar[0] <= 1.0 or hero.lidar[1] <= 1.0:
            reward -= 0.5
            # print("close to wall")

        # =======================================================
        # 5. 结算
        # =======================================================

        # 手动判定胜利 (为了解决 Mod 延迟)
        if boss.hp <= 0:
            reward += 5.0  # 给个小彩头
            done = True
            print("VICTORY (Python Logic)")

        elif state.done == DoneState.DEAD:
            reward -= 5.0
            done = True
            print("DEAD")

        elif self.current_step >= self.max_steps:
            done = True

        # 截断
        reward = max(-5.0, min(5.0, reward))
        return reward, done

    def _get_observation(self, state: GameState, img: np.ndarray) -> Dict:
        """构建观察值"""
        if state is None:
            return self._get_default_observation()

        hero = state.hero
        boss = state.boss

        # === [修改] 掩码处理 ===
        # 1. 使用 mask_resolution 获取掩码字典
        mh, mw = self.mask_resolution
        masks_dict = CollisionUtils.get_all_masks(state.colliders, width=mw, height=mh, categories_in=self.mask_categories)

        # 2. 准备基础观测字典
        obs_dict = {}

        # 3. 填充各个掩码到字典中 (不堆叠)
        for cat in self.mask_categories:
            # 原始 mask 是 (H, W)
            mask_2d = masks_dict.get(cat, np.zeros((mh, mw), dtype=np.uint8))

            # 扩展为 (H, W, 1) 以符合 Gym Box 定义
            mask_3d = np.expand_dims(mask_2d, axis=-1)

            obs_dict[f"mask_{cat}"] = mask_3d

        # === 填充其他数据 ===

        # 主角状态
        obs_dict['hero_state'] = np.array([
            hero.hp, hero.max_hp, hero.soul,
            hero.pos[0], hero.pos[1],
            hero.vel[0], hero.vel[1],
            hero.facing,
            float(hero.on_ground), float(hero.on_wall),
            float(hero.can_dash),
            hero.shadow_timer, float(hero.has_shadow_dash),
            float(hero.is_recoiling), float(hero.can_double_jump),
            float(hero.is_invincible), float(hero.is_healing),
            hero.nail_damage,
            float(boss.exists),
            self.current_step / self.max_steps
        ], dtype=np.float32)

        # Boss状态
        obs_dict['boss_state'] = np.array([
            float(boss.exists), boss.hp,
            boss.pos[0], boss.pos[1],
            boss.vel[0], boss.vel[1],
            boss.rel_pos[0], boss.rel_pos[1]
        ], dtype=np.float32)

        # 雷达
        obs_dict['lidar'] = np.array(hero.lidar, dtype=np.float32)

        # [修改] 图像处理部分，传入 state.colliders
        ih, iw = self.image_resolution

        # 无论开关是否开启，都传入 colliders，具体是否使用由 _process_image 内部判断
        processed_img = self._process_image(img, state.colliders) if img is not None else np.zeros(
            (ih, iw, self.channels), dtype=np.uint8)

        obs_dict['image'] = processed_img
        obs_dict['action_masks'] = self._get_action_mask(state)

        return obs_dict

    def _process_image(self, img: np.ndarray, colliders: Optional[Any] = None) -> np.ndarray:
        """
        处理图像：调整大小 -> [可选:视觉注意力掩码] -> 翻转 -> 颜色转换
        """
        target_h, target_w = self.image_resolution

        # 1. 空数据兜底
        if img is None:
            c = 1 if self.use_grayscale else 3
            return np.zeros((target_h, target_w, c), dtype=np.uint8)

        # 2. 调整大小 (Resize)
        # 先Resize再处理Mask性能最好
        img_resized = cv2.resize(img, (target_w, target_h))

        # === [新增] 3. 可选：应用视觉注意力掩码 ===
        if self.use_visual_attention and colliders is not None:
            # 调用 BaseEnv 中定义的静态方法
            img_resized = CollisionUtils.apply_visual_mask(
                img_resized,
                colliders,
                self.mask_expansion_config
            )
        # ==========================================

        # 4. 颜色与通道处理 (保持原有逻辑)
        if self.use_grayscale:
            if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
                img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            else:
                img_processed = img_resized

            if len(img_processed.shape) == 2:
                img_processed = np.expand_dims(img_processed, axis=-1)
        else:
            if len(img_resized.shape) == 2:
                img_processed = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            elif len(img_resized.shape) == 3 and img_resized.shape[2] == 1:
                img_processed = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            else:
                img_processed = img_resized

            # 确保是 (H, W, 3)
            if len(img_processed.shape) == 2:
                # 这段逻辑通常不会触发，除非 resize 破坏了维度
                img_processed = np.expand_dims(img_processed, axis=-1)
                img_processed = np.repeat(img_processed, 3, axis=-1)

        return img_processed

    def _get_default_observation(self) -> Dict:
        """获取默认观察值"""
        ih, iw = self.image_resolution
        mh, mw = self.mask_resolution

        obs_dict = {
            'hero_state': np.zeros(20, dtype=np.float32),
            'boss_state': np.zeros(8, dtype=np.float32),
            'lidar': np.ones(8, dtype=np.float32) * 5,
            'image': np.zeros((ih, iw, self.channels), dtype=np.uint8),
            'action_masks': self._get_action_mask(state=None),
        }

        # [新增] 为每个掩码类别填充全0矩阵
        for cat in self.mask_categories:
            obs_dict[f"mask_{cat}"] = np.zeros((mh, mw, 1), dtype=np.uint8)

        return obs_dict

    def _get_info(self, state: GameState) -> Dict:
        """获取环境信息"""
        if state is None:
            return {}

        return {
            'hero_hp': state.hero.hp,
            'boss_hp': state.boss.hp if state.boss.exists else 0,
            'step': self.current_step,
            'done_state': state.done
        }

    def render(self):
        """渲染环境"""
        pass

    def close(self):
        """关闭环境"""
        self.controller.close()

    def update_reward_params(self, new_params: Dict):
        """动态更新奖励参数"""
        for key, value in new_params.items():
            if key in self.reward_params:
                old_value = self.reward_params[key]
                self.reward_params[key] = value
                print(f"更新奖励参数 {key}: {old_value} -> {value}")