import gymnasium as gym
import numpy as np
import random


class ExpertGuidedWrapper(gym.Wrapper):
    """
    [严厉复刻版] 专家规则引导
    完全照搬成功案例的 Agent.py 逻辑，强制纠正 AI 的乱动行为。
    """

    def __init__(self, env, epsilon=0.9, epsilon_decay=0.99998, min_epsilon=0.1):
        super().__init__(env)
        # 初始 epsilon 设为 0.9，意味着 90% 的时间听专家的，别让 AI 乱闪避
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # 动作索引映射 [Hor, Ver, Jump, Attack, Dash, Spell]
        self.IDX_HOR = 0
        self.IDX_VER = 1
        self.IDX_JUMP = 2
        self.IDX_ATTACK = 3
        self.IDX_DASH = 4
        self.IDX_SPELL = 5

    def step(self, action):
        # 获取当前帧的真实状态
        # 必须确保 Env 返回了 pre_state (我们在 HollowKnightEnv_sb3 中定义了)
        obs_state = self.env.unwrapped.pre_state

        # 如果处于训练初期，或者随机到了 epsilon，强制使用专家逻辑
        if obs_state and random.random() < self.epsilon:
            # 获取专家建议的动作 (覆盖原动作)
            final_action = self._get_hard_coded_action(obs_state, action)
        else:
            # 即使是 AI 自己玩，也要做基本的合法性过滤 (比如没蓝别放波)
            final_action = self._filter_ai_action(obs_state, action)

        # 衰减 epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return self.env.step(final_action)

    def _get_hard_coded_action(self, state, ai_action):
        """
        完全复刻 Agent.py 的 better_move 和 better_action 逻辑
        """
        hero = state.hero
        boss = state.boss

        # 复制一份，作为基础
        act = np.array(ai_action, copy=True)

        # === 0. 全局重置：默认不许乱冲、不许乱跳 ===
        # 成功案例里，大部分时候是走路和平A，只有特定情况才 Dash
        act[self.IDX_DASH] = 0
        act[self.IDX_JUMP] = 0
        act[self.IDX_SPELL] = 0

        if not boss.exists:
            return act

        # 计算距离
        dx = hero.pos[0] - boss.pos[0]
        dy = hero.pos[1] - boss.pos[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # 模拟 Horont_Skill1 (小姐姐放技能)：
        # 我们很难完美判断，但可以通过 Boss Y轴位置判断 (小姐姐在空中通常是放技能)
        hornet_skill1 = (boss.pos[1] > 24.0)  # 假设的阈值，根据地图调整，神居通常地面是y=14左右

        # ==========================================
        # Part 1: 移动逻辑 (对应 move_model)
        # ==========================================
        # 逻辑：
        # 1. 如果 Boss 放技能且距离近 (<6) -> 逃跑
        # 2. 如果距离太近 (<2.5) -> 逃跑
        # 3. 如果距离适中 (<5) -> 保持距离或微调
        # 4. 如果距离远 (>5) -> 追击

        move_cmd = 0  # 0:停, 1:左, 2:右

        if hornet_skill1 and dist < 6.0:
            # 逃跑
            move_cmd = 1 if dx > 0 else 2

        elif dist < 1.5:  # 太近了，拉开一点距离
            move_cmd = 1 if dx > 0 else 2

        elif dist > 5.0:  # 太远了，追
            move_cmd = 2 if dx > 0 else 1

        else:
            # 距离适中 (2.0 ~ 5.0)，可以停下来输出，或者根据AI意愿微调
            # 这里我们强制面向 Boss
            move_cmd = 0
            # 甚至可以强制不许动，方便输出

        act[self.IDX_HOR] = move_cmd

        # ==========================================
        # Part 2: 动作逻辑 (对应 act_model)
        # ==========================================
        # 逻辑：
        # 1. 贴脸 (<1.5) -> 可能是冲刺穿过，或者是无脑平A
        # 2. 中距离 (<5) -> 平A 或者 发波

        # 修正：成功案例里，Action 6 (Rush) 只在 dis < 1.5 或 特定技能时触发
        # 你的 AI 一直闪避，是因为没有这个限制。

        if hornet_skill1:
            if dist < 2.5:
                act[self.IDX_DASH] = 1  # 只有这时候才允许冲刺！
            else:
                act[self.IDX_ATTACK] = 0  # 远距离躲技能不贪刀

        # 普通状态
        else:
            if dist < 1.5:
                # 极近距离：要么冲刺穿身(换边)，要么拼命平A
                # 简单起见，我们强制平A，偶尔冲刺
                if random.random() < 0.2:
                    act[self.IDX_DASH] = 1
                else:
                    act[self.IDX_ATTACK] = 1

            elif dist < 4.5:
                # 最佳攻击距离：狂按攻击
                act[self.IDX_ATTACK] = 1
                # 这种距离严禁冲刺，容易撞身上
                act[self.IDX_DASH] = 0

                # 如果有蓝，且 Boss 在地上，可以放波 (Skill)
                if hero.soul >= 33 and not hornet_skill1 and random.random() < 0.3:
                    act[self.IDX_SPELL] = 1

            elif dist < 12.0:
                    if random.random() < 0.05:
                        act[self.IDX_JUMP] = 1

        return act

    def _filter_ai_action(self, state, action):
        """
        即使是 AI 自主决策阶段，也要过滤掉纯粹的傻瓜操作
        """
        hero = state.hero
        act = np.array(action, copy=True)

        # 1. 没蓝严禁按法术 (防止惩罚机制不够快)
        if hero.soul < 33:
            act[self.IDX_SPELL] = 0

        # 2. 只有静止时才能回血 (防止移动施法被打断浪费蓝)
        if act[self.IDX_SPELL] == 1 and (act[self.IDX_HOR] != 0 or act[self.IDX_JUMP] != 0):
            # 如果想放法术但正在动，优先停下来（或者改成白波）
            # 这里简单处理：如果想移动，就别放法术了
            act[self.IDX_SPELL] = 0

        # 3. 抑制冲刺频率
        # 如果 AI 总是倾向于把 Dash 置 1，我们这里加一个硬锁：
        # 如果距离 > 4.0，禁止冲刺 (除非你有很强的理由，但在初期通常是噪声)
        dx = hero.pos[0] - state.boss.pos[0]
        dy = hero.pos[1] - state.boss.pos[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        if dist > 4.0:
            act[self.IDX_DASH] = 0

        return act


import gymnasium as gym
import numpy as np
import random
import math


class SpatialExpertWrapper(gym.Wrapper):
    """
    [战术规则型] 空间感知专家系统
    核心理念：不再进行复杂的未来模拟，而是执行预设的"最优解"战术动作。
    """

    def __init__(self, env, epsilon=0.9, epsilon_decay=0.99998, min_epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # 动作索引
        self.IDX_HOR = 0
        self.IDX_VER = 1
        self.IDX_JUMP = 2
        self.IDX_ATTACK = 3
        self.IDX_DASH = 4
        self.IDX_SPELL = 5

        self.last_attack_pressed = False

        # 简单的记忆，用于计算速度方向
        self.prev_boss_pos = None
        self.prev_threat_pos = {}  # id -> pos

    def reset(self, **kwargs):
        self.last_attack_pressed = False
        self.prev_boss_pos = None
        self.prev_threat_pos = {}
        return self.env.reset(**kwargs)

    def step(self, action):
        state = self.env.unwrapped.pre_state

        if state and random.random() < self.epsilon:
            final_action = self._derive_tactical_action(state)
        else:
            final_action = self._filter_ai_action(state, action)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if state:
            self._update_memory(state)

        return self.env.step(final_action)

    def _derive_tactical_action(self, state):
        hero = state.hero
        boss = state.boss
        colliders = state.colliders

        act = np.zeros(6, dtype=int)

        # 基础数据
        dx = hero.pos[0] - boss.pos[0]
        dy = hero.pos[1] - boss.pos[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # 默认面向
        face_dir = 2 if dx < 0 else 1  # 1:Right, 2:Left
        away_dir = 1 if dx < 0 else 2

        act[self.IDX_HOR] = face_dir  # 默认时刻面向Boss，方便攻击

        # =========================================================
        # 第一步：威胁识别 (Threat Identification)
        # =========================================================
        threat_type = "NONE"
        threat_obj = None

        # 1. 检测 Boss 冲撞 (Charge)
        # 速度快，且朝向主角，且在地面(或低空)
        boss_vel_x = boss.vel[0]
        if abs(boss_vel_x) > 3.0 and (boss.pos[1] < hero.pos[1] + 2.0):
            # 判断方向是否朝向主角
            if (dx > 0 and boss_vel_x > 0) or (dx < 0 and boss_vel_x < 0):
                if dist < 8.0:
                    threat_type = "CHARGE"

        # 2. 检测 Boss 下砸 (Slam/Fall)
        # Boss 在主角上方，且垂直速度向下
        if threat_type == "NONE":
            if (boss.pos[1] > hero.pos[1] + 1.0) and (boss.vel[1] < -1.0) and (abs(dx) < 3.5):
                threat_type = "SLAM"

        # 3. 检测 飞行道具 (Projectile)
        if threat_type == "NONE":
            proj = self._find_incoming_projectile(hero, colliders)
            if proj:
                threat_type = "PROJECTILE"
                threat_obj = proj

        # 4. 检测 静态陷阱/身体接触 (Contact)
        if threat_type == "NONE":
            if dist < 2.0:  # 贴脸了，很危险
                threat_type = "CONTACT"

        # =========================================================
        # 第二步：闪避/防御决策 (Evasion Logic)
        # =========================================================
        is_dodging = False

        # --- 战术 A: 应对下砸 (绝对不能贪刀) ---
        if threat_type == "SLAM":
            is_dodging = True
            # 策略：向远离 Boss X轴 的方向冲刺
            act[self.IDX_DASH] = 1
            act[self.IDX_HOR] = away_dir

        # --- 战术 B: 应对水平冲撞/飞针 ---
        elif threat_type == "CHARGE" or threat_type == "PROJECTILE":
            is_dodging = True

            # 判断是否拥有黑冲 (Shadow Dash)
            can_shadow_dash = hero.has_shadow_dash and hero.shadow_timer <= 0

            # 如果是扁平判定的Bug攻击，必须跳
            is_flat_bug = (threat_obj and threat_obj.get('is_flat'))

            if can_shadow_dash and not is_flat_bug:
                # 策略 B1: 黑冲穿过 (最稳)
                # 迎着威胁冲
                act[self.IDX_DASH] = 1
                act[self.IDX_HOR] = face_dir
            else:
                # 策略 B2: 跳跃 + 下劈 (Pogo)
                act[self.IDX_JUMP] = 1
                # 如果已经跳起来了，或者威胁就在脚下，准备下劈
                # 稍微判断一下高度
                threat_y = threat_obj['pos'][1] if threat_obj else boss.pos[1]
                if hero.pos[1] > threat_y - 0.5:
                    act[self.IDX_ATTACK] = 1
                    act[self.IDX_VER] = 2

        # --- 战术 C: 贴脸防身 ---
        elif threat_type == "CONTACT":
            # 如果太近，优先拉开距离
            act[self.IDX_HOR] = away_dir
            if random.random() < 0.1: act[self.IDX_JUMP] = 1  # 偶尔小跳防卡死

        # --- 战术 D: 绝境下砸 (Dive) ---
        # "如果boss的攻击无法躲避且灵魂够就下砸"
        # 这里的"无法躲避"简化判定为：威胁极近且没有冲刺冷却
        if is_dodging and not hero.can_dash and dist < 2.5 and hero.soul >= 33:
            act[self.IDX_HOR] = 0
            act[self.IDX_VER] = 2
            act[self.IDX_SPELL] = 1
            return act  # 下砸优先级最高，直接覆盖所有动作

        # =========================================================
        # 第三步：攻击决策 (Attack Logic) - 按用户要求严格执行
        # =========================================================
        # 只有在没有全力闪避(比如冲刺中)的时候才攻击
        # 跳跃闪避时是可以攻击的

        if act[self.IDX_DASH] == 0 and act[self.IDX_SPELL] == 0:

            # 1. 释放波 (Vengeful Spirit)
            # "如果boss离得太远 且 在同一水平面上 且 灵魂足够"
            if hero.soul >= 33 and dist > 6.0 and abs(dy) < 2.0:
                act[self.IDX_VER] = 0
                act[self.IDX_SPELL] = 1

            # 2. 释放上吼 (Howling Wraiths)
            # "如果boss在正上方 且 灵魂足够 且 没有被攻击到的可能"
            # 没有被攻击到的可能 -> 这里简化为 threat_type == "NONE"
            elif hero.soul >= 33 and (boss.pos[1] > hero.pos[1] + 1.0) and (abs(dx) < 2.5) and (threat_type == "NONE"):
                act[self.IDX_VER] = 1
                act[self.IDX_SPELL] = 1

            # 3. 骨钉攻击 (Nail)
            else:
                ATTACK_RANGE = 4.2
                if dist < ATTACK_RANGE:
                    # 连发逻辑
                    if not self.last_attack_pressed:
                        act[self.IDX_ATTACK] = 1
                        self.last_attack_pressed = True

                        # 方向判定
                        if boss.pos[1] > hero.pos[1] + 0.5:
                            # Boss在上方 -> 上劈
                            act[self.IDX_VER] = 1
                        elif hero.pos[1] > boss.pos[1] + 1.0:
                            # Boss在下方 -> 下劈
                            act[self.IDX_VER] = 2
                        else:
                            # 侧劈
                            act[self.IDX_VER] = 0
                    else:
                        act[self.IDX_ATTACK] = 0
                        self.last_attack_pressed = False
                else:
                    self.last_attack_pressed = False

        # =========================================================
        # 第四步：走位逻辑 (Movement) - 辅助
        # =========================================================
        # 如果没有在闪避，且不需要攻击定身，调整位置

        if not is_dodging and act[self.IDX_SPELL] == 0:
            # 保持甜点位 (Sweet Spot)
            SAFE_DIST = 3.0

            if dist < SAFE_DIST:
                # 稍微太近了，后退
                act[self.IDX_HOR] = away_dir
            elif dist > 5.0:
                # 太远了，追击
                act[self.IDX_HOR] = face_dir
            else:
                # 距离合适，如果不攻击就停下，如果攻击会自动停下
                pass

        # 冲突清洗
        if act[self.IDX_SPELL] == 1: act[self.IDX_ATTACK] = 0
        if act[self.IDX_DASH] == 1: act[self.IDX_ATTACK] = 0  # 冲刺不能平A

        return act

    # =========================================================
    #  辅助函数
    # =========================================================

    def _find_incoming_projectile(self, hero, colliders):
        """寻找正在飞向主角的飞行道具"""
        threats = colliders.enemy_attacks + self._extract_bugged_attacks(colliders.hero_attacks)

        for shape in threats:
            cx, cy = self._get_centroid(shape)

            # 计算简单的相对速度
            # 如果没有历史数据，默认威胁存在
            # 这里简化：只要距离近且位于运动方向上

            # 估算世界坐标 (假设 grid scale 30)
            wx = (cx - 0.5) * 30.0 + hero.pos[0]
            wy = (cy - 0.5) * 18.0 + hero.pos[1]

            dist = math.sqrt((wx - hero.pos[0]) ** 2 + (wy - hero.pos[1]) ** 2)

            if dist < 6.0:
                # 检查速度方向
                # 我们需要通过ID或者位置追踪来计算速度，这里使用 _get_velocity 简化版
                vx, vy = self._get_velocity(shape, (wx, wy))

                # 判定：距离很近，或者 速度朝向主角
                is_incoming = False
                if abs(vx) > 0.1:
                    if (wx > hero.pos[0] and vx < 0) or (wx < hero.pos[0] and vx > 0):
                        is_incoming = True

                if is_incoming or dist < 2.5:
                    return {'pos': (wx, wy), 'vel': (vx, vy), 'is_flat': (shape in colliders.hero_attacks)}
        return None

    def _get_velocity(self, shape, curr_pos):
        # 简单的一帧差分法，需要 shape 有唯一标识，或者暴力最近邻
        # 这里用最近邻
        if not self.prev_threat_pos: return 0, 0

        min_d = 100.0
        best_prev = None
        for p in self.prev_threat_pos.values():
            d = (curr_pos[0] - p[0]) ** 2 + (curr_pos[1] - p[1]) ** 2
            if d < min_d:
                min_d = d
                best_prev = p

        if best_prev and min_d < 4.0:
            return curr_pos[0] - best_prev[0], curr_pos[1] - best_prev[1]
        return 0, 0

    def _update_memory(self, state):
        self.prev_boss_pos = state.boss.pos

        new_threats = {}
        # 仅存储 World Coords
        # 注意：这里需要重新计算 World Coords，为了简化，直接用 list 索引作为 ID 是不靠谱的
        # 但在 Gym 环境里，我们只能做这种近似
        threat_list = state.colliders.enemy_attacks + self._extract_bugged_attacks(state.colliders.hero_attacks)
        for i, shape in enumerate(threat_list):
            cx, cy = self._get_centroid(shape)
            wx = (cx - 0.5) * 30.0 + state.hero.pos[0]
            wy = (cy - 0.5) * 18.0 + state.hero.pos[1]
            new_threats[i] = (wx, wy)

        self.prev_threat_pos = new_threats

    def _get_centroid(self, shape):
        if shape['type'] == 'circle': return shape.get('cx', 0.5), shape.get('cy', 0.5)
        if shape['type'] == 'poly':
            pts = shape.get('pts', [])
            if pts: return pts[0], pts[1]  # 简单取第一点
        return 0.5, 0.5

    def _extract_bugged_attacks(self, hero_attacks):
        bugged = []
        for shape in hero_attacks:
            if shape['type'] == 'poly':
                pts = shape.get('pts', [])
                if len(pts) >= 4:
                    xs = pts[0::2]
                    ys = pts[1::2]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    if h > 0 and (w / h) > 3.0: bugged.append(shape)
        return bugged

    def _filter_ai_action(self, state, action):
        hero = state.hero
        act = np.array(action, copy=True)
        if hero.soul < 33: act[self.IDX_SPELL] = 0
        return act