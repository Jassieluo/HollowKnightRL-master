import math
import random

import numpy as np


class SpatialExpertLogic:
    """
    空间感知专家战术逻辑
    负责根据 Raw GameState 计算战术动作。
    """

    def __init__(self):
        # 动作索引映射 (MultiDiscrete)
        self.IDX_HOR = 0
        self.IDX_VER = 1
        self.IDX_JUMP = 2
        self.IDX_ATTACK = 3
        self.IDX_DASH = 4
        self.IDX_SPELL = 5

        # 内部记忆
        self.last_attack_pressed = False
        self.prev_boss_pos = None
        self.prev_threat_pos = {}

    def reset_state(self):
        self.last_attack_pressed = False
        self.prev_boss_pos = None
        self.prev_threat_pos = {}

    def compute_action(self, state) -> np.ndarray:
        """
        核心接口：输入 GameState -> 输出 Action Array (6,)
        """
        if state is None:
            return np.zeros(6, dtype=int)

        # 更新记忆
        self._update_memory(state)
        # 计算动作
        return self._derive_tactical_action(state)

    def _derive_tactical_action(self, state):
        hero = state.hero
        boss = state.boss
        colliders = state.colliders

        act = np.zeros(6, dtype=int)

        # 基础数据
        dx = hero.pos[0] - boss.pos[0]
        dy = hero.pos[1] - boss.pos[1]
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # 默认面向 Boss
        face_dir = 2 if dx < 0 else 1  # 1:Right, 2:Left
        away_dir = 1 if dx < 0 else 2

        act[self.IDX_HOR] = face_dir

        # --- 第一步：威胁识别 ---
        threat_type = "NONE"
        threat_obj = None

        # 1. Boss 冲撞
        boss_vel_x = boss.vel[0]
        if abs(boss_vel_x) > 3.0 and (boss.pos[1] < hero.pos[1] + 2.0):
            if (dx > 0 and boss_vel_x > 0) or (dx < 0 and boss_vel_x < 0):
                if dist < 8.0:
                    threat_type = "CHARGE"

        # 2. Boss 下砸
        if threat_type == "NONE":
            if (boss.pos[1] > hero.pos[1] + 1.0) and (boss.vel[1] < -1.0) and (abs(dx) < 3.5):
                threat_type = "SLAM"

        # 3. 飞行道具
        if threat_type == "NONE":
            proj = self._find_incoming_projectile(hero, colliders)
            if proj:
                threat_type = "PROJECTILE"
                threat_obj = proj

        # 4. 接触判定
        if threat_type == "NONE":
            if dist < 2.0:
                threat_type = "CONTACT"

        # --- 第二步：闪避决策 ---
        is_dodging = False

        if threat_type == "SLAM":
            is_dodging = True
            act[self.IDX_DASH] = 1
            act[self.IDX_HOR] = away_dir

        elif threat_type == "CHARGE" or threat_type == "PROJECTILE":
            is_dodging = True
            can_shadow_dash = hero.has_shadow_dash and hero.shadow_timer <= 0
            is_flat_bug = (threat_obj and threat_obj.get('is_flat'))

            if can_shadow_dash and not is_flat_bug:
                act[self.IDX_DASH] = 1
                act[self.IDX_HOR] = face_dir  # 迎面黑冲
            else:
                act[self.IDX_JUMP] = 1  # 跳跃躲避
                # Pogo (下劈) 准备
                threat_y = threat_obj['pos'][1] if threat_obj else boss.pos[1]
                if hero.pos[1] > threat_y - 0.5:
                    act[self.IDX_ATTACK] = 1
                    act[self.IDX_VER] = 2

        elif threat_type == "CONTACT":
            act[self.IDX_HOR] = away_dir
            if random.random() < 0.1: act[self.IDX_JUMP] = 1

        # 绝境下砸 (Dive)
        if is_dodging and not hero.can_dash and dist < 2.5 and hero.soul >= 33:
            act[self.IDX_HOR] = 0
            act[self.IDX_VER] = 2
            act[self.IDX_SPELL] = 1
            return act

            # --- 第三步：攻击决策 ---
        if act[self.IDX_DASH] == 0 and act[self.IDX_SPELL] == 0:
            # 1. 波动拳 (Vengeful Spirit)
            if hero.soul >= 33 and dist > 6.0 and abs(dy) < 2.0:
                act[self.IDX_VER] = 0
                act[self.IDX_SPELL] = 1

            # 2. 上吼 (Howling Wraiths)
            elif hero.soul >= 33 and (boss.pos[1] > hero.pos[1] + 1.0) and (abs(dx) < 2.5) and (threat_type == "NONE"):
                act[self.IDX_VER] = 1
                act[self.IDX_SPELL] = 1

            # 3. 平A
            else:
                ATTACK_RANGE = 5.0
                if dist < ATTACK_RANGE:
                    if not self.last_attack_pressed:
                        act[self.IDX_ATTACK] = 1
                        self.last_attack_pressed = True
                        # 方向修正
                        if boss.pos[1] > hero.pos[1] + 0.5:
                            act[self.IDX_VER] = 1  # Up
                        elif hero.pos[1] > boss.pos[1] + 1.0:
                            act[self.IDX_VER] = 2  # Down
                        else:
                            act[self.IDX_VER] = 0
                    else:
                        act[self.IDX_ATTACK] = 0
                        self.last_attack_pressed = False
                else:
                    self.last_attack_pressed = False

        # --- 第四步：走位微调 ---
        if not is_dodging and act[self.IDX_SPELL] == 0:
            SAFE_DIST = 2.0
            if dist < SAFE_DIST:
                act[self.IDX_HOR] = away_dir
            elif dist > 5.0:
                act[self.IDX_HOR] = face_dir

        # 冲突清洗
        if act[self.IDX_SPELL] == 1: act[self.IDX_ATTACK] = 0
        if act[self.IDX_DASH] == 1: act[self.IDX_ATTACK] = 0

        return act

    # === 辅助函数 ===
    def _find_incoming_projectile(self, hero, colliders):
        threats = colliders.enemy_attacks + self._extract_bugged_attacks(colliders.hero_attacks)
        for shape in threats:
            cx, cy = self._get_centroid(shape)
            wx = (cx - 0.5) * 30.0 + hero.pos[0]
            wy = (cy - 0.5) * 18.0 + hero.pos[1]
            dist = math.sqrt((wx - hero.pos[0]) ** 2 + (wy - hero.pos[1]) ** 2)

            if dist < 6.0:
                vx, vy = self._get_velocity(shape, (wx, wy))
                is_incoming = False
                if abs(vx) > 0.1:
                    if (wx > hero.pos[0] and vx < 0) or (wx < hero.pos[0] and vx > 0):
                        is_incoming = True
                if is_incoming or dist < 2.5:
                    return {'pos': (wx, wy), 'vel': (vx, vy), 'is_flat': (shape in colliders.hero_attacks)}
        return None

    def _get_velocity(self, shape, curr_pos):
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
            if pts: return pts[0], pts[1]
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