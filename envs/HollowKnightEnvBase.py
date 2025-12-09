import socket
import struct
import json
import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ==========================================
# 1. 常量定义 (Enums / Constants)
# ==========================================

class ActionType:
    DISCRETE = 0        # 旧版离散 ID (例如 Actions.LEFT = 1)
    MULTI_BINARY = 1    # 新版位掩码/数组 (例如 ActionKeys.UP | ActionKeys.ATTACK)

class Actions:
    """
    [动作指令集]
    直接在 step() 函数中使用这些常量。
    """
    NO_OP = 0  # 不进行任何操作 (松开所有按键)
    LEFT = 1  # 向左移动
    RIGHT = 2  # 向右移动
    UP = 3  # 向上 (看/爬墙)
    DOWN = 4  # 向下 (蹲/下看)
    JUMP = 5  # 跳跃 (短按小跳，长按大跳)
    ATTACK = 6  # 骨钉攻击 (普通平A)
    ATTACK_UP = 7 # 上劈
    ATTACK_DOWN = 8 # 下劈
    ATTACK_LEFT = 9 # 向左走劈
    ATTACK_RIGHT = 10 # 向右走劈
    DASH = 11  # 冲刺 (包含地面冲刺和空中冲刺)
    CAST_SPELL = 12  # 施法/回血 (Focus/Cast) - 原地不动按住回血，点击发射白波
    QUICK_SPELL_UP = 13  # 快速施法上 (Howl/Shriek) - 也就是"吼叫"
    QUICK_SPELL_DOWN = 14  # 快速施法下 (Dive/Dark) - 也就是"下砸"


class ActionKeys:
    """
    [新版原子按键定义 - MultiBinary]
    对应 Mod 接收的 8 位二进制掩码。
    使用按位或 (|) 运算来组合动作。
    """
    NONE = 0

    # 移动键 (Bit 0-3)
    UP = 1 << 0  # W
    LEFT = 1 << 1  # A
    DOWN = 1 << 2  # S
    RIGHT = 1 << 3  # D

    # 功能键 (Bit 4-7)
    ATTACK = 1 << 4  # J (攻击)
    JUMP = 1 << 5  # K (跳跃)
    DASH = 1 << 6  # L (冲刺)
    SPELL = 1 << 7  # I (施法/技能)

class Scenes:
    """
    [场景预设]
    用于 set_scene() 方法，快速切换训练场地。
    """
    MANTIS_LORDS = "GG_Mantis_Lords"  # 螳螂领主 (神居版)
    HORNET_1 = "GG_Hornet_1"  # 小姐姐 1 (神居版)
    HORNET_2 = "GG_Hornet_2"  # 小姐姐 2 (神居版)
    GRIMM = "GG_Grimm"  # 格林团长 (神居版)
    HOLLOW_KNIGHT = "GG_Hollow_Knight"  # 空洞骑士 (神居版)
    RADIANCE = "GG_Radiance"  # 辐光 (神居版)
    GRUZ_MOTHER = "GG_Gruz_Mother"  # 格鲁兹之母 (神居版 - 简单测试用)
    TUTORIAL = "Tutorial_01"  # 游戏开始的教学关 (国王之路)


class DoneState:
    """
    [游戏结束状态]
    用于判断当前 Episode 是否结束。
    """
    PLAYING = "false"  # 游戏进行中
    DEAD = "dead"  # 玩家血量归零 (失败)
    VICTORY = "victory"  # Boss 血量归零 (胜利)

class ResponseModeState:
    """响应模式"""
    ON_DEMAND = "on_demand"
    ASYNC = "async"

class LidarDir:
    """
    [雷达方向索引]
    对应 state.hero.lidar 列表中的索引位置。
    """
    RIGHT = 0  # 右 (0度)
    LEFT = 1  # 左 (180度)
    UP = 2  # 上 (90度)
    DOWN = 3  # 下 (270度)
    UP_RIGHT = 4  # 右上 (45度)
    DOWN_RIGHT = 5  # 右下 (315度)
    UP_LEFT = 6  # 左上 (135度)
    DOWN_LEFT = 7  # 左下 (225度)


# ==========================================
# 2. 数据结构 (Data Classes)
# ==========================================
@dataclass
class ColliderData:
    """存储从Mod传来的原始碰撞体几何数据 (Normalized Viewport Coordinates 0.0-1.0)"""
    hero: List[dict] = field(default_factory=list)
    terrain: List[dict] = field(default_factory=list)
    hero_attacks: List[dict] = field(default_factory=list)
    enemies: List[dict] = field(default_factory=list)
    destructibles: List[dict] = field(default_factory=list)
    enemy_attacks: List[dict] = field(default_factory=list)
    traps: List[dict] = field(default_factory=list)

    def get_category(self, key: str) -> List[dict]:
        """根据字符串键获取对应的列表"""
        return getattr(self, key, [])


# 新增一个工具类用于处理栅格化
class CollisionUtils:
    @staticmethod
    def rasterize_category(shape_list: List[dict], width: int, height: int) -> np.ndarray:
        """
        将某一类别的形状列表转换为二值掩码矩阵 (uint8, 0 or 1)。

        参数:
        - shape_list: 包含 {'type': 'circle'/'poly', ...} 的字典列表
        - width: 目标图像宽度
        - height: 目标图像高度

        返回:
        - mask: (height, width) 的 numpy 数组
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        if not shape_list:
            return mask

        for shape in shape_list:
            try:
                if shape['type'] == 'poly':
                    # 处理多边形: pts 是 [x1, y1, x2, y2, ...]
                    raw_pts = shape.get('pts', [])
                    if len(raw_pts) < 4: continue

                    # 将坐标对转换为 (N, 2) 的数组
                    # 注意：Mod发来的是 Viewport 坐标 (0,0在左下), 图像 (0,0在左上)
                    # 所以 Y 轴需要翻转: y_img = (1.0 - y_vp) * height
                    points = []
                    for i in range(0, len(raw_pts), 2):
                        x = int(raw_pts[i] * width)
                        y = int((1.0 - raw_pts[i + 1]) * height)
                        points.append([x, y])

                    pts_np = np.array([points], dtype=np.int32)
                    cv2.fillPoly(mask, pts_np, 1)

                elif shape['type'] == 'circle':
                    # 处理圆形/椭圆
                    cx = shape.get('cx', 0)
                    cy = shape.get('cy', 0)
                    rx = shape.get('rx', 0)
                    ry = shape.get('ry', 0)

                    # 坐标转换与Y轴翻转
                    center_x = int(cx * width)
                    center_y = int((1.0 - cy) * height)
                    axis_x = int(rx * width)
                    axis_y = int(ry * height)

                    # 绘制椭圆 (填充)
                    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y),
                                0, 0, 360, 1, -1)
            except Exception as e:
                # 防止单个形状数据错误导致崩溃
                print(f"Rasterize error: {e}")
                continue

        return mask

    @staticmethod
    def get_all_masks(collider_data: 'ColliderData', width: int, height: int, categories_in = None) -> dict:
        """
        一次性获取所有类别的掩码字典
        """
        if categories_in is None:
            categories = ['hero',
                          'terrain',
                          'hero_attacks',
                          'enemies',
                          'destructibles',
                          'enemy_attacks',
                          'traps']
        else:
            categories = categories_in
        masks = {}
        for cat in categories:
            data_list = getattr(collider_data, cat)
            masks[cat] = CollisionUtils.rasterize_category(data_list, width, height)
        return masks

    @staticmethod
    def apply_visual_mask(img: np.ndarray, colliders: 'ColliderData', expansion_config: dict) -> np.ndarray:
        """
        利用碰撞体数据生成掩码，并将其作用于图像，过滤背景噪声。

        参数:
        - img: 输入图像 (H, W, C) 或 (H, W)
        - colliders: 碰撞体数据对象
        - expansion_config: 字典，定义各类别相对于图像宽度的膨胀比例，例如 {'hero': 0.1}

        返回:
        - masked_img: 处理后的图像
        """
        if img is None or colliders is None:
            return img

        h, w = img.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)

        has_any_mask = False

        for category, ratio in expansion_config.items():
            # 1. 获取形状
            shapes = colliders.get_category(category)
            if not shapes:
                continue

            # 2. 栅格化 (复用现有逻辑)
            cat_mask = CollisionUtils.rasterize_category(shapes, w, h)

            # 判断该类别是否有内容，如果全是0就跳过膨胀计算，节省时间
            if not np.any(cat_mask):
                continue

            # 3. 膨胀处理 (Expansion)
            if ratio > 0:
                k_size = int(w * ratio)
                # 确保核大小是奇数
                if k_size % 2 == 0: k_size += 1
                if k_size < 3: k_size = 3  # 最小核

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                cat_mask = cv2.dilate(cat_mask, kernel, iterations=1)

            # 4. 合并掩码
            final_mask = cv2.bitwise_or(final_mask, cat_mask)
            has_any_mask = True

        # 如果没有任何碰撞体被绘制（极端情况），直接返回全黑图，或者原图（取决于需求）
        # 这里选择返回全黑，表示没有关注点
        if not has_any_mask:
            return np.zeros_like(img)

        # 5. 应用掩码
        # 处理通道匹配问题
        if len(img.shape) == 3:
            # 彩色图/多通道，扩展掩码
            mask_applied = cv2.bitwise_and(img, img, mask=final_mask)
        else:
            # 单通道图
            mask_applied = cv2.bitwise_and(img, img, mask=final_mask)

        return mask_applied

@dataclass
class ConfigState:
    """游戏的详细状态"""
    time_scale: float # 时间流速倍率
    target_fps: int # 目标帧率
    frame_skip: int  # 当前的帧采样间隔 (Mod内部设置)
    resolution: Tuple[int, int] # 采样图像分辨率
    response_mode: str # 回传模式
    target_scene: str # 目标场景

@dataclass
class HeroState:
    """主角(小骑士)的详细状态"""
    # --- 基础属性 ---
    hp: int  # 当前血量
    max_hp: int  # 最大血量
    soul: int  # 当前灵魂量 (0-99)，33放一次法术
    pos: Tuple[float, float]  # 世界坐标位置 (x, y)
    vel: Tuple[float, float]  # 当前速度向量 (vx, vy)
    facing: int  # 朝向：1 为向右，-1 为向左

    # --- 物理/动作状态 (布尔值) ---
    on_ground: bool  # 是否在地面上
    on_wall: bool  # 是否在爬墙/贴墙
    can_dash: bool  # 冲刺是否冷却完毕 (True代表可以冲刺)
    # air_dashed: bool  # 是否已经使用过空中冲刺 (落地前不可再次空冲)
    shadow_timer: float  # 暗影冲刺(黑冲)冷却剩余时间 (秒)，0代表可用
    has_shadow_dash: bool  # 是否拥有暗影冲刺能力
    is_recoiling: bool  # 是否处于受击硬直中 (此时不可控)
    can_double_jump: bool  # 是否可以使用二段跳
    is_invincible: bool  # 是否处于无敌帧 (冲刺或受击后)
    is_healing: bool  # 是否正在聚集灵魂回血
    nail_damage: int  # 当前骨钉攻击力

    # --- 感知数据 ---
    lidar: List[float]  # 视觉雷达：8个方向到最近障碍物的距离 (最大5.0)
    # 顺序见 LidarDir 类


@dataclass
class BossState:
    """Boss 的详细状态"""
    exists: bool  # 当前场景是否存在 Boss
    name: str = "None"  # Boss 名称
    hp: int = 0  # Boss 当前血量
    pos: Tuple[float, float] = (0.0, 0.0)  # Boss 坐标 (x, y)
    vel: Tuple[float, float] = (0.0, 0.0)  # Boss 速度 (vx, vy)
    rel_pos: Tuple[float, float] = (0.0, 0.0)  # Boss 相对于主角的向量 (dx, dy)
    # dx > 0 表示 Boss 在主角右边


@dataclass
class GameState:
    """每一帧返回的完整游戏状态包"""
    ready: bool
    mode: str  # 当前模式: "training" 或 "normal"
    paused: bool  # 游戏是否处于暂停状态
    done: str  # 结束标记: 见 DoneState (false/dead/victory)
    colliders: ColliderData # 碰撞体数据
    config: ConfigState
    hero: HeroState  # 主角状态对象
    boss: BossState  # Boss状态对象


# ==========================================
# 3. 控制器核心 (HKController)
# ==========================================

class HKController:
    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self._last_image = None
        self.client = None
        self.hard_mode = False

    def connect(self):
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((self.host, self.port))
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connected = True
            print(f"Connected to HK Mod at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self.connected = False
            return False

    def close(self):
        if self.client:
            self.client.close()
            self.connected = False

    def step(self, action, action_type=ActionType.MULTI_BINARY, visualize=False):
        """
        发送动作并获取下一帧状态。

        参数:
        - action: 动作数据
        - action_type: ActionType.DISCRETE 或 ActionType.MULTI_BINARY
        - visualize: 是否解码图像 (为了速度通常设为 False)
        """
        # 这里的 sleep 保留你原来的逻辑，用于物理缓冲，但训练时可能需要移除
        time.sleep(0.01)

        if not self.connected:
            return None, None

        # 1. 根据类型显式编码
        cmd = self._encode_action(action, action_type)

        try:
            # 2. 发送指令
            self.client.sendall(cmd.encode('utf-8'))

            # 3. 接收回传
            state_data, img_data = self._recv_packet()

            # 4. 解析 JSON
            game_state = None
            if state_data:
                game_state = self._parse_json_to_object(json.loads(state_data))

            # 5. 解析图像 (仅在需要时)
            image = None
            if img_data and len(img_data) > 0 and visualize:
                import cv2
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.flip(image, 0)

            return game_state, image

        except Exception as e:
            print(f"Step error: {e}")
            self.connected = False
            return None, None

    def _encode_action(self, action, action_type):
        """
        根据 action_type 明确决定协议格式
        """
        # --- 模式 1: 旧版离散动作 ---
        if action_type == ActionType.DISCRETE:
            # 直接发送数字 ID，例如 "5\n"
            return f"{int(action)}\n"

        # --- 模式 2: 新版 Multi-Binary ---
        elif action_type == ActionType.MULTI_BINARY:
            mask = 0

            # 子情况 A: 输入是数组/列表 (例如 [0, 1, 0...]) -> 转换成掩码
            if isinstance(action, (list, tuple, np.ndarray)):
                for i, val in enumerate(action):
                    if val > 0.5: mask |= (1 << i)

            # 子情况 B: 输入已经是整数或枚举 (例如 ActionKeys.JUMP) -> 直接使用
            elif isinstance(action, (int, np.integer, ActionKeys)):
                mask = int(action)

            # 发送 ACT 协议
            return f"ACT:{mask}\n"

        # 默认回落
        return "0\n"

    def _recv_packet(self):
        """处理 Mod 自定义的二进制包协议"""

        def recv_exact(n):
            data = b''
            while len(data) < n:
                packet = self.client.recv(n - len(data))
                if not packet: return None
                data += packet
            return data

        # 1. JSON 长度
        len_bytes = recv_exact(4)
        if not len_bytes: return None, None
        json_len = struct.unpack('<I', len_bytes)[0]

        # 2. JSON 数据
        json_bytes = recv_exact(json_len)

        # 3. Image 长度
        len_bytes = recv_exact(4)
        if not len_bytes: return None, None
        img_len = struct.unpack('<I', len_bytes)[0]

        # 4. Image 数据
        img_bytes = recv_exact(img_len) if img_len > 0 else b''

        return json_bytes, img_bytes

    def reset_game(self):
        # print("Resetting game")
        self._send_command("RESET")
        time.sleep(1.2)

    def pause_game(self):
        print("[CMD] 暂停游戏")
        self._send_command("PAUSE")

    def resume_game(self):
        """恢复游戏 (TimeScale = 1)"""
        print("[CMD] 恢复游戏")
        self._send_command("RESUME")

    def set_load_delay(self, delay_seconds: float):
        """
        设置场景加载后的硬等待时间。
        :param delay_seconds: 秒数，例如 1.5
        """
        # 确保发送的是点号小数
        self._send_command(f"SET_LOAD_DELAY:{delay_seconds:.2f}")
        time.sleep(0.1) # 给一点点时间让 Mod 处理

    def set_hard_mode(self, enabled: bool):
        """设置是否受伤即重置"""
        cmd = "TRUE" if enabled else "FALSE"
        self.hard_mode = enabled
        self._send_command(f"SET_HARD_MODE:{cmd}")

    def set_response_mode(self, enable: bool):
        """
        设置数据返回模式。
        True (OnDemand): Python发送动作 -> Mod等待下一个采样帧 -> Mod截图发回 -> Python解除阻塞。
                         (推荐用于训练，数据不重复，游戏不暂停)
        False (Async): Mod一直在后台截图，Python随时获取最新的缓存帧。
                       (测试/体验，画面最流畅)
        """
        mode_str = "TRUE" if enable else "FALSE"
        print(f"[CMD] 设置响应模式 (OnDemand) -> {mode_str}")
        self._send_command(f"SET_SYNC:{mode_str}")
        time.sleep(0.2)

    def set_mode_training(self):
        self._send_command("SET_MODE:TRAINING")
        time.sleep(1.2)

    def set_mode_normal(self):
        self._send_command("SET_MODE:NORMAL")
        time.sleep(1.2)

    def set_scene(self, scene_name: str):
        self._send_command(f"SET_SCENE:{scene_name}")
        time.sleep(1.2)

    def set_frame_skip(self, skip: int):
        self._send_command(f"SET_SKIP:{skip}")
        time.sleep(0.2)

    def set_frame_resolution(self, resolution: tuple):
        self._send_command(f"SET_FRAME_RESOLUTION:{resolution[0]}x{resolution[1]}")

    def set_timescale(self, timescale: float):
        self._send_command(f"SET_TIMESCALE:{timescale}")

    def set_fps(self, fps: int):
        self._send_command(f"SET_FPS:{fps}")
        time.sleep(0.2)

    def _send_command(self, cmd):
        if self.connected:
            try:
                self.client.sendall(cmd.encode('utf-8'))
            except:
                self.connected = False

    def _parse_json_to_object(self, d: dict) -> GameState:
        """将原始字典转换为强类型的 GameState 对象"""
        c = d.get("config", {})
        h = d.get('hero', {})
        b = d.get('boss', {})
        col = d.get('camera_colliders', {})

        config_resolution = (c.get('resolution', (480, 270)))
        if isinstance(config_resolution, str):
            config_resolution = config_resolution.split('x')
            config_resolution = (int(config_resolution[0]), int(config_resolution[1]))

        collider_obj = ColliderData(
            hero=col.get('hero', []),
            terrain=col.get('terrain', []),
            hero_attacks=col.get('hero_attacks', []),
            enemies=col.get('enemies', []),
            destructibles=col.get('destructibles', []),
            enemy_attacks=col.get('enemy_attacks', []),
            traps=col.get('traps', [])
        )

        config_obj = ConfigState(
            time_scale=c.get('time_scale', 1),
            target_fps=c.get('target_fps', 144),
            frame_skip=c.get('frame_skip', 1),
            resolution=config_resolution,
            response_mode=c.get('response_mode', ResponseModeState.ON_DEMAND),
            target_scene=c.get('target_scene', Scenes.MANTIS_LORDS)
        )

        hero_obj = HeroState(
            hp=h.get('hp', 0),
            max_hp=h.get('max_hp', 0),
            soul=h.get('soul', 0),
            pos=tuple(h.get('pos', (0., 0.))),
            vel=tuple(h.get('vel', (0., 0.))),
            facing=h.get('facing', 1),
            on_ground=h.get('on_ground', False),
            on_wall=h.get('on_wall', False),
            can_dash=h.get('can_dash', False),
            # air_dashed=h.get('air_dashed', False),
            shadow_timer=h.get('shadow_timer', 0.0),
            has_shadow_dash=h.get('has_shadow_dash', False),
            is_recoiling=h.get('is_recoiling', False),
            can_double_jump=h.get('can_double_jump', False),
            is_invincible=h.get('is_invincible', False),
            is_healing=h.get('is_healing', False),
            nail_damage=h.get('nail_damage', 0),
            lidar=h.get('lidar', [0, 0, 0, 0, 0, 0, 0, 0])
        )

        boss_obj = BossState(
            exists=b.get('exists', False),
            name=b.get('name', 'None'),
            hp=b.get('hp', 0),
            pos=tuple(b.get('pos', (0, 0))),
            vel=tuple(b.get('vel', (0, 0))),
            rel_pos=tuple(b.get('rel_pos', (0, 0)))
        )

        return GameState(
            ready=d.get('ready', True),
            mode=d.get('mode', 'normal'),
            paused=d.get('paused', False),
            done=d.get('done', 'false'),
            config=config_obj,
            hero=hero_obj,
            boss=boss_obj,
            colliders=collider_obj
        )

# === 使用示例 ===
if __name__ == '__main__':
    env = HKController()
    if env.connect():
        env.set_mode_training()
        env.set_frame_skip(30)

        # 开启按需响应模式！(游戏不暂停，但必须等指令才发数据)
        env.set_response_mode(True)

        count = 0
        while True:
            # 发送 JUMP
            # Python 会在这里停顿一小会儿 (约 5/60 秒)
            # 等待 Mod 捕捉到执行 JUMP 后的第 5 帧
            if count % 2 == 0:
                state, img = env.step(Actions.ATTACK, visualize=True)
            else:
                state, img = env.step(Actions.JUMP, visualize=True)
            count += 1
            if count % 50 == 0:
                env.pause()
            if state is None: break
            print(state['hero']['pos'])
            # time.sleep(0.1)