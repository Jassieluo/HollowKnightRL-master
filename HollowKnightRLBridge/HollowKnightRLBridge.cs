using GlobalEnums;
using InControl;
using Modding;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HollowKnightRL
{
    public class RLSettings
    {
        public bool IsTrainingMode = false;
        public string TargetScene = "GG_Mantis_Lords";
        public bool EnableCamera = true;
        public int ResolutionIndex = 2;
        public int FrameSkip = 3;
        public bool IsStepResponseMode = false;

        // [NEW] 时间流速 (1.0f = 正常, 2.0f = 2倍速)
        public float TimeScale = 1.0f;
        // [NEW] 目标帧率 (-1 = 不限制, 60 = 60fps)
        public int TargetFPS = 30;

        public bool ResetOnDamage = false;

        public float LevelLoadDelay = 2.0f;
    }

    public class HollowKnightRL : Mod, IMenuMod, IGlobalSettings<RLSettings>, ITogglableMod
    {
        public override string GetVersion() => "39.4.0-TimeCtrl";

        public static RLSettings settings = new RLSettings();
        public void OnLoadGlobal(RLSettings s) => settings = s;
        public RLSettings OnSaveGlobal() => settings;

        private int serverPort = 5555;
        private TcpListener tcpListener;
        private Thread networkThread;
        private volatile bool isClientConnected = false;
        private volatile int currentAction = 0;
        private volatile bool[] multiBinaryActions = new bool[8];
        private volatile bool useMultiBinary = false;
        private volatile bool stopServer = false;

        private object dataLock = new object();
        private byte[] cachedJsonBytes = new byte[0];
        private byte[] cachedImageBytes = new byte[0];

        private AutoResetEvent captureReadyEvent = new AutoResetEvent(false);
        private volatile bool requestPending = false;

        private bool needsReset = false;
        private bool ignoreInput = false;
        private volatile bool isPaused = false;
        private string currentDoneState = "false";
        private bool isResetting = false;

        private FieldInfo stateField, lastStateField, boolStateField;
        private FieldInfo dashCooldownField, airDashedField, shadowDashTimerField;
        private Type bossSceneControllerType;
        private PropertyInfo bossesDeadProp;
        private bool reflectionInitialized = false;
        private MethodInfo fsmSendEventMethod;
        private int spellCooldownCounter = 0;

        private int captureWidth = 128;
        private int captureHeight = 128;
        private Texture2D screenShotTex;
        private RenderTexture renderTex;
        private bool textureNeedsUpdate = false;
        private int terrainLayerMask;

        private readonly int[][] resolutions = new int[][]
        {
            new int[] { 64, 64 }, new int[] { 84, 84 }, new int[] { 128, 128 }, new int[] { 240, 135 },
            new int[] { 480, 270 }, new int[] { 640, 360 }
        };
        private string[] resNames = new string[] { "64x64", "84x84", "128x128", "240x135", "480x270", "640x360" };
        private string[] sampleNames = new string[] { "Every Frame", "Every 2 Frames", "Every 3 Frames", "Every 4 Frames", "Every 5 Frames",
            "Every 6 Frames", "Every 7 Frames", "Every 8 Frames", "Every 9 Frames", "Every 10 Frames", "Every 11 Frames",
            "Every 12 Frames", "Every 13 Frames", "Every 14 Frames", "Every 15 Frames"};

        // [NEW] UI用的时间流速选项
        private string[] timeScaleNames = new string[] { "0.5x", "1.0x", "1.5x", "2.0x", "3.0x", "5.0x" };
        private float[] timeScaleValues = new float[] { 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f };

        // [NEW] UI用的帧率选项
        private string[] fpsNames = new string[] { "10 FPS", "30 FPS", "60 FPS", "100 FPS", "144 FPS", "Unlimited" };
        private int[] fpsValues = new int[] { 10, 30, 60, 100, 144, -1 };

        private string[] delayNames = new string[] { "0.0s", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "4.0s", "5.0s" };
        private float[] delayValues = new float[] { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 5.0f };

        private string[] scenePresets = new string[]
        {
            "GG_Mantis_Lords", "GG_Hornet_2", "GG_Grimm", "GG_Hollow_Knight",
            "GG_Radiance", "GG_Gruz_Mother", "Tutorial_01"
        };
        private int sceneIndex = 0;

        public override void Initialize()
        {
            Log("=== Init RL Mod (MultiBinary) ===");
            InitReflection();
            UpdateResolutionVariables();
            ApplyFpsSettings();
            //ApplyTimeScaleSettings();
            terrainLayerMask = 1 << 8;
            Application.runInBackground = true;

            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++) { if (args[i] == "-rlport" && int.TryParse(args[i + 1], out int p)) serverPort = p; }

            stopServer = false;
            networkThread = new Thread(NetworkLoop) { IsBackground = true };
            networkThread.Start();

            UnityEngine.SceneManagement.SceneManager.activeSceneChanged += OnSceneChanged;
            ModHooks.HeroUpdateHook += OnHeroUpdate;
            ModHooks.TakeHealthHook += OnTakeDamage;

            On.InputHandler.Update += InputHandler_Update_Hook;
            On.HeroController.LookForInput += HC_LookForInput_Hook;

            GameManager.instance.StartCoroutine(CaptureFrameCoroutine());
        }

        private void UpdateResolutionCustom(int resolution_weight, int resolution_height)
        {
            if (resolution_weight > 0 && resolution_height > 0) { 
                captureWidth = resolution_weight;
                captureHeight = resolution_height;
                textureNeedsUpdate = true;
            }
        }

        private void UpdateResolutionVariables()
        {
            if (settings.ResolutionIndex < 0 || settings.ResolutionIndex >= resolutions.Length) settings.ResolutionIndex = 3;
            captureWidth = resolutions[settings.ResolutionIndex][0];
            captureHeight = resolutions[settings.ResolutionIndex][1];
            textureNeedsUpdate = true;
        }

        // [NEW] 应用帧率设置
        private void ApplyFpsSettings()
        {
            Application.targetFrameRate = settings.TargetFPS;
            // VSync 需要关闭才能让 targetFrameRate 生效（通常情况）
            QualitySettings.vSyncCount = 0;
        }

        //// [NEW] 应用时间流速（如果在游戏中）
        //private void ApplyTimeScaleSettings()
        //{
        //    if (!isPaused)
        //    {
        //        Time.timeScale = settings.TimeScale;
        //    }
        //}

        // [NEW] 应用时间流速（如果在游戏中）
        private void ApplyTimeScaleSettings()
        {
            if (!isPaused)
            {
                Time.timeScale = settings.TimeScale;
            }

            // [FIX] 同时设置 Time.fixedDeltaTime 以确保物理系统正常工作
            //Time.fixedDeltaTime = 0.01666667f * settings.TimeScale; // 0.01666667f 是 60fps 的固定时间步长
        }

        private IEnumerator CaptureFrameCoroutine()
        {
            int frameCounter = 0;
            while (true)
            {
                yield return new WaitForEndOfFrame();

                if (textureNeedsUpdate)
                {
                    if (renderTex != null) renderTex.Release();
                    renderTex = new RenderTexture(captureWidth, captureHeight, 24);
                    screenShotTex = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
                    textureNeedsUpdate = false;
                }

                frameCounter++;
                bool isSampleFrame = (frameCounter % settings.FrameSkip == 0);

                bool doCapture = false;
                bool signalNetwork = false;

                if (settings.IsStepResponseMode)
                {
                    if (isSampleFrame && requestPending)
                    {
                        doCapture = true;
                        signalNetwork = true;
                    }
                }
                else
                {
                    if (isSampleFrame) doCapture = true;
                }

                if (doCapture && isClientConnected && settings.EnableCamera)
                {
                    try
                    {
                        RenderTexture fullScreenRT = RenderTexture.GetTemporary(Screen.width, Screen.height, 24);
                        Graphics.Blit(null, fullScreenRT);
                        Graphics.Blit(fullScreenRT, renderTex);
                        RenderTexture.ReleaseTemporary(fullScreenRT);
                        RenderTexture oldRT = RenderTexture.active;
                        RenderTexture.active = renderTex;
                        screenShotTex.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
                        screenShotTex.Apply();
                        RenderTexture.active = oldRT;
                        byte[] imgBytes = ImageConversion.EncodeToJPG(screenShotTex, 30);
                        lock (dataLock) { cachedImageBytes = imgBytes; }

                        string jsonStr = GetCurrentStateJSON();
                        byte[] jsonBytes = Encoding.UTF8.GetBytes(jsonStr);
                        lock (dataLock) { cachedJsonBytes = jsonBytes; }

                        if (signalNetwork)
                        {
                            requestPending = false;
                            captureReadyEvent.Set();
                        }
                    }
                    catch (Exception ex) { Log("Capture Error: " + ex.Message); }
                }
            }
        }

        private void NetworkLoop()
        {
            try { tcpListener = new TcpListener(IPAddress.Any, serverPort); tcpListener.Start(); } catch { return; }
            while (!stopServer)
            {
                try
                {
                    if (!tcpListener.Pending()) { Thread.Sleep(10); continue; }
                    TcpClient client = tcpListener.AcceptTcpClient(); 
                    isClientConnected = true;

                    ThreadingHelper.Instance.ExecuteSync(() => ResetAllInputs());

                    //client.NoDelay = true;
                    using (NetworkStream stream = client.GetStream())
                    {
                        byte[] buffer = new byte[4096];
                        while (client.Connected && !stopServer)
                        {
                            int r = stream.Read(buffer, 0, buffer.Length); if (r == 0) break;
                            string msg = Encoding.UTF8.GetString(buffer, 0, r).Trim();
                            Log(msg); // 如果指令太多，可以注释掉
                            if (msg.StartsWith("ACT:"))
                            {
                                // [新增] 解析二进制掩码协议
                                if (int.TryParse(msg.Substring(4).Trim(), out int mask))
                                {
                                    // 将整数每一位拆解为 bool 存入数组
                                    for (int i = 0; i < 8; i++)
                                    {
                                        multiBinaryActions[i] = ((mask >> i) & 1) == 1;
                                    }
                                    useMultiBinary = true; // 激活新模式
                                }
                            }
                            else if (msg.StartsWith("RESET"))
                            {
                                if (settings.IsTrainingMode) needsReset = true;
                            }
                            else if (msg.StartsWith("SET_HARD_MODE:"))
                            {
                                settings.ResetOnDamage = msg.ToUpper().Contains("TRUE");
                            }
                            else if (msg.StartsWith("SET_LOAD_DELAY:"))
                            {
                                if (float.TryParse(msg.Substring(15).Trim(), out float d))
                                {
                                    settings.LevelLoadDelay = Mathf.Clamp(d, 0f, 10f);
                                    Log($"Set Load Delay to: {settings.LevelLoadDelay}");
                                }
                            }
                            else if (msg.StartsWith("SET_MODE:"))
                            {
                                settings.IsTrainingMode = msg.ToUpper().Contains("TRAINING");
                                if (settings.IsTrainingMode) needsReset = true;
                            }
                            else if (msg.StartsWith("SET_SCENE:"))
                            {
                                //settings.TargetScene = msg.Substring(10).Trim();
                                string temp = msg.Substring(10).Trim();
                                int newlineIndex = temp.IndexOf('\n');
                                settings.TargetScene = newlineIndex >= 0 ? temp.Substring(0, newlineIndex) : temp;
                                if (settings.IsTrainingMode) needsReset = true;
                            }
                            else if (msg.StartsWith("SET_SKIP:"))
                            {
                                if (int.TryParse(msg.Substring(9).Trim(), out int skip))
                                    settings.FrameSkip = Mathf.Clamp(skip, 1, 60);
                                Log($"Set FrameSkip: {settings.FrameSkip}");
                            }
                            else if (msg.StartsWith("SET_SYNC:"))
                            {
                                settings.IsStepResponseMode = msg.ToUpper().Contains("TRUE");
                                requestPending = false; captureReadyEvent.Reset();
                            }
                            // [NEW] Python 设置时间流速
                            else if (msg.StartsWith("SET_TIMESCALE:"))
                            {
                                if (float.TryParse(msg.Substring(14).Trim(), out float ts))
                                {
                                    settings.TimeScale = Mathf.Clamp(ts, 0.1f, 10.0f);
                                    ThreadingHelper.Instance.ExecuteSync(() => ApplyTimeScaleSettings());
                                }
                            }
                            // [NEW] Python 设置帧率
                            else if (msg.StartsWith("SET_FPS:"))
                            {
                                if (int.TryParse(msg.Substring(8).Trim(), out int fps))
                                {
                                    settings.TargetFPS = fps;
                                    ThreadingHelper.Instance.ExecuteSync(() => ApplyFpsSettings());
                                }
                            }
                            else if (msg.StartsWith("SET_FRAME_RESOLUTION:"))
                            {
                                string resolutionStr = msg.Substring("SET_FRAME_RESOLUTION:".Length).Trim();
                                string[] dimensions = resolutionStr.Split('x');

                                if (dimensions.Length == 2)
                                {
                                    string widthStr = dimensions[0].Trim();
                                    string heightStr = dimensions[1].Trim();

                                    if (int.TryParse(widthStr, out int width) &&
                                        int.TryParse(heightStr, out int height) &&
                                        width > 0 && height > 0) // 确保分辨率是正数
                                    {
                                        UpdateResolutionCustom(width, height);
                                    }
                                }
                            }
                            else if (msg.StartsWith("PAUSE")) ThreadingHelper.Instance.ExecuteSync(() => SetPaused(true));
                            else if (msg.StartsWith("RESUME")) ThreadingHelper.Instance.ExecuteSync(() => SetPaused(false));
                            else
                            {
                                string[] parts = msg.Split(new char[] { '\n', '\r', ' ' }, StringSplitOptions.RemoveEmptyEntries);
                                if (parts.Length > 0 && int.TryParse(parts[parts.Length - 1], out int id))
                                {
                                    currentAction = id;
                                    useMultiBinary = false; // [重要] 收到旧指令，关闭新模式
                                }
                            }

                            if (settings.IsStepResponseMode)
                            {
                                requestPending = true;
                                captureReadyEvent.WaitOne(1000);
                            }

                            byte[] jsonToSend; byte[] imgToSend;
                            lock (dataLock) { jsonToSend = cachedJsonBytes; imgToSend = cachedImageBytes; }
                            if (jsonToSend.Length == 0) jsonToSend = Encoding.UTF8.GetBytes("{}");

                            try
                            {
                                stream.Write(BitConverter.GetBytes(jsonToSend.Length), 0, 4);
                                stream.Write(jsonToSend, 0, jsonToSend.Length);
                                stream.Write(BitConverter.GetBytes(imgToSend.Length), 0, 4);
                                if (imgToSend.Length > 0) stream.Write(imgToSend, 0, imgToSend.Length);
                            }
                            catch { break; }
                        }
                    }
                    isClientConnected = false;
                    currentAction = 0;
                    ThreadingHelper.Instance.ExecuteSync(() => ResetAllInputs());
                }
                catch { isClientConnected = false;
                    ThreadingHelper.Instance.ExecuteSync(() => ResetAllInputs());
                }
            }
        }

        public bool ToggleButtonInsideMenu => true;
        public List<IMenuMod.MenuEntry> GetMenuData(IMenuMod.MenuEntry? t)
        {
            return new List<IMenuMod.MenuEntry>
            {
                new IMenuMod.MenuEntry { Name = "Mode", Values = new string[] { "Normal", "Training" }, Saver = opt => settings.IsTrainingMode = opt == 1, Loader = () => settings.IsTrainingMode ? 1 : 0 },
                new IMenuMod.MenuEntry { Name = "Response Mode", Values = new string[] { "Async", "OnDemand" }, Saver = opt => settings.IsStepResponseMode = opt == 1, Loader = () => settings.IsStepResponseMode ? 1 : 0 },
                // [NEW] 时间流速 UI
                new IMenuMod.MenuEntry { Name = "Time Scale", Values = timeScaleNames, Saver = opt => { settings.TimeScale = timeScaleValues[opt]; ApplyTimeScaleSettings(); }, Loader = () => Array.IndexOf(timeScaleValues, settings.TimeScale) >= 0 ? Array.IndexOf(timeScaleValues, settings.TimeScale) : 1 },
                // [NEW] 帧率 UI
                new IMenuMod.MenuEntry { 
                    Name = "Target FPS", 
                    Values = fpsNames, 
                    Saver = opt => { 
                        settings.TargetFPS = fpsValues[opt];
                        ApplyFpsSettings(); 
                    }, 
                    Loader = () => Array.IndexOf(fpsValues, settings.TargetFPS) >= 0 ? 
                    Array.IndexOf(fpsValues, settings.TargetFPS) : 2 
                },

                new IMenuMod.MenuEntry { Name = "Target Scene", Values = scenePresets, Saver = opt => { sceneIndex = opt; settings.TargetScene = scenePresets[sceneIndex]; }, Loader = () => { int idx = Array.IndexOf(scenePresets, settings.TargetScene); return idx >= 0 ? idx : 0; } },
                new IMenuMod.MenuEntry { Name = "Resolution", Values = resNames, Saver = opt => { settings.ResolutionIndex = opt; UpdateResolutionVariables(); }, Loader = () => settings.ResolutionIndex },
                new IMenuMod.MenuEntry { Name = "Sampling", Values = sampleNames, Saver = opt => { settings.FrameSkip = opt + 1; }, Loader = () => settings.FrameSkip - 1 },
                new IMenuMod.MenuEntry { Name = "Camera", Values = new string[] { "Off", "On" }, Saver = opt => settings.EnableCamera = opt == 1, Loader = () => settings.EnableCamera ? 1 : 0 },
                new IMenuMod.MenuEntry {
                Name = "Load Delay",
                    Values = delayNames,
                    Saver = opt => settings.LevelLoadDelay = delayValues[opt],
                    Loader = () => {
                        int idx = Array.IndexOf(delayValues, settings.LevelLoadDelay);
                        return idx >= 0 ? idx : 3; // 默认为 1.5s (index 3)
                    }
                },
            };
        }

        private void ResetGame()
        {
            if (GameManager.instance == null || HeroController.instance == null)
            {
                Log("ResetGame: GameManager or HeroController is null, aborting reset.");
                return;
            }
            if (isResetting)
            {
                Log("ResetGame: Already resetting, skipping.");
                return;
            }
            if (GameManager.instance.gameState == GameState.ENTERING_LEVEL || GameManager.instance.gameState == GameState.EXITING_LEVEL)
            {
                Log($"ResetGame: GameState is {GameManager.instance.gameState}, skipping reset.");
                return;
            }

            isResetting = true;
            SetPaused(false);
            Log($"Resetting Level -> {settings.TargetScene}");
            ignoreInput = true;
            isBusy = true;
            needsReset = false;
            // 确保done state在重置前被记录
            Time.timeScale = settings.TimeScale;
            Log($"Reset triggered with done state: {currentDoneState}");

            GameManager.instance.BeginSceneTransition(new GameManager.SceneLoadInfo
            {
                SceneName = settings.TargetScene,
                EntryGateName = "door_dreamEnter",
                PreventCameraFadeOut = false
            });
            GameManager.instance.StartCoroutine(ForceLevelStart());
        }

        // [NEW] 修改后的暂停逻辑
        private void SetPaused(bool pause)
        {
            isPaused = pause;
            Log($"Paused: {isPaused}, TimeScale: {(isPaused ? 0f : settings.TimeScale)}");
            // 如果暂停则为0，否则恢复到用户设置的时间流速
            Time.timeScale = isPaused ? 0f : settings.TimeScale;
        }
        private bool IsHeroReady()
        {
            if (HeroController.instance == null || GameManager.instance == null) return false;

            // 1. 基础状态检查
            if (isResetting || isBusy || isPaused) return false;
            if (GameManager.instance.gameState != GameState.PLAYING) return false;

            // 2. 核心控制器检查
            var hc = HeroController.instance;

            // acceptingInput: 游戏逻辑允许输入
            // transitionState: 必须是 WAITING_TO_ENTER_LEVEL 结束后的状态
            // actor_state: 不能是看地图、交互等状态 (通常 Idle 或 Running 是允许的)
            // controlReqlinquished: 即使 acceptingInput 为 true，剧情动画期间这个也可能是 true

            bool isControlFree = hc.acceptingInput && !hc.cState.transitioning && !hc.controlReqlinquished;

            // [额外保险] 检查是否处于“硬直”或“施法”后的锁定状态
            bool isNotBusy = !hc.cState.recoiling && !hc.cState.casting && !hc.cState.preventBackDash;

            return isControlFree && isNotBusy;
        }

        private IEnumerator ForceLevelStart()
        {
            Log("ForceLevelStart: Sequence Begin.");

            // 1. 确保时间是流动的，否则无法加载
            Time.timeScale = 1.0f;

            // 2. 等待场景淡入和状态切换 (这是Unity引擎的硬性要求，必须等)
            yield return new WaitForSecondsRealtime(0.5f);

            if (GameManager.instance != null)
            {
                GameManager.instance.FadeSceneIn();
                GameManager.instance.SetState(GameState.PLAYING);
            }

            // 确保进入 PLAYING 状态
            while (GameManager.instance.gameState != GameState.PLAYING)
                yield return null;

            // 3. [核心修改] 硬等待！
            // 不再去检测什么 acceptingInput 或者 actor_state
            // 直接根据设置的时间，死等。
            Log($"Waiting for user-defined delay: {settings.LevelLoadDelay} seconds...");

            // 这里使用 WaitForSeconds (受 TimeScale 影响) 还是 Realtime 看你需求
            // 建议用 Realtime，比较稳，不受游戏变速影响
            yield return new WaitForSecondsRealtime(settings.LevelLoadDelay);

            // 4. 强制接管控制权
            if (HeroController.instance != null)
            {
                // 强制停止任何可能残留的动画（比如起身动作）
                HeroController.instance.StopAnimationControl();
                HeroController.instance.RegainControl();

                // 强制落地状态，防止第一帧被判定为在空中而放不出技能
                HeroController.instance.cState.onGround = true;
                HeroController.instance.cState.dashing = false;

                // 接受输入
                HeroController.instance.AcceptInput();
            }

            // 5. 清理之前的残留输入，干干净净开始
            ResetAllInputs();

            Log("ForceLevelStart: Timer up. Unlocking controls.");

            // 6. 解锁
            ignoreInput = false;
            isBusy = false;
            isResetting = false;
            currentDoneState = "false";
            needsReset = false;

            // 恢复训练用的时间流速
            Time.timeScale = settings.TimeScale;
        }

        private void OnSceneChanged(Scene o, Scene n)
        {
            SetPaused(false); currentDoneState = "false"; isResetting = false;
            if (settings.IsTrainingMode) { ignoreInput = true; if (GameManager.instance != null) GameManager.instance.StartCoroutine(ForceLevelStart()); }
            else ignoreInput = false;
        }

        //private int OnTakeDamage(int damage)
        //{
        //    if (!settings.IsTrainingMode) return damage;
        //    if (needsReset) return 0;
        //    if (PlayerData.instance.health - damage <= 0) { currentDoneState = "dead"; needsReset = true; return 0; }
        //    return damage;
        //}
        private volatile bool isBusy = false;

        // 修改 OnTakeDamage 逻辑
        private int OnTakeDamage(int damage)
        {
            if (!settings.IsTrainingMode) return damage;
            if (needsReset) return 0; // 已经死透了，不再计算伤害

            int projectedHealth = PlayerData.instance.health - damage;
            bool isDead = projectedHealth <= 0;

            // 如果开启了“受伤即重置” 或者 真的没血了
            if (settings.ResetOnDamage || isDead)
            {
                currentDoneState = "dead";
                needsReset = true;

                // [关键优化] 立即冻结游戏时间！
                // 这样无论 Python 延迟多少毫秒，游戏状态都停留在“死亡瞬间”，不会继续播放动画或掉落
                if (!isPaused)
                {
                    Time.timeScale = 0f;
                }
                return 0; // 阻止扣血，防止触发游戏原生的死亡重生逻辑（那个太慢了）
            }

            // 正常扣血，恢复原本的时间流速（防止暂停后恢复出错）
            if (!isPaused)
            {
                Time.timeScale = settings.TimeScale;
            }

            return damage;
        }

        public void Unload()
        {
            stopServer = true;
            //Time.timeScale = 1f; // 卸载时恢复正常流速
            if (networkThread != null) try { networkThread.Abort(); } catch { }
            if (tcpListener != null) try { tcpListener.Stop(); } catch { }
            if (renderTex != null) renderTex.Release();
            UnityEngine.SceneManagement.SceneManager.activeSceneChanged -= OnSceneChanged;
            ModHooks.HeroUpdateHook -= OnHeroUpdate;
            ModHooks.TakeHealthHook -= OnTakeDamage;
            On.InputHandler.Update -= InputHandler_Update_Hook;
            On.HeroController.LookForInput -= HC_LookForInput_Hook;
        }

        private void OnHeroUpdate()
        {
            if (!settings.IsTrainingMode) return;

            //if (!isPaused && Math.Abs(Time.timeScale - settings.TimeScale) > 0.01f)
            //{
            //    // 再次强制应用
            //    Time.timeScale = settings.TimeScale;
            //}

            if (!needsReset && !isResetting)
            {
                bool victory = CheckVictoryCondition();
                if (victory)
                {
                    Log("VIC");
                    currentDoneState = "victory";
                    needsReset = true;
                    Log($"Victory condition met! needsReset set to {needsReset}");
                }
            }

            if (needsReset && !isResetting)
            {
                Log($"Attempting reset... needsReset: {needsReset}, isResetting: {isResetting}");
                ResetGame();
            }
        }

        // 定义已知存在多Boss或明显转场空窗期的场景
        private readonly HashSet<string> MultiBossScenes = new HashSet<string>
        {
            "GG_Mantis_Lords",      // 螳螂领主 / 战斗姐妹
            "GG_Watcher_Knights",   // 守望者骑士 (有非常长的空窗期)
            "GG_Oblobbles",         // 奥波路堡 (两只)
            "GG_Nailmasters",       // 奥罗 & 马托
            "GG_Grimm_Nightmare",   // 梦魇格林 (有时会有分身阶段，保险起见)
            "GG_Sly",               // 斯莱 (有二阶段转场)
            "GG_Radiance",          // 辐光 (有多阶段转场)
            "GG_Hollow_Knight"      // 纯粹容器 (如果是普通空洞可能有阶段，纯粹容器通常还好，但保险起见)
        };

        private float victoryConfirmationTimer = 0f;
        private bool hasBossesBeenDetected = false; // 新增：记录是否曾经检测到Boss

        private bool CheckVictoryCondition()
        {
            // 0. 基础检查
            if (!settings.TargetScene.StartsWith("GG_")) return false;

            // 刚进图的前5秒绝对不判胜 (给Boss生成留更多时间)
            if (UnityEngine.Time.timeSinceLevelLoad < 5.0f)
            {
                victoryConfirmationTimer = 0f;
                hasBossesBeenDetected = false; // 重置检测状态
                return false;
            }

            // --- 方法 1: 官方控制器检查 ---
            if (bossSceneControllerType != null)
            {
                var instanceProp = bossSceneControllerType.GetProperty("Instance", BindingFlags.Public | BindingFlags.Static);
                var instance = instanceProp?.GetValue(null, null);

                if (instance != null && bossesDeadProp != null)
                {
                    bool bossesDead = (bool)bossesDeadProp.GetValue(instance, null);
                    if (bossesDead)
                    {
                        Log("Victory: BossSceneController confirmed (Immediate).");
                        hasBossesBeenDetected = false; // 重置状态
                        return true;
                    }
                }
            }

            // --- 方法 2: 改进的HP扫描 ---
            HealthManager[] healthManagers = UnityEngine.Object.FindObjectsOfType<HealthManager>();
            bool isAnyBossAlive = false;
            bool foundAnyBossObjectThisFrame = false;

            foreach (HealthManager hm in healthManagers)
            {
                if (hm.gameObject == null || !hm.gameObject.activeInHierarchy) continue;

                string name = hm.gameObject.name.ToLower();
                int layer = hm.gameObject.layer;

                // Boss识别条件
                if (name.Contains("boss") ||
                    name.Contains("mantis") ||
                    name.Contains("sister") ||
                    name.Contains("grimm") ||
                    name.Contains("hornet") ||
                    name.Contains("hollow") ||
                    name.Contains("radiance") ||
                    name.Contains("brother") ||
                    name.Contains("watcher") ||
                    name.Contains("knight") ||
                    name.Contains("defender") ||
                    name.Contains("galien") ||
                    name.Contains("zote") ||
                    (layer == 11 && hm.hp > 50))
                {
                    foundAnyBossObjectThisFrame = true;

                    // 如果发现Boss对象，标记为已检测到Boss
                    if (!hasBossesBeenDetected)
                    {
                        hasBossesBeenDetected = true;
                        victoryConfirmationTimer = 0f; // 重置计时器
                        Log($"Boss detected: {hm.gameObject.name}");
                    }

                    // 检查是否存活
                    if (hm.hp > 0 && !hm.isDead)
                    {
                        isAnyBossAlive = true;
                        break;
                    }
                }
            }

            // 关键修改：只有在曾经检测到Boss的情况下才进行胜利判断
            if (isAnyBossAlive)
            {
                // 场上有活着的Boss -> 正在战斗
                victoryConfirmationTimer = 0f;
                return false;
            }
            else if (hasBossesBeenDetected) // 只有曾经检测到Boss才进入胜利判断
            {
                // 曾经检测到Boss，但现在没有活着的Boss -> 可能胜利了
                victoryConfirmationTimer += UnityEngine.Time.unscaledDeltaTime;

                // 延长确认时间到5-6秒，确保转场动画完成
                if (victoryConfirmationTimer > 5.0f)
                {
                    Log($"Victory: No alive boss detected for {victoryConfirmationTimer:F1}s after boss detection. Resetting.");
                    hasBossesBeenDetected = false; // 重置状态
                    return true;
                }
            }
            else
            {
                // 从未检测到Boss -> 可能是Boss还没生成，保持等待状态
                victoryConfirmationTimer = 0f;
            }

            return false;
        }

        private string GetRaycastData(Vector2 origin)
        {
            Vector2[] dirs = { Vector2.right, Vector2.left, Vector2.up, Vector2.down, new Vector2(1, 1), new Vector2(1, -1), new Vector2(-1, 1), new Vector2(-1, -1) };
            StringBuilder sb = new StringBuilder(); sb.Append("[");
            for (int i = 0; i < dirs.Length; i++)
            {
                RaycastHit2D hit = Physics2D.Raycast(origin, dirs[i], 5.0f, terrainLayerMask);
                sb.Append((hit.collider != null ? hit.distance : 5.0f).ToString("F2"));
                if (i < dirs.Length - 1) sb.Append(",");
            }
            sb.Append("]"); return sb.ToString();
        }

        private bool HasComponentInParent(GameObject go, string typeName)
        {
            Transform t = go.transform;
            while (t != null)
            {
                if (t.GetComponent(typeName) != null) return true;
                t = t.parent;
            }
            return false;
        }

        // [新增] 辅助函数：泛型查找 (保留给 HealthManager 等已有引用的类使用)
        private T GetComponentInParentRecursive<T>(GameObject go) where T : Component
        {
            if (go == null) return null;
            Transform t = go.transform;
            while (t != null)
            {
                T comp = t.GetComponent<T>();
                if (comp != null) return comp;
                t = t.parent;
            }
            return null;
        }

        // [MODIFIED] 终极分类：地形、主角、攻击、敌人(AI)、可破坏物(无AI)、弹幕、静态陷阱
        // [修复版] 碰撞体分类逻辑
        // 修复思路：威胁优先。先检测是否伤害主角，再检测是否是主角的攻击。
        private void AppendCameraVisibleColliders(StringBuilder sb)
        {
            Camera cam = Camera.main;
            if (cam == null)
            {
                sb.Append("\"hero\":[],\"terrain\":[],\"hero_attacks\":[],\"enemies\":[],\"destructibles\":[],\"enemy_attacks\":[],\"traps\":[]");
                return;
            }

            float heroZ = 0.004f;
            if (HeroController.instance != null) heroZ = HeroController.instance.transform.position.z;
            float distance = 50f;

            Vector3 minScreen = cam.ViewportToWorldPoint(new Vector3(0, 0, distance));
            Vector3 maxScreen = cam.ViewportToWorldPoint(new Vector3(1, 1, distance));

            Collider2D[] cols = Physics2D.OverlapAreaAll(minScreen, maxScreen);

            List<string> terrainList = new List<string>();
            List<string> heroList = new List<string>();
            List<string> heroAttackList = new List<string>();
            List<string> enemyList = new List<string>();
            List<string> destructibleList = new List<string>();
            List<string> enemyAttackList = new List<string>();
            List<string> trapList = new List<string>();

            foreach (var col in cols)
            {
                if (col == null || !col.isActiveAndEnabled) continue;

                GameObject go = col.gameObject;
                string name = go.name.ToLower();
                int layer = go.layer;

                // --- 1. 黑名单过滤 (保持不变) ---
                if (name.Contains("cameralock") || name.Contains("scene_border") ||
                    name.Contains("room_bounds") || name.Contains("load") ||
                    name.Contains("haz_respawn") || name.Contains("audio") ||
                    name.Contains("music") || name.Contains("atmos") ||
                    name.Contains("hero detection") ||
                    name.Contains("check") || name.Contains("range") ||
                    name.Contains("door") || name.Contains("region"))
                {
                    continue;
                }

                string shapeJson = GetColliderShapeData(cam, col);
                if (string.IsNullOrEmpty(shapeJson)) continue;

                // --- 2. 获取组件 (关键修改：递归查找父级) ---
                // 很多伤害判定挂在父物体上，比如 Boss 的攻击判定框
                var damageHero = GetComponentInParentRecursive<DamageHero>(go);
                var damageEnemies = GetComponentInParentRecursive<DamageEnemies>(go); // 同样递归查找
                var healthManager = GetComponentInParentRecursive<HealthManager>(go);

                // 检查是否是主角自己
                bool isHero = (layer == 9) || (go == HeroController.instance.gameObject);

                // --- 3. 分类逻辑 (优先级调整) ---

                // [P1] 地形 (Terrain) - 最高优先级，静态环境
                if (layer == 8 || layer == 30)
                {
                    terrainList.Add(shapeJson);
                    continue;
                }

                // [P2] 主角 (Hero)
                if (isHero)
                {
                    heroList.Add(shapeJson);
                    continue;
                }

                // [P3] 威胁判定 (Threats) - 只要有 DamageHero，就是坏东西
                // 必须在判断 "Hero Attack" 之前判断这个！
                if (damageHero != null)
                {
                    // 细分威胁类型：

                    // A. 如果是 Layer 11 (Enemy)，且有血条 -> 它是敌人本体
                    if (layer == 11 && healthManager != null && !healthManager.isDead)
                    {
                        enemyList.Add(shapeJson);
                    }
                    // B. 如果是 Layer 11 但没血条 (比如飞针、无敌的怪)，或者 Layer 12/22 等 -> 视为敌人攻击/陷阱
                    else
                    {
                        // 尝试区分 "攻击(动态)" 和 "陷阱(静态)"
                        // 简单的区分方法：看是否有刚体且非静态，或者看名字
                        var rb = GetComponentInParentRecursive<Rigidbody2D>(go);
                        bool isDynamic = (rb != null && rb.bodyType != RigidbodyType2D.Static);

                        // 很多Boss技能其实是 Kinematic 的刚体，所以这里放宽一点
                        // 只要不是纯静态陷阱，都算 Enemy Attack
                        if (isDynamic || name.Contains("shot") || name.Contains("ball") || name.Contains("attack"))
                        {
                            enemyAttackList.Add(shapeJson);
                        }
                        else
                        {
                            trapList.Add(shapeJson);
                        }
                    }
                    continue; // 既然伤害主角，处理完直接跳过，绝不可能是主角攻击
                }

                // [P4] 主角攻击 (Hero Attacks)
                // 条件：不能伤害主角 (前面P3已过滤) AND (Layer 17 OR 有DamageEnemies组件)
                // Layer 17 是 "Hero Attack" 层，比如法术
                if (layer == 17 || damageEnemies != null)
                {
                    heroAttackList.Add(shapeJson);
                    continue;
                }

                // [P5] 可破坏物/中立物体 (Destructibles)
                // 既不伤人，也不是主角攻击，但有物理碰撞
                if (GetComponentInParentRecursive<Breakable>(go) != null)
                {
                    destructibleList.Add(shapeJson);
                    continue;
                }

                // [P6] 兜底逻辑
                // 如果是 Layer 11 (敌人层) 但居然没有 DamageHero (可能此时没激活判定)，依然算作敌人以防万一
                if (layer == 11)
                {
                    enemyList.Add(shapeJson);
                    continue;
                }

                // 剩下的非 Trigger 的层归为地形
                if (!col.isTrigger)
                {
                    terrainList.Add(shapeJson);
                }
            }

            sb.Append("\"hero\":[" + string.Join(",", heroList.ToArray()) + "],");
            sb.Append("\"terrain\":[" + string.Join(",", terrainList.ToArray()) + "],");
            sb.Append("\"hero_attacks\":[" + string.Join(",", heroAttackList.ToArray()) + "],");
            sb.Append("\"enemies\":[" + string.Join(",", enemyList.ToArray()) + "],");
            sb.Append("\"destructibles\":[" + string.Join(",", destructibleList.ToArray()) + "],");
            sb.Append("\"enemy_attacks\":[" + string.Join(",", enemyAttackList.ToArray()) + "],");
            sb.Append("\"traps\":[" + string.Join(",", trapList.ToArray()) + "]");
        }

        private string JsonFloat(float f)
        {
            return f.ToString("F3", CultureInfo.InvariantCulture);
        }
        // [DEBUG VERSION] 形状获取逻辑
        // [FIXED] 增加了 Z-Plane Flattening，解决透视畸变问题
        // [STANDARD] 标准版：不做任何 Z 轴黑魔法，防止图形炸裂
        private string GetColliderShapeData(Camera cam, Collider2D col)
        {
            try
            {
                if (col == null) return "";

                if (col is CircleCollider2D circleCol)
                {
                    Vector2 worldCenter = col.transform.TransformPoint(circleCol.offset);
                    Vector3 vpCenter = cam.WorldToViewportPoint(worldCenter);

                    // 获取最大缩放比例
                    float maxScale = Mathf.Max(Mathf.Abs(col.transform.lossyScale.x), Mathf.Abs(col.transform.lossyScale.y));
                    float worldRadius = circleCol.radius * maxScale;

                    // [修复] 分别计算水平和垂直方向的屏幕投影半径

                    // 1. 向右偏移计算 rx
                    Vector3 vpRight = cam.WorldToViewportPoint(worldCenter + new Vector2(worldRadius, 0));
                    float radiusW = Mathf.Abs(vpRight.x - vpCenter.x);

                    // 2. 向上偏移计算 ry (之前这里是用 vpRight 计算的，导致 ry=0)
                    Vector3 vpUp = cam.WorldToViewportPoint(worldCenter + new Vector2(0, worldRadius));
                    float radiusH = Mathf.Abs(vpUp.y - vpCenter.y);

                    return $"{{\"type\":\"circle\",\"cx\":{JsonFloat(vpCenter.x)},\"cy\":{JsonFloat(vpCenter.y)},\"rx\":{JsonFloat(radiusW)},\"ry\":{JsonFloat(radiusH)}}}";
                }
                else
                {
                    List<Vector2> worldPoints = new List<Vector2>();

                    if (col is BoxCollider2D box)
                    {
                        Vector2 size = box.size;
                        Vector2 offset = box.offset;
                        Vector2 p1 = offset + new Vector2(-size.x, -size.y) * 0.5f;
                        Vector2 p2 = offset + new Vector2(size.x, -size.y) * 0.5f;
                        Vector2 p3 = offset + new Vector2(size.x, size.y) * 0.5f;
                        Vector2 p4 = offset + new Vector2(-size.x, size.y) * 0.5f;
                        worldPoints.Add(col.transform.TransformPoint(p1));
                        worldPoints.Add(col.transform.TransformPoint(p2));
                        worldPoints.Add(col.transform.TransformPoint(p3));
                        worldPoints.Add(col.transform.TransformPoint(p4));
                    }
                    else if (col is PolygonCollider2D poly)
                    {
                        foreach (var p in poly.points) worldPoints.Add(col.transform.TransformPoint(p + poly.offset));
                    }
                    else if (col is EdgeCollider2D edge)
                    {
                        foreach (var p in edge.points) worldPoints.Add(col.transform.TransformPoint(p + edge.offset));
                    }
                    else if (col is CapsuleCollider2D capsule)
                    {
                        Vector2 size = capsule.size;
                        Vector2 offset = capsule.offset;
                        worldPoints.Add(col.transform.TransformPoint(offset + new Vector2(-size.x, -size.y) * 0.5f));
                        worldPoints.Add(col.transform.TransformPoint(offset + new Vector2(size.x, -size.y) * 0.5f));
                        worldPoints.Add(col.transform.TransformPoint(offset + new Vector2(size.x, size.y) * 0.5f));
                        worldPoints.Add(col.transform.TransformPoint(offset + new Vector2(-size.x, size.y) * 0.5f));
                    }

                    if (worldPoints.Count == 0) return "";

                    StringBuilder ptsSb = new StringBuilder();
                    ptsSb.Append("[");
                    for (int i = 0; i < worldPoints.Count; i++)
                    {
                        Vector3 vp = cam.WorldToViewportPoint(worldPoints[i]);
                        ptsSb.Append($"{JsonFloat(vp.x)},{JsonFloat(vp.y)}");
                        if (i < worldPoints.Count - 1) ptsSb.Append(",");
                    }
                    ptsSb.Append("]");

                    return $"{{\"type\":\"poly\",\"pts\":{ptsSb.ToString()}}}";
                }
            }
            catch { return ""; }
        }

        // [NEW] 完整的配置和状态信息回传
        private string GetCurrentStateJSON()
        {
            try
            {
                var hero = HeroController.instance; if (hero == null) return "{}";
                var pd = PlayerData.instance; var rb = hero.GetComponent<Rigidbody2D>();
                float dashCD = 0f; if (dashCooldownField != null) dashCD = (float)dashCooldownField.GetValue(hero);
                float shadowDashTimer = 0f; if (shadowDashTimerField != null) shadowDashTimer = (float)shadowDashTimerField.GetValue(hero);

                GameObject bossObj = null; int maxHP = -1; HealthManager bossHM = null;
                foreach (HealthManager hm in UnityEngine.Object.FindObjectsOfType<HealthManager>())
                {
                    if (hm.hp > maxHP && !hm.isDead && hm.gameObject.layer == 11)
                    {
                        maxHP = hm.hp; bossObj = hm.gameObject; bossHM = hm;
                    }
                }
                StringBuilder sb = new StringBuilder();

                //bool isReady = !isBusy && !isPaused && !isResetting;
                bool isReady = IsHeroReady();

                sb.Append("{");

                // --- 基础信息 ---
                sb.Append($"\"ready\":{isReady.ToString().ToLower()},");
                sb.Append($"\"mode\":\"{(settings.IsTrainingMode ? "training" : "normal")}\",");
                sb.Append($"\"paused\":{isPaused.ToString().ToLower()},");
                sb.Append($"\"done\":\"{currentDoneState}\",");

                // --- [NEW] Config 回传所有配置 ---
                sb.Append("\"config\":{");
                sb.Append($"\"time_scale\":{settings.TimeScale:F2},");
                sb.Append($"\"target_fps\":{settings.TargetFPS},");
                sb.Append($"\"frame_skip\":{settings.FrameSkip},");
                sb.Append($"\"resolution\":\"{captureWidth}x{captureHeight}\",");
                sb.Append($"\"response_mode\":{(settings.IsStepResponseMode ? "\"on_demand\"" : "\"async\"")},");
                sb.Append($"\"target_scene\":\"{settings.TargetScene}\"");
                sb.Append("},");

                // --- 英雄信息 ---
                sb.Append("\"hero\":{");
                sb.Append($"\"hp\":{pd.health},");
                sb.Append($"\"max_hp\":{pd.maxHealth},");
                sb.Append($"\"soul\":{pd.MPCharge},");
                sb.Append($"\"pos\":[{hero.transform.position.x:F2},{hero.transform.position.y:F2}],");
                sb.Append($"\"vel\":[{(rb != null ? rb.velocity.x : 0):F2},{(rb != null ? rb.velocity.y : 0):F2}],");
                sb.Append($"\"facing\":{(hero.cState.facingRight ? 1 : -1)},");
                sb.Append($"\"on_ground\":{hero.cState.onGround.ToString().ToLower()},");
                sb.Append($"\"on_wall\":{hero.cState.touchingWall.ToString().ToLower()},");
                sb.Append($"\"can_dash\":{(dashCD <= 0.1f).ToString().ToLower()},");
                sb.Append($"\"shadow_timer\":{shadowDashTimer:F2},");
                sb.Append($"\"has_shadow_dash\":{pd.hasShadowDash.ToString().ToLower()},");
                sb.Append($"\"is_recoiling\":{hero.cState.recoiling.ToString().ToLower()},");
                sb.Append($"\"can_double_jump\":{(!hero.cState.doubleJumping && pd.hasDoubleJump).ToString().ToLower()},");
                sb.Append($"\"is_invincible\":{hero.cState.invulnerable.ToString().ToLower()},");
                sb.Append($"\"is_healing\":{hero.cState.focusing.ToString().ToLower()},");
                sb.Append($"\"nail_damage\":{pd.nailDamage},");
                sb.Append($"\"lidar\":{GetRaycastData(hero.transform.position)}");
                sb.Append("},");

                // --- Boss信息 ---
                sb.Append("\"boss\":{");
                if (bossObj != null)
                {
                    var bRb = bossObj.GetComponent<Rigidbody2D>();
                    float distX = bossObj.transform.position.x - hero.transform.position.x;
                    float distY = bossObj.transform.position.y - hero.transform.position.y;

                    sb.Append($"\"exists\":true,");
                    sb.Append($"\"name\":\"{bossObj.name}\",");
                    sb.Append($"\"hp\":{bossHM.hp},");
                    sb.Append($"\"pos\":[{bossObj.transform.position.x:F2},{bossObj.transform.position.y:F2}],");
                    sb.Append($"\"vel\":[{(bRb != null ? bRb.velocity.x : 0):F2},{(bRb != null ? bRb.velocity.y : 0):F2}],");
                    sb.Append($"\"rel_pos\":[{distX:F2},{distY:F2}]");
                }
                else { sb.Append($"\"exists\":false"); }
                sb.Append("},");

                // --- 5. [新增] 摄像机视野内的所有碰撞体信息 ---
                // 这里会调用下面新增的函数
                sb.Append("\"camera_colliders\":{");
                AppendCameraVisibleColliders(sb);
                sb.Append("}");

                sb.Append("}");
                return sb.ToString();
            }
            catch { return "{}"; }
        }

        private void InitReflection()
        {
            var t = typeof(OneAxisInputControl);
            stateField = t.GetField("thisState", BindingFlags.NonPublic | BindingFlags.Instance) ?? t.GetField("state", BindingFlags.NonPublic | BindingFlags.Instance);
            lastStateField = t.GetField("lastState", BindingFlags.NonPublic | BindingFlags.Instance) ?? t.GetField("m_LastState", BindingFlags.NonPublic | BindingFlags.Instance);
            if (stateField != null) boolStateField = stateField.FieldType.GetField("State", BindingFlags.Public | BindingFlags.Instance);
            var ht = typeof(HeroController);
            dashCooldownField = ht.GetField("dashCooldownTimer", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
            airDashedField = ht.GetField("airDashed", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
            shadowDashTimerField = ht.GetField("shadowDashTimer", BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
            bossSceneControllerType = Type.GetType("BossSceneController, Assembly-CSharp");
            if (bossSceneControllerType != null) 
                bossesDeadProp = bossSceneControllerType.GetProperty("BossesDead", BindingFlags.Public | BindingFlags.Instance) ?? bossSceneControllerType.GetProperty("IsBossDead", BindingFlags.Public | BindingFlags.Instance);
            reflectionInitialized = true;
        }

        private void SendEventToSpellControl(HeroController h, string e) {
            try { 
                var f = typeof(HeroController).GetField("spellControl"); 
                if (f == null) return; 
                var i = f.GetValue(h); 
                if (i == null) return; 
                if (fsmSendEventMethod == null) 
                    fsmSendEventMethod = i.GetType().GetMethod("SendEvent", new Type[] 
                    { typeof(string) }); 
                if (fsmSendEventMethod != null) 
                    fsmSendEventMethod.Invoke(i, new object[] { e }); 
            } catch { } 
        }
        private void HC_LookForInput_Hook(On.HeroController.orig_LookForInput orig, HeroController self)
        {
            orig(self);
            if (isPaused) return;

            if (ignoreInput)
            {
                self.move_input = 0f;
                self.vertical_input = 0f;
                return;
            }

            if (isClientConnected && settings.IsTrainingMode && !ignoreInput)
            {
                if (InputHandler.Instance != null)
                {
                    ApplyInputToActions(InputHandler.Instance.inputActions, self);
                }

                if (useMultiBinary)
                {
                    // 1. 获取方向键状态
                    bool up = multiBinaryActions[0];   // W
                    bool left = multiBinaryActions[1]; // A
                    bool down = multiBinaryActions[2]; // S
                    bool right = multiBinaryActions[3];// D

                    // 2. 计算水平轴 (同时按左右则抵消为0)
                    float h = 0f;
                    if (right) h += 1f;
                    if (left) h -= 1f;
                    self.move_input = h;

                    // 3. 计算垂直轴 (这对下劈至关重要)
                    float v = 0f;
                    if (up) v += 1f;
                    if (down) v -= 1f;
                    self.vertical_input = v;
                }
                else
                {
                    if (currentAction == 1)
                        self.move_input = -1f;
                    else if (currentAction == 2)
                        self.move_input = 1f;
                    else if (currentAction == 3)
                        self.vertical_input = 1f;
                    else if (currentAction == 4)
                        self.vertical_input = -1f;
                    else if (currentAction >= 7 && currentAction <= 10) // 新的攻击动作
                        HandleAttackWithDirection(self, currentAction);
                    else if (currentAction >= 11 && currentAction <= 14) // 修改后的法术动作
                        HandleSpellFSM(self, currentAction);
                }
            }
        }
        private void HandleSpellFSM(HeroController h, int a) {
            if (a == 13) { 
                h.vertical_input = 1f;
                OverrideButton(InputHandler.Instance.inputActions.up, true);
            } else if (a == 14) { 
                h.vertical_input = -1f; 
                OverrideButton(InputHandler.Instance.inputActions.down, true); 
            } else if (Mathf.Abs(h.vertical_input) > 0.1f) 
                h.vertical_input = 0f; 
            spellCooldownCounter++; 
            if (spellCooldownCounter >= 3) {
                spellCooldownCounter = 0; 
                SendEventToSpellControl(h, "QUICK CAST"); 
            } 
        }
        private void InputHandler_Update_Hook(On.InputHandler.orig_Update orig, InputHandler self) {
            orig(self);

            if (ignoreInput)
            {
                // 只有在重置阶段，我们才强制清空输入，防止“幽灵移动”
                ResetAllInputs();
            }
            else if (isClientConnected && settings.IsTrainingMode && !ignoreInput 
                && reflectionInitialized && !isPaused)
            {
                //ApplyInputToActions(self.inputActions, null);
                var actions = self.inputActions;
                if (useMultiBinary)
                {
                    // 1. 普通按键覆盖 (OverrideButton 是你现有的函数)
                    // Bit 4: Attack, 5: Jump, 6: Dash
                    if (multiBinaryActions[4])
                    {
                        OverrideButton(actions.attack, multiBinaryActions[4]);
                    }
                    if (multiBinaryActions[5])
                    {
                        OverrideButton(actions.jump, multiBinaryActions[5]);
                    }
                    if (multiBinaryActions[6])
                    {
                        OverrideButton(actions.dash, multiBinaryActions[6]);
                    }

                    // 2. [关键] 施法逻辑 (Bit 7)
                    // 复用你的 FSM 逻辑，而不是简单按下按钮

                    bool up = multiBinaryActions[0];   // W
                    bool down = multiBinaryActions[2]; // S
                    bool no_up_down = (up && down) || (!up && !down);

                    if (multiBinaryActions[7])
                    {
                        if (no_up_down)
                        {
                            HeroController.instance.vertical_input = 0f;
                        }
                        else if (down)
                        {
                            HeroController.instance.vertical_input = -1f;
                            OverrideButton(actions.down, true);
                        }
                        else if (up)
                        {
                            HeroController.instance.vertical_input = 1f;
                            OverrideButton(actions.up, true);
                        }
                        spellCooldownCounter++;
                        if (spellCooldownCounter >= 3)
                        {
                            spellCooldownCounter = 0;
                            // [重要] 方向已经在 HC_LookForInput_Hook 里设定好了
                            // 所以这里不需要再手动 set vertical_input，直接发事件即可
                            SendEventToSpellControl(HeroController.instance, "QUICK CAST");
                        }
                    }
                    else
                    {
                        // 松开时重置
                        OverrideButton(actions.quickCast, false);
                        spellCooldownCounter = 0;
                    }
                }
                else ApplyBasicActions(self.inputActions);
            }
        }
        private void HandleAttackWithDirection(HeroController h, int action)
        {
            // 先确保角色面向正确的方向
            switch (action)
            {
                case 7: // 上劈
                        // 确保角色面向上方
                    h.vertical_input = 1f;
                    OverrideButton(InputHandler.Instance.inputActions.attack, false);
                    break;
                case 8: // 下劈
                        // 确保角色面向下方
                    h.vertical_input = -1f;
                    OverrideButton(InputHandler.Instance.inputActions.attack, false);
                    break;
                case 9: // 向左走劈
                        // 确保角色面向左方
                    h.move_input = -1f;
                    OverrideButton(InputHandler.Instance.inputActions.attack, false);
                    break;
                case 10: // 向右走劈
                         // 确保角色面向右方
                    h.move_input = 1f;
                    OverrideButton(InputHandler.Instance.inputActions.attack, false);
                    break;
            }
        }

        private void ApplyInputToActions(HeroActions actions, HeroController hc_instance = null)
        {
            if (useMultiBinary)
            {
                // MultiBinary 模式
                // Bit 0-3: Directions (W, A, S, D)
                bool up = multiBinaryActions[0];
                bool left = multiBinaryActions[1];
                bool down = multiBinaryActions[2];
                bool right = multiBinaryActions[3];

                if (up) OverrideButton(actions.up, true);
                if (left) OverrideButton(actions.left, true);
                if (down) OverrideButton(actions.down, true);
                if (right) OverrideButton(actions.right, true);

                // Bit 4-7: Actions
                if (multiBinaryActions[4]) OverrideButton(actions.attack, true);
                if (multiBinaryActions[5]) OverrideButton(actions.jump, true);
                if (multiBinaryActions[6]) OverrideButton(actions.dash, true);

                // Bit 7: Spell (复杂逻辑)
                if (multiBinaryActions[7] && hc_instance != null)
                {
                    // 如果 hc_instance 存在，我们可以处理施法方向
                    // 但基本的按钮状态由 OverrideButton 处理
                    OverrideButton(actions.quickCast, true);
                    // 注意：具体的施法方向逻辑(spellCooldownCounter)在 Update 里处理即可，
                    // 这里主要确保按钮是按下的，让动画播放。
                }
            }
            else
            {
                // 旧版离散模式
                ApplyBasicActions(actions);
            }
        }

        private void ApplyBasicActions(HeroActions a) {
            switch (currentAction) 
            { 
                case 1: 
                    OverrideButton(a.left, true);
                    break; 
                case 2: 
                    OverrideButton(a.right, true); 
                    break; 
                case 3: 
                    OverrideButton(a.up, true); 
                    break; 
                case 4: 
                    OverrideButton(a.down, true); 
                    break; 
                case 5: 
                    OverrideButton(a.jump, true); 
                    break; 
                case 6: 
                    OverrideButton(a.attack, false); 
                    break; 
                case 11: 
                    OverrideButton(a.dash, true); 
                    break; 
            } 
        }

        private void ReleaseButton(PlayerAction a)
        {
            if (a == null) return;
            try
            {
                var s = stateField.GetValue(a);
                boolStateField.SetValue(s, false); // <--- 这里设为 false
                stateField.SetValue(a, s);

                // 同时把 LastState 也设为 false，防止逻辑混乱
                if (lastStateField != null)
                {
                    var l = lastStateField.GetValue(a);
                    boolStateField.SetValue(l, false);
                    lastStateField.SetValue(a, l);
                }
            }
            catch { }
        }

        // 重置所有输入状态
        private void ResetAllInputs()
        {
            if (InputHandler.Instance == null) return;
            var actions = InputHandler.Instance.inputActions;

            ReleaseButton(actions.left);
            ReleaseButton(actions.right);
            ReleaseButton(actions.up);
            ReleaseButton(actions.down);
            ReleaseButton(actions.jump);
            ReleaseButton(actions.dash);
            ReleaseButton(actions.attack);
            ReleaseButton(actions.superDash); // 最好把这几个也加上
            ReleaseButton(actions.dreamNail);

            //Log("All inputs forced to RELEASE state.");
        }
        private void OverrideButton(PlayerAction a, bool h) {
            if (a == null) return; 
            try { 
                var s = stateField.GetValue(a);
                boolStateField.SetValue(s, true);
                stateField.SetValue(a, s);
                if (lastStateField != null && !h) {
                    var l = lastStateField.GetValue(a);
                    boolStateField.SetValue(l, false);
                    lastStateField.SetValue(a, l); 
                } 
            } catch { } 
        }
    }

    public class ThreadingHelper : MonoBehaviour { 
        private static ThreadingHelper _instance; 
        public static ThreadingHelper Instance { 
            get { 
                if (_instance == null) { 
                    var obj = new GameObject("ThreadingHelper"); 
                    _instance = obj.AddComponent<ThreadingHelper>(); 
                    DontDestroyOnLoad(obj); 
                } return _instance; 
            } 
        } 
        private readonly Queue<Action> _actions = new Queue<Action>();
        public void ExecuteSync(Action action) { 
            lock (_actions) { 
                _actions.Enqueue(action); 
            } 
        } 
        void Update() { 
            lock (_actions) { 
                while (_actions.Count > 0) 
                    _actions.Dequeue().Invoke(); 
            } 
        } 
    }
}