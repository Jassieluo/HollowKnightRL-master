def unwrap_to_base(env):
    """
    简化版：递归解包到最底层
    """
    # 处理VectorEnv
    while hasattr(env, 'envs') and env.envs:
        env = env.envs[0]

    # 处理普通包装器
    while hasattr(env, 'env') and env.env is not None and env.env is not env:
        env = env.env

    return env
