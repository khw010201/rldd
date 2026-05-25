from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from utils import create_env
from callbacks import TensorboardCallback, CustomEvalCallback
from models_arch import LidarCNNExtractor
from stable_baselines3.common.vec_env import VecTransposeImage
import torch

import numpy as np
import random
from torch.nn import Mish


if __name__ == "__main__":
    save_interval = 5e4
    eva_freq = 5e4
    n_eval_episodes = 20
    learn_steps = 1e7
    log_name = "td3_stage1"

    save_path = f"./models/{log_name}"
    log_dir = "./metrics/"
    maps = list(range(1, 450))

    random.seed(8)

    from gym import spaces as gym_spaces

    env = create_env(maps=maps, seed=8, domain_randomize=False, n_envs=1)
    eval_env = create_env(maps=maps, seed=8, domain_randomize=False, n_envs=1)

    # action space float64 → float32 변환
    env.action_space = gym_spaces.Box(
        low=env.action_space.low.astype(np.float32),
        high=env.action_space.high.astype(np.float32),
        dtype=np.float32
    )
    eval_env.action_space = gym_spaces.Box(
        low=eval_env.action_space.low.astype(np.float32),
        high=eval_env.action_space.high.astype(np.float32),
        dtype=np.float32
    )

    # TD3 필수: 액션 노이즈 (탐색용)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # SAC/TD3는 net_arch에서 vf 대신 qf, 리스트 감싸지 않음
    policy_kwargs = dict(
        features_extractor_class=LidarCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=Mish,
    )

    model = TD3(
        "MlpPolicy",
        env,
        verbose=2,
        learning_rate=3e-4,
        buffer_size=300_000,       # 리플레이 버퍼 크기
        learning_starts=10_000,    # 이 스텝 이후부터 학습 시작
        batch_size=256,
        tau=0.005,                 # 타겟 네트워크 소프트 업데이트 계수
        gamma=0.99,
        train_freq=1,  # 에피소드마다 업데이트
        gradient_steps=1,         # train_freq에 맞춰 자동 계산
        action_noise=action_noise,  # TD3 필수 탐색 노이즈
        policy_delay=2,            # actor 업데이트 지연 (TD3 핵심)
        target_policy_noise=0.2,   # 타겟 정책 노이즈
        target_noise_clip=0.5,     # 타겟 노이즈 클리핑
        tensorboard_log=log_dir,
        device="cuda",
        policy_kwargs=policy_kwargs
    )

    callbacks = CallbackList([
        TensorboardCallback(save_interval, save_path),
        CustomEvalCallback(eval_env,
                          best_model_save_path="./best_models/td3/",
                          log_path=log_dir,
                          n_eval_episodes=n_eval_episodes,
                          eval_freq=eva_freq)
    ])

    model.learn(
        total_timesteps=learn_steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=log_name,
        reset_num_timesteps=True,
    )

    env.close()
