from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from utils import create_env
from callbacks import TensorboardCallback, CustomEvalCallback
from models_arch import LidarCNNExtractor

import random
from torch.nn import Mish


if __name__ == "__main__":
    save_interval = 5e4
    eva_freq = 5e4
    n_eval_episodes = 20
    learn_steps = 1e7
    log_name = "sac_stage1"

    save_path = f"./models/{log_name}"
    log_dir = "./metrics/"
    maps = list(range(1, 450))

    random.seed(8)

    # SAC는 off-policy라 n_envs=1 (병렬 환경 지원 안 됨)
    env = create_env(maps=maps, seed=8, domain_randomize=False, n_envs=1)
    eval_env = create_env(maps=maps, seed=8,
                         domain_randomize=False, n_envs=1)

    # SAC/TD3는 net_arch에서 vf 대신 qf, 리스트 감싸지 않음
    policy_kwargs = dict(
        features_extractor_class=LidarCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=Mish,
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=2,
        learning_rate=3e-4,
        buffer_size=300_000,       # 리플레이 버퍼 크기
        learning_starts=10_000,    # 이 스텝 이후부터 학습 시작
        batch_size=256,
        tau=0.005,                 # 타겟 네트워크 소프트 업데이트 계수
        gamma=0.99,
        train_freq=1,              # 매 스텝마다 업데이트
        gradient_steps=1,
        ent_coef="auto",           # 엔트로피 자동 조정 (SAC 핵심 특징)
        target_update_interval=1,
        tensorboard_log=log_dir,
        device="cuda",
        policy_kwargs=policy_kwargs
    )

    callbacks = CallbackList([
        TensorboardCallback(save_interval, save_path),
        CustomEvalCallback(eval_env,
                          best_model_save_path="./best_models/sac/",
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
