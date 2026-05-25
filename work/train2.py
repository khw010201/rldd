from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from utils import create_env, linear_schedule
from callbacks import TensorboardCallback, CustomEvalCallback
from models_arch import LidarCNNExtractor

import random
from torch.nn import Mish

if __name__ == "__main__":
    save_interval = 5e4
    eva_freq = 5e4
    n_eval_episodes = 20
    learn_steps = 1e7
    log_name = "cnn_stage1"

    save_path = f"./models/{log_name}"
    log_dir = "./metrics/"
    maps = list(range(1, 450))

    random.seed(8)

    # 1단계: 랜덤화 OFF
    env = create_env(maps=maps, seed=8, domain_randomize=True, n_envs=4)
    eval_env = create_env(maps=maps, seed=8,
                         domain_randomize=False, n_envs=1)

    policy_kwargs = dict(
        features_extractor_class=LidarCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=Mish,
    )

    model = PPO.load(
        "./models/cnn_stage1_4000k",
        env=env,
        tensorboard_log=log_dir,
        device="cuda",
        custom_objects={
            "learning_rate": 1e-4,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "clip_range": 0.1,
            "n_steps": 4096,      # 추가
            "n_epochs": 4,        # 추가
            "target_kl": 0.05,    # 추가 (핵심)
        }
    )
    
    callbacks = CallbackList([
        TensorboardCallback(save_interval, save_path),
        CustomEvalCallback(eval_env,
                          best_model_save_path="./best_models/cnn/",
                          log_path=log_dir,
                          n_eval_episodes=n_eval_episodes,
                          eval_freq=eva_freq)
    ])

    model.learn(
        total_timesteps=learn_steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=log_name,
        reset_num_timesteps=False,
    )

    env.close()
