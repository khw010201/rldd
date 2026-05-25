import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces


class LidarCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]
        lidar_dim = 2155
        extra_dim = input_dim - lidar_dim

        # LiDAR용 1D CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # CNN 출력 크기 계산
        with torch.no_grad():
            dummy = torch.zeros(1, 1, lidar_dim)
            cnn_out = self.cnn(dummy).shape[1]

        # 나머지 관측값용 MLP
        self.extra_net = nn.Sequential(
            nn.Linear(extra_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 합치는 레이어
        self.merge = nn.Sequential(
            nn.Linear(cnn_out + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float()  # ← 이 줄 추가
        lidar = observations[:, :2155].unsqueeze(1)
        extra = observations[:, 2155:]

        cnn_out = self.cnn(lidar)
        extra_out = self.extra_net(extra)

        merged = torch.cat([cnn_out, extra_out], dim=1)
        return self.merge(merged)
