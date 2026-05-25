import gym
from gym import spaces

from copy import copy
import numpy as np

from sklearn.neighbors import KDTree

NUM_BEAMS = 2155
DTYPE = np.float32

class DelayedAction(gym.Wrapper):
    def __init__(self, env, delay_prob=0.1, drop_prob=0.05):
        super().__init__(env)
        self.delay_prob = delay_prob
        self.drop_prob = drop_prob
        self.last_action = None
        self.last_executed_action = None

    def reset(self, **kwargs):
        self.last_action = None
        self.last_executed_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        # Randomly delay the action
        if self.last_action is not None and np.random.random() < self.delay_prob:
            action_to_take = self.last_action
        else:
            action_to_take = action

        # Randomly drop the action
        if np.random.random() < self.drop_prob:
            action_to_take = None

        # Remember the current action for potential delay in the next step
        self.last_action = action

        if action_to_take is None:
            # If the action was dropped, repeat the last executed action.
            # If there is no last executed action (beginning of the episode), execute a random action.
            action_to_take = self.last_executed_action if self.last_executed_action is not None else self.env.action_space.sample()

        observation, reward, done, info = self.env.step(action_to_take)
        
        # Remember the last executed action
        self.last_executed_action = action_to_take

        return observation, reward, done, info


class LidarRandomizer(gym.ObservationWrapper):
    def __init__(self, env, epsilon=0.05, zone_p=0.1, extreme_p=0.05):
        super().__init__(env)
        self.epsilon = epsilon
        self.zone_p = zone_p
        self.extreme_p = extreme_p

    def observation(self, obs):
        lidar_data = obs["scans"]
        
        # Try normal vs uniform noise
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=lidar_data.shape)
        lidar_data += noise

        # Randomly choose areas to increase/decrease.
        if np.random.random() < self.zone_p:
            # Define size of the area (20% of the readings).
            size = int(len(lidar_data) * 0.2)
            start = np.random.randint(0, len(lidar_data) - size)
            end = start + size
            # Randomly choose whether to increase or decrease, and by how much.
            change = np.random.uniform(-0.1, 0.1)
            lidar_data[start:end] += change

        # Randomly set some readings to very high or very low.
        if np.random.random() < self.extreme_p:
            index = np.random.randint(len(lidar_data))
            lidar_data[index] = np.random.choice([0, 1])

        # Make sure the output is still between 0 and 1.
        lidar_data = np.clip(lidar_data, 0, 1)
        
        
        obs["scans"] = lidar_data

        return obs
    

class ActionRandomizer(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon
    def action(self, action):
        noise = np.random.uniform(-self.epsilon, self.epsilon, size=action.shape)
        action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        return action


class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, obs):
        # s값 증가량 (실제 진행 거리)
        frenet_wrapper = self.env
        # 래퍼 체인에서 FrenetObsWrapper 찾기
        env = self.env
        while not isinstance(env, FrenetObsWrapper):
            env = env.env
        
        delta_s = env.curr_s - env.prev_s
        
        # 맵 끝에서 처음으로 돌아오는 경우 처리
        if delta_s < -50:
            delta_s = 0.0
    
        vx = np.asarray(obs["linear_vels_x"]).flatten()[0]
        d = np.asarray(obs["poses_d"]).flatten()[0]

        reward = 0.0

        # 진행 거리 보상 (핵심)
        reward += delta_s * 2.0

        # 저속 페널티
        if abs(float(vx)) <= 0.25:
            reward -= 1.0

        # 충돌 페널티
        if self.env.collisions[0]:
            reward -= 5.0

        # 중심선 이탈 페널티
        reward -= 0.01 * abs(float(d))

        reward = float(np.clip(reward, -5.0, 5.0))
        return reward

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        info['poses_s'] = obs['poses_s']
        info['collision'] = float(self.env.collisions[0])
    
    # toggle로 완주 직접 감지 (lap_count는 reset 후 0이 될 수 있음)
        base_env = self.env.unwrapped
        toggle = float(base_env.toggle_list[0])
        is_collision = bool(self.env.collisions[0])
        lap_done = (toggle >= 2) and not is_collision
    
        info['is_success'] = lap_done
        info['lap_count'] = np.array([1.0]) if lap_done else info.get('lap_count', np.array([0.0]))
    
        new_reward = self.reward(obs)

    # 완주 보상
        if lap_done:
            new_reward += 10.0

    # 충돌 시 즉시 종료
        if is_collision:
            done = True

        return obs, np.float64(new_reward), done, info


class FrenetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FrenetObsWrapper, self).__init__(env)

        self.map_data = env.map_data.to_numpy()
        self.kdtree = KDTree(self.map_data[:, 1:3])

        self.observation_space = spaces.Dict(
            {
                "ego_idx": spaces.Box(0, self.num_agents - 1, (1,), np.int32),
                "scans": spaces.Box(0, 1, (NUM_BEAMS,), DTYPE),
                "poses_x": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
                "poses_y": spaces.Box(-1000, 1000, (self.num_agents,), DTYPE),
                "poses_theta": spaces.Box(
                    -2 * np.pi, 2 * np.pi, (self.num_agents,), DTYPE
                ),
                "linear_vels_x": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "linear_vels_y": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "ang_vels_z": spaces.Box(-10, 10, (self.num_agents,), DTYPE),
                "collisions": spaces.Box(0, 1, (self.num_agents,), DTYPE),
                "lap_times": spaces.Box(0, 1e6, (self.num_agents,), DTYPE),
                "lap_counts": spaces.Box(0, 999, (self.num_agents,), np.int32),
                "poses_s": spaces.Box(-1000, 1000, (1,), DTYPE),
                "poses_d": spaces.Box(-1000, 1000, (1,), DTYPE),
                "linear_vels_s": spaces.Box(-10, 10, (1,), DTYPE),
                "linear_vels_d": spaces.Box(-10, 10, (1,), DTYPE),
                "linear_vel": spaces.Box(0, 1, (self.num_agents,), DTYPE),
            }
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # 맵이 바뀐 후 실제 F110Env에 접근
        base_env = self.env.unwrapped
        self.map_data = base_env.map_data.to_numpy()
        self.kdtree = KDTree(self.map_data[:, 1:3])
        return self.observation(obs)

    def observation(self, obs):
        new_obs = copy(obs)

        frenet_coords = convert_to_frenet(new_obs["poses_x"][0],
                                          new_obs["poses_y"][0],
                                          new_obs["linear_vels_x"], 
                                          new_obs["poses_theta"][0], 
                                          self.map_data, 
                                          self.kdtree
        )
        
        new_obs["poses_s"] = np.array(frenet_coords[0]).reshape((1, -1))
        new_obs["poses_d"] = np.array(frenet_coords[1])
        new_obs["linear_vels_s"] = np.array(frenet_coords[2]).reshape((1, -1))
        new_obs["linear_vels_d"] = np.array(frenet_coords[3])
        
        # Scale the scans and add linear_vel
        clipped_indices = np.where(new_obs["scans"] >= 10)
        noise = np.random.uniform(-0.5, 0, clipped_indices[0].shape)
        
        new_obs["scans"] = np.clip(new_obs["scans"], None, 10)
        new_obs["scans"][clipped_indices] += noise
        new_obs["scans"] /= 10.0

        new_obs["linear_vel"] = new_obs["linear_vels_x"] / 3.2
        
        self.prev_s = getattr(self, 'curr_s', 0.0)
        self.curr_s = float(np.asarray(frenet_coords[0]).flatten()[0])

        return new_obs
    
    
def get_closest_point_index(x, y, kdtree):
    # nan이나 inf 들어오면 0번 인덱스 반환
    if not np.isfinite(x) or not np.isfinite(y):
        return 0
    _, indices = kdtree.query(np.array([[x, y]]), k=1)
    closest_point_index = indices[0, 0]
    return closest_point_index

def convert_to_frenet(x, y, vel_magnitude, pose_theta, map_data, kdtree):
    # nan/inf 방어
    if not np.isfinite(x) or not np.isfinite(y):
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    closest_point_index = get_closest_point_index(x, y, kdtree)
    closest_point = map_data[closest_point_index]
    s_m, x_m, y_m, psi_rad = closest_point[0:4]

    dx = x - x_m
    dy = y - y_m

    s = -dx * np.sin(psi_rad) + dy * np.cos(psi_rad) + s_m
    d = dx * np.cos(psi_rad) + dy * np.sin(psi_rad)

    vs = vel_magnitude * np.sin(pose_theta - psi_rad)
    vd = vel_magnitude * np.cos(pose_theta - psi_rad)

    # 결과값도 방어
    s = float(s) if np.isfinite(s) else 0.0
    d = float(d) if np.isfinite(d) else 0.0

    return s, d, vs, vd
