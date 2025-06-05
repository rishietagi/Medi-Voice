import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class hr_sim_env(gym.Env):
    def __init__(self, csv_path):
        super(hr_sim_env, self).__init__()
        
        self.df = pd.read_csv(csv_path)
        self.current_idx = 0
        
        # Features exclude filename and simulated_hr
        feature_cols = [col for col in self.df.columns if col not in ['filename', 'simulated_hr']]
        if 'gender' in self.df.columns:
            self.df['gender'] = self.df['gender'].map({'male': 0, 'female': 1, 'unknown': -1})
        self.features = self.df[feature_cols].values.astype(np.float32)
        
        self.targets = self.df['simulated_hr'].values.astype(np.float32)
        
        # Normalize features to mean=0, std=1 for stable training
        self.mean = np.mean(self.features, axis=0)
        self.std = np.std(self.features, axis=0) + 1e-8  # avoid div zero
        self.features = (self.features - self.mean) / self.std
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32)
        
        # Use normalized action space [-1,1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
    def _get_obs(self):
        return self.features[self.current_idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        terminated = False
        truncated = False
        
        true_hr = self.targets[self.current_idx]
        
        # Rescale action from [-1,1] to [50,160]
        pred_hr = 0.5 * (action[0] + 1) * (160 - 50) + 50
        
        error = abs(pred_hr - true_hr)
        reward = float(-error)  # negative absolute error
        
        self.current_idx += 1
        
        if self.current_idx >= len(self.df):
            terminated = True
            obs = np.zeros_like(self.features[0])
        else:
            obs = self.features[self.current_idx]
        
        info = {'true_hr': true_hr, 'pred_hr': pred_hr, 'error': error}
        return obs, reward, terminated, truncated, info
