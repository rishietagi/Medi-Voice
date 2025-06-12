import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class hr_sim_env(gym.Env):
    def __init__(self, csv_path):
        super(hr_sim_env, self).__init__()
        
        self.df = pd.read_csv(csv_path)
        self.current_idx = 0

        # Map gender to numeric
        if 'gender' in self.df.columns:
            self.df['gender'] = self.df['gender'].map({'male': 0, 'female': 1, 'unknown': -1})

        # One-hot encode age
        if 'age' in self.df.columns:
            age_dummies = pd.get_dummies(self.df['age'], prefix='age')
            self.df = pd.concat([self.df.drop('age', axis=1), age_dummies], axis=1)

        # Exclude target and filename from features
        feature_cols = [col for col in self.df.columns if col not in ['filename', 'simulated_hr']]
        self.features_raw = self.df[feature_cols]  # save for reference

        # Identify columns to normalize (skip categorical: gender and age dummies)
        numeric_cols = [col for col in feature_cols if col not in ['gender'] and not col.startswith('age_')]
        
        # Normalize only numeric columns
        self.mean = self.features_raw[numeric_cols].mean()
        self.std = self.features_raw[numeric_cols].std() + 1e-8
        self.features_raw[numeric_cols] = (self.features_raw[numeric_cols] - self.mean) / self.std

        self.features = self.features_raw.values.astype(np.float32)
        self.targets = self.df['simulated_hr'].values.astype(np.float32)

        # Define observation and action space
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.features.shape[1],),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
    def _get_obs(self):
        return self.features[self.current_idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        obs = self._get_obs()
        info = {}

        # Ensure shape and dtype match observation space
        obs = np.array(obs, dtype=np.float32).reshape(self.observation_space.shape)
        return obs, info

    def step(self, action):
        terminated = False
        truncated = False
        
        true_hr = self.targets[self.current_idx]
        
        pred_hr = 0.5 * (action[0] + 1) * (160 - 50) + 50
        
        error = abs(pred_hr - true_hr)
        reward = float(-error)  
        
        self.current_idx += 1
        
        if self.current_idx >= len(self.df):
            terminated = True
            obs = np.zeros_like(self.features[0])
        else:
            obs = self.features[self.current_idx]
        
        info = {'true_hr': true_hr, 'pred_hr': pred_hr, 'error': error}
        return obs, reward, terminated, truncated, info
