import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np

from hr_sim_env import hr_sim_env  

env = hr_sim_env("speech_features_with_hr.csv")

check_env(env, warn=True)
n_actions = env.action_space.shape[0]


model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    batch_size=256,
    buffer_size=int(1e6),
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
)

model.learn(total_timesteps=50000)

model.save("ddpg_hr_model")

obs, info = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    print(f"True HR: {info['true_hr']:.2f}, Predicted HR: {info['pred_hr']:.2f}, Reward: {reward:.2f}")
