import pandas as pd
from stable_baselines3 import DDPG
from hr_sim_env import hr_sim_env
import matplotlib.pyplot as plt

env = hr_sim_env("speech_features_with_hr.csv")

model = DDPG.load("ddpg_hr_model")

obs, info = env.reset()
done = False
results = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    results.append({
        'true_hr': info['true_hr'],
        'predicted_hr': info['pred_hr'],
        'reward': reward
    })

results_df = pd.DataFrame(results)
results_df.to_csv("rl_predictions.csv", index=False)

print(results_df.describe())
print(results_df.head(10))

plt.plot(results_df['true_hr'], label='True HR')
plt.plot(results_df['predicted_hr'], label='Predicted HR')
plt.legend()
plt.grid(True)
plt.title("True vs Predicted Heart Rate")
plt.xlabel("Step")
plt.ylabel("Heart Rate (bpm)")
plt.show()
