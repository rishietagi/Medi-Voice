import pandas as pd
from stable_baselines3 import DDPG
from hr_sim_env import hr_sim_env
import matplotlib.pyplot as plt
import numpy as np

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




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


true_hr = results_df['true_hr'].values
predicted_hr = results_df['predicted_hr'].values

mae = mean_absolute_error(true_hr, predicted_hr)
mse = mean_squared_error(true_hr, predicted_hr)
rmse = np.sqrt(mse)
r2 = r2_score(true_hr, predicted_hr)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")