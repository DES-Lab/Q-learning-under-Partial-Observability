import gym
import gym_partially_observable_grid
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# install with pip install stable-baselines3[extra]

env = gym.make(id='poge-v1',
               world_file_path='worlds/world1.txt',
               is_partially_obs=False,
               force_determinism=False,
               indicate_slip=False,
               indicate_wall=False,
               one_time_rewards=True)

# MlpPolicy or MultiInputPolicy
model = A2C('MlpPolicy', env).learn(total_timesteps=10000)

avg_rew, _ = evaluate_policy(model, env, 100)
print(avg_rew)