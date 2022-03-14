from statistics import mean

import gym
import numpy as np
from stable_baselines import A2C, ACER
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecEnv, VecEnvWrapper, DummyVecEnv, SubprocVecEnv

import gym_partially_observable_grid

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from utils import process_output

env = gym.make(id='poge-v1',
               world_file_path='worlds/world1.txt',
               is_partially_obs=True,
               force_determinism=False,
               indicate_slip=False,
               indicate_wall=False,
               one_time_rewards=True,
               step_penalty=0.1, )

env = DummyVecEnv([lambda: env])

# MlpPolicy or MultiInputPolicy
model = ACER('MlpLstmPolicy', env).learn(total_timesteps=10000)


def evaluate_ltsm(model, env, num_episodes=100):
    """
    For DummyVecEnv, LSTM only.
    """
    all_rewards = []
    steps = 0
    goal_reached = 0
    for i in range(num_episodes):
        _states = None
        obs = env.reset()
        temp_reward = 0
        done = [False]

        while not done[0]:
            action, _states = model.predict(obs,
                                            state=_states,
                                            mask=done)
            steps += 1
            obs, rewards, done, info = env.step(action)
            if rewards[0]:
                goal_reached += 1

            temp_reward += rewards[0]
        all_rewards.append(temp_reward)

    print(f'Tested on {num_episodes} episodes:')
    print(f'Goal reached    : {goal_reached}')
    print(f'Avg. reward     : {round(mean(all_rewards), 2)}')
    print(f'Avg. step count : {round(steps / 100, 2)}')


evaluate_ltsm(model, env, 100)
