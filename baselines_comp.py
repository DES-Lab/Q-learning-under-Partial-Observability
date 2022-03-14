import os

import gym
import numpy as np
from stable_baselines import A2C, ACER, PPO2, ACKTR
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv

import gym_partially_observable_grid

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def evaluate_ltsm(model, env, num_episodes=100):
    """
    For DummyVecEnv, LSTM only.
    """
    all_rewards = []
    steps = 0
    cumulative_reward = 0
    goal_reached = 0
    for i in range(num_episodes):
        _states = None
        obs = env.reset()
        done = [False]

        while not done[0]:
            action, _states = model.predict(obs,
                                            state=_states,
                                            mask=done)
            steps += 1
            obs, rewards, done, info = env.step(action)

            if rewards[0] == env.envs[0].goal_reward:
                goal_reached += 1

            cumulative_reward += rewards[0]

    print(f'Tested on {num_episodes} episodes:')
    print(f'Goal reached    : {goal_reached}')
    print(f'Avg. reward     : {round(cumulative_reward / num_episodes, 2)}')
    print(f'Avg. step count : {round(steps / num_episodes, 2)}')


poge = gym.make(id='poge-v1',
                world_file_path='worlds/world1.txt',
                is_partially_obs=True,
                force_determinism=False,
                indicate_slip=False,
                indicate_wall=False,
                one_time_rewards=True,
                step_penalty=0.1, )

poge = Monitor(poge, 'test')
env = DummyVecEnv([lambda: poge])
training_time_stpes = 1000000

model = ACER('MlpLstmPolicy', env).learn(total_timesteps=training_time_stpes)

evaluate_ltsm(model, env, 100)
