from statistics import mean

import gym
import numpy as np
from stable_baselines import A2C
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecEnv, VecEnvWrapper, DummyVecEnv, SubprocVecEnv

import gym_partially_observable_grid

# install with pip install stable-baselines3[extra]
from baselines.common import set_global_seeds
from utils import process_output

env = gym.make(id='poge-v1',
               world_file_path='worlds/world1.txt',
               is_partially_obs=False,
               force_determinism=False,
               indicate_slip=False,
               indicate_wall=False,
               one_time_rewards=True)


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env.seed(seed + rank)
        return env_id

    set_global_seeds(seed)
    return _init


env = SubprocVecEnv([make_env(env, i) for i in range(4)])

# MlpPolicy or MultiInputPolicy
model = A2C('MlpLstmPolicy', env).learn(total_timesteps=10000)

num_steps_per_ep = []
goal_reached = 0
max_ep_len = 100

for _ in range(100):
    step_counter = 0
    scheduler_step_valid = True
    obs = env.reset()

    zero_completed_obs = np.zeros((model.n_envs,) + model.observation_space.shape)
    zero_completed_obs = obs
    obs = zero_completed_obs
    state = None

    while True:
        if step_counter == max_ep_len:
            break

        action, state = model.predict(obs, state=state, deterministic=False)

        encoded_i = env.actions_dict[action]
        o, r, _, _ = env.step(encoded_i)
        o = process_output(env, o, r)
        step_counter += 1

        if r == env.goal_reward:
            goal_reached += 1
            break

    num_steps_per_ep.append(step_counter)

print(f'Tested on {100} episodes:')
print(f'Goal reached  : {goal_reached}')
print(f'Avg. step count : {mean(num_steps_per_ep)}')
