import gym

import numpy as np
from stable_baselines import A2C, ACER, PPO2, ACKTR
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import DummyVecEnv

import tensorflow as tf

from utils import add_statistics_to_file
from world_repository import get_world

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.seterr(all="ignore")

learning_alg_name = {A2C: 'A2C', ACER: 'ACER', PPO2: 'PPO2', ACKTR: 'ACTOR'}


def evaluate_ltsm(model, env, num_episodes=100, verbose=True):
    """
    For DummyVecEnv, LSTM only.
    """
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

    avg_rew = round(cumulative_reward / num_episodes, 2)
    avg_step = round(steps / num_episodes, 2)
    if verbose:
        print(f'Tested on {num_episodes} episodes:')
        print(f'Goal reached    : {goal_reached}')
        print(f'Avg. reward     : {avg_rew}')
        print(f'Avg. step count : {avg_step}')
    return goal_reached, avg_rew, avg_step


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, env, check_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.check_freq = check_freq
        self.data = []
        self.verbose = True if verbose == 1 else 0

    def _on_step(self):
        if self.env.envs[0].training_episode % self.check_freq == 0:
            if self.verbose:
                print(f'Training Episode: {self.env.poge_env.training_episode}')
            data = evaluate_ltsm(self.model, self.env, num_episodes=100, verbose=self.verbose)
            # self.env.envs[0].training_episode -= check_freq # subtract eval episodes
            self.data.append(data)


def lstm_experiment(poge, learning_alg, training_steps, interval_size=1000, verbose=False):
    assert learning_alg in {A2C, ACER, PPO2, ACKTR}

    poge.training_episode = 0

    env = DummyVecEnv([lambda: poge])

    statistic_collector = TrainingMonitorCallback(env, check_freq=interval_size, verbose=verbose)

    model = ACER('MlpLstmPolicy', env, n_cpu_tf_sess=None)
    model.learn(total_timesteps=training_steps, callback=statistic_collector)

    exp_setup_str = f'{learning_alg_name[learning_alg]},{training_steps}'
    statistics = statistic_collector.data
    statistics.insert(0, exp_setup_str)

    add_statistics_to_file(env.envs[0].world_file_path, statistics, interval_size)

    return evaluate_ltsm(model, env)


env = get_world('gravity')

lstm_experiment(env, ACER, 10000)
