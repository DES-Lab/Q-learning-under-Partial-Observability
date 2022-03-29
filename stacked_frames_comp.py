import sys

import gym
import numpy as np
from gym.spaces import Box
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines import DQN, A2C, ACKTR

from utils import add_statistics_to_file
from world_repository import get_world

learning_alg_name = {DQN: 'DQN', A2C: 'A2C', ACKTR: 'ACKTR'}

# surprise tensorflow future warnings
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

python_version = sys.version_info
if python_version.major != 3 and python_version.minor != 6:
    print('Ensure that you are using Python 3.6 and libraries version defined in "Comparison" section of readme.')
    assert False


class StackedPoge(gym.Env):
    def __init__(self, poge_env, stacked_frames_num=5):
        self.poge_env = poge_env
        self.frame_size = stacked_frames_num
        self.action_space = self.poge_env.action_space

        num_abstract_tiles = self.poge_env.observation_space.n
        self.empty_obs = num_abstract_tiles
        self.observation_space = Box(low=0, high=num_abstract_tiles, shape=(1, self.frame_size))

        self.observation_frame = [self.empty_obs for i in range(self.frame_size)]

    def step(self, action):
        out, rew, done, info = self.poge_env.step(action)
        self._update_output(out)
        return self.to_np_array(), rew, done, info

    def reset(self):
        self.observation_frame = [self.empty_obs for i in range(self.frame_size)]
        out = self.poge_env.reset()
        self._update_output(out)
        return self.to_np_array()

    def _update_output(self, new_out):
        self.observation_frame.pop(0)
        self.observation_frame.append(new_out)

    def to_np_array(self):
        arr = np.array([self.observation_frame])
        return arr

    def render(self, mode='human'):
        pass


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, env, check_freq: int = 1000, early_stopping_percentage=0.98, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.check_freq = check_freq
        self.data = []
        self.verbose = True if verbose == 1 else 0
        self.early_stopping_percentage = early_stopping_percentage

    def _on_step(self):
        if self.env.poge_env.training_episode % self.check_freq == 0:
            if self.verbose:
                print(f'Training Episode: {self.env.poge_env.training_episode}')

            data = evaluate_dqn(self.model, self.env, num_episodes=100, verbose=self.verbose)
            # self.env.poge_env.training_episode -= 100  # subtract eval episodes
            self.data.append(data)

            goal_reached = data[0]
            if goal_reached / 100 >= self.early_stopping_percentage:
                if self.verbose:
                    print(f'Early stopping triggered. Goal reached in {goal_reached}% of test episodes.')
                return False


def evaluate_dqn(model, env, num_episodes=100, verbose=True):
    steps = 0
    cumulative_reward = 0
    goal_reached = 0
    for i in range(num_episodes):
        _states = None
        obs = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            steps += 1
            obs, rewards, done, info = env.step(action)
            if rewards == env.poge_env.goal_reward:
                goal_reached += 1

            cumulative_reward += rewards

    avg_rew = round(cumulative_reward / num_episodes, 2)
    avg_step = round(steps / num_episodes, 2)
    if verbose:
        print(f'Tested on {num_episodes} episodes:')
        print(f'Goal reached    : {goal_reached}')
        print(f'Avg. reward     : {avg_rew}')
        print(f'Avg. step count : {avg_step}')
    return goal_reached, avg_rew, avg_step


def stacked_experiment(experiment_name, poge_env: StackedPoge, learning_alg, training_steps, interval_size=1000,
                       num_frames=5, early_stopping_acc=1.01, verbose=False):
    assert learning_alg in {DQN, A2C, ACKTR}

    poge_env.training_episode = 0

    env = StackedPoge(poge_env, num_frames)

    statistic_collector = TrainingMonitorCallback(env, check_freq=interval_size,
                                                  early_stopping_percentage=early_stopping_acc,
                                                  verbose=verbose)

    model = learning_alg('MlpPolicy', env, verbose=False)
    print(
        f'Training started for {experiment_name}. Algorithm: {learning_alg_name[learning_alg]}, Num. Steps: {training_steps}')
    model.learn(total_timesteps=training_steps, callback=statistic_collector)
    print(f'Training finished.')

    exp_setup_str = f'{learning_alg_name[learning_alg]},Training Steps:{training_steps}, NumFrames:{num_frames}'
    statistics = statistic_collector.data
    statistics.insert(0, exp_setup_str)

    add_statistics_to_file(experiment_name, statistics, interval_size, subfolder='stacked')

    return evaluate_dqn(model, env)


if __name__ == '__main__':
    exp = 'mid_line'
    num_training_episodes = 12000
    frame_size = 5
    poge = get_world(exp)

    # https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

    stacked_experiment(exp, poge, ACKTR, training_steps=num_training_episodes * poge.max_ep_len, num_frames=frame_size,
                       verbose=True, early_stopping_acc=1)
