import gym
import numpy as np
from gym.spaces import Box
from stable_baselines.deepq import MlpPolicy

import gym_partially_observable_grid

from stable_baselines import DQN


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

poge = gym.make(id='poge-v1',
                world_file_path='worlds/world1.txt',
                is_partially_obs=True,
                force_determinism=False,
                indicate_slip=False,
                indicate_wall=False,
                one_time_rewards=True,
                step_penalty=0.1)

num_frames = 10
training_steps = 1000000
env = StackedPoge(poge, num_frames)

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=training_steps)

evaluate_dqn(model, env)
