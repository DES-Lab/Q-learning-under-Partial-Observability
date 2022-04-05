import gym
import gym_partially_observable_grid


def get_all_world_ids():
    return ['gravity', 'officeWorld', 'confusingOfficeWorld', 'thinMaze']


def get_world(world_id):
    env = None
    if world_id == 'gravity':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/big_gravity_2.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.2)
    if world_id == 'officeWorld':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world1.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.1)
    if world_id == 'confusingOfficeWorld':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world1_confusing.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.1)
    if world_id == 'thinMaze':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/simple_showcase2.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=200,
                       goal_reward=200,
                       step_penalty=0.1)
    assert env is not None
    return env
