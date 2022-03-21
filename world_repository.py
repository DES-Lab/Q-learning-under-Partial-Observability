import gym
import gym_partially_observable_grid


def get_all_world_ids():
    return ['gravity', 'gravity2', 'corridor_permanent_rew', 'corridor_one_time_rew', 'corridor-rew',
            'big_office_one_time_rew', 'big_office_permanent_rew', 'misleading_office_one_time',
            'world1', 'world1+rew', 'world2', 'world2+rew', 'corner', 'thin_maze', 'world1_confusing',
            'simple_showcase', 'simple_showcase2']


def get_world(world_id):
    env = None
    if world_id == 'gravity':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/confusing_big_gravity.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.1)
    if world_id == 'gravity2':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/big_gravity_2.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.2)
    if world_id == 'corridor_permanent_rew':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/corridor.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=100,
                       step_penalty=0.2)
    if world_id == 'corridor_one_time_rew':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/corridor.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=100,
                       step_penalty=0.2)
    if world_id == 'corridor-rew':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/corridor-rew.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=200,
                       goal_reward=100,
                       step_penalty=0.2)
    if world_id == 'big_office_one_time_rew':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/office_world1.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=100,
                       step_penalty=0.2)
    if world_id == 'big_office_permanent_rew':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/office_world1.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=150,
                       goal_reward=100,
                       step_penalty=0.2)
    if world_id == 'misleading_office_one_time':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/misleading_office_world.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=True,
                       max_ep_len=150,
                       goal_reward=100,
                       step_penalty=0.2)
    if world_id == 'world1':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world1.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.1)
    if world_id == 'world1_confusing':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world1_confusing.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.1)
    if world_id == 'world1+rew':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world1.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=10,
                       step_penalty=0.1)
    if world_id == 'world2':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world2-reward.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=400,
                       goal_reward=100,
                       step_penalty=2)
    if world_id == 'world2+rew':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world2-reward.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=200,
                       goal_reward=100,
                       step_penalty=2)
    if world_id == 'corner':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/corner.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=100,
                       step_penalty=0.1)
    if world_id == 'thin_maze':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/thin_maze.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=True,
                       max_ep_len=200,
                       goal_reward=100,
                       step_penalty=0.1)
    if world_id == 'maze':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/maze.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=True,
                       max_ep_len=150,
                       goal_reward=100,
                       step_penalty=0.1)
    if world_id == 'simple_showcase':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/simple_showcase.txt',
                       is_partially_obs=True,
                       one_time_rewards=True,
                       indicate_wall=False,
                       max_ep_len=150,
                       goal_reward=100,
                       step_penalty=0.1)
    if world_id == 'simple_showcase2':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/simple_showcase2.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=200,
                       step_penalty=0.1)

    assert env is not None
    return env
