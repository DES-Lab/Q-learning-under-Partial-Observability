import gym
import gym_partially_observable_grid


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
    if world_id == 'corridor':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/corridor.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=100,
                       goal_reward=100,
                       step_penalty=0.2)
    if world_id == 'big_office':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/office_world1.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=200,
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
    if world_id == 'world2':
        env = gym.make(id='poge-v1',
                       world_file_path='worlds/world2-reward.txt',
                       is_partially_obs=True,
                       one_time_rewards=False,
                       indicate_wall=True,
                       max_ep_len=250,
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

    assert env is not None
    return env
