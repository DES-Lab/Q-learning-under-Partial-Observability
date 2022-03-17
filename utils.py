import random
from statistics import mean
import gym
import gym_partially_observable_grid

from aalpy.base import SUL
from gym.spaces import Discrete

from prism_schedulers import PrismInterface


class CookieDomain:
    def __init__(self):
        self.env = gym.make(id='poge-v1',
                            world_file_path='worlds/cookie_domain.txt',
                            force_determinism=False,
                            indicate_slip=False,
                            is_partially_obs=True,
                            indicate_wall=False,
                            one_time_rewards=False)

        self.possible_cookies_locations = list(self.env.goal_locations.copy())
        self.env.env.goal_locations = set()

        self.goal_reward = self.env.goal_reward

        self.button_location = None
        for y, line in enumerate(self.env.abstract_world):
            for x, tile in enumerate(line):
                if tile == '@':
                    self.button_location = y, x
        assert self.button_location

        env_size = self._get_obs_space()

        self.action_space = Discrete(4)
        self.observation_space = Discrete(env_size)

        self.actions_dict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.action_space_to_act_map = {i: k for k, i in self.actions_dict.items()}
        self.actions = [0, 1, 2, 3]

    def reset(self):
        self.env.reset()
        x, y = self.env.player_location
        self.env.env.goal_locations = set()
        return self.env.encode((x, y, self.env.get_observation(), self.is_cookie_in_the_room()))

    def step(self, action):
        abstract_obs, rewards, done, _ = self.env.step(action)
        player_x, player_y = self.env.player_location

        abstract_obs = self.env.decode(abstract_obs)

        env_state = self.env.encode((player_x, player_y, abstract_obs, self.is_cookie_in_the_room()))

        if abstract_obs == 'button':
            self.env.env.goal_locations = {random.choice(self.possible_cookies_locations)}

        # enforce correct behaviour
        if rewards == self.env.goal_reward and self.env.player_location not in self.env.env.goal_locations:
            rewards = 0

        return env_state, rewards, done, _

    def encode(self, o):
        return self.env.encode(o)

    def decode(self, o):
        return self.env.decode(o)

    def play(self):
        def render():
            from copy import deepcopy
            world_copy = deepcopy(self.env.world)
            for x, y in self.possible_cookies_locations:
                world_copy[x][y] = ' '
            for x, y in self.env.goal_locations:
                world_copy[x][y] = 'G'
            world_copy[self.button_location[0]][self.button_location[1]] = '@'
            world_copy[self.env.player_location[0]][self.env.player_location[1]] = 'E'
            for l in world_copy:
                print("".join(l))

        self.reset()
        user_input_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}
        print('Agent is controlled with w,a,s,d; for up,left,down,right actions.')
        while True:
            render()
            action = input('Action: ', )
            output, reward, done, info = self.step(user_input_map[action])
            print(f'Output: {self.env.decode(output), reward, done, info}')

    def _get_obs_space(self):
        counter = 0
        self.env.env.state_2_one_hot_map = {}
        # hack to make reset work and abstract steps work
        x, y = self.env.player_location
        reset_tile = self.env.abstract_symbol_name_map[self.env.abstract_world[x][y]]
        self.env.env.state_2_one_hot_map[reset_tile] = 0
        counter += 1
        for ab_tile in list(set(self.env.abstract_symbol_name_map.values())):
            self.env.env.state_2_one_hot_map[ab_tile] = counter
            counter += 1

        world_to_process = self.env.world
        for x, row in enumerate(world_to_process):
            for y, tile in enumerate(row):
                if tile not in {'#', 'D', 'E'}:
                    if tile == ' ' or tile == 'G':
                        abstract_tile = self.env.abstract_symbol_name_map[self.env.abstract_world[x][y]]
                        # x, y, abstract output, is_cookie_in_the_room
                        self.env.env.state_2_one_hot_map[(x, y, abstract_tile, False)] = counter
                        self.env.env.state_2_one_hot_map[(x, y, abstract_tile, True)] = counter + 1
                        counter += 2

        self.env.env.one_hot_2_state_map = {v: k for k, v in self.env.env.state_2_one_hot_map.items()}

        return counter

    def is_cookie_in_the_room(self):
        if not self.env.goal_locations:
            return False
        x, y = self.env.player_location
        curr_room = self.env.abstract_symbol_name_map[self.env.abstract_world[x][y]]
        cookie_x, cookie_y = list(self.env.goal_locations)[0]
        cookie_room = self.env.abstract_symbol_name_map[self.env.abstract_world[cookie_x][cookie_y]]
        return curr_room == cookie_room


if __name__ == '__main__':
    ck = CookieDomain()
    ck.env.env.step_penalty = -1
    ck.play()


class StochasticWorldSUL(SUL):
    def __init__(self, stochastic_world):
        super().__init__()
        self.world = stochastic_world
        self.goal_reached = False
        self.is_done = False

    def pre(self):
        self.goal_reached = False
        self.is_done = False
        self.world.reset()

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            output = self.world.get_abstraction()
            if output[0].isdigit().isdigit():
                output = f'state_{output}'
            return output

        output, reward, done, info = self.world.step(self.world.actions_dict[letter])

        if reward == self.world.goal_reward or self.goal_reached:
            self.goal_reached = True
            return "GOAL"

        if done or self.is_done:
            self.is_done = True
            return "MAX_STEPS_REACHED"

        output = self.world.decode(output)
        if isinstance(output, tuple):
            output = f'{output[0]}_{output[1]}'
        if reward != 0 and reward != self.world.step_penalty:
            reward = reward if reward > 0 else f'neg_{reward * -1}'

        if output[0].isdigit():
            output = f'state_{output}'
        if reward != 0:
            output = f'{output}_r_{reward}'

        return output


def process_output(env, output, reward=0):
    if reward == env.goal_reward:
        return "GOAL"

    output = env.decode(output)
    if isinstance(output, tuple):
        output = 's_'.join([str(s) for s in output])

    if reward != 0 and reward != env.step_penalty:
        reward = reward if reward > 0 else f'neg_{reward * -1}'

    if output[0].isdigit():
        output = f'state_{output}'
    if reward != 0 and reward != env.step_penalty:
        output = f'{output}_r_{reward}'

    return output


def test_model(model, env, input_al, num_episodes, max_ep_len=100):
    num_steps_per_ep = []
    goal_reached = 0

    prism_interface = PrismInterface("GOAL", model)
    print(f'Goal Reachability: {prism_interface.property_val}')

    for _ in range(num_episodes):
        step_counter = 0
        scheduler_step_valid = True
        env.reset()
        prism_interface.reset()
        while True:
            if step_counter == max_ep_len:
                break
            i = prism_interface.get_input()
            if not scheduler_step_valid or i is None:
                i = random.choice(input_al)
            encoded_i = env.actions_dict[i]
            o, r, _, _ = env.step(encoded_i)
            o = process_output(env, o, r)
            step_counter += 1
            scheduler_step_valid = prism_interface.step_to(i, o)
            if r == env.goal_reward:
                goal_reached += 1
                break

        num_steps_per_ep.append(step_counter)

    print(f'Tested on {num_episodes} episodes:')
    print(f'Goal reached  : {goal_reached}')
    print(f'Avg. step count : {mean(num_steps_per_ep)}')


def visualize_episode(env, coordinate_list, step_time=0.7):
    from time import sleep
    from copy import deepcopy
    env.reset()

    for xy in coordinate_list:
        env.player_location = xy
        world = deepcopy(env.env.world)
        world[xy[0]][xy[1]] = 'E'
        for line in world:
            print(f'{"".join(line)}')
        sleep(step_time)


def get_initial_data(env, input_al, initial_sample_num=5000, min_seq_len=10, max_seq_len=50, incl_rewards=False):
    # Generate random initial samples
    random_samples = []
    for _ in range(initial_sample_num):
        sample = ['Init']
        env.reset()
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            i = random.choice(input_al)
            encoded_i = env.actions_dict[i]
            o, r, _, _ = env.step(encoded_i)
            if incl_rewards:
                o = process_output(env, o, r)
            else:
                o = env.decode(o)
            sample.append((i, o))
        random_samples.append(sample)
    return random_samples


def get_samples_reaching_goal(env, num_samples=10):
    explored = set()
    actions = list(env.actions_dict.values())
    queue = [[env.player_location]]
    path_locations = []

    while queue:
        path = queue.pop(0)
        location = path[-1]
        if location not in explored:
            for a in actions:

                # reset the env
                env.reset()
                env.use_stochastic_tiles = False

                env.env.player_location = location
                _, r, _, _ = env.step(a)

                new_path = list(path)
                new_path.append(env.env.player_location)
                queue.append(new_path)

                if r == env.goal_reward:
                    path_locations.append(new_path)

            # mark node as explored
            explored.add(location)

    env.use_stochastic_tiles = True

    return path_locations


def add_statistics_to_file(path_to_world, statistics, statistic_interval_size):
    import csv

    world_name = path_to_world.split('/')[-1].split('.')[0] + '.csv'

    experiment_setup = statistics.pop(0)

    intervals, goal_reached, avg_reward, avg_step = [], [], [], []

    current_interval = statistic_interval_size
    for s in statistics:
        intervals.append(current_interval)
        goal_reached.append(s[0])
        avg_reward.append(s[1])
        avg_step.append(s[2])
        current_interval += statistic_interval_size

    with open(f'statistics/{world_name}', 'a', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow([experiment_setup])
        writer.writerow(intervals)
        writer.writerow(goal_reached)
        writer.writerow(avg_reward)
        writer.writerow(avg_step)
