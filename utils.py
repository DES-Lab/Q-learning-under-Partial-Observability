import random
from statistics import mean
import gym

from aalpy.base import SUL
from gym.spaces import Discrete


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
        self.env.goal_locations = set()

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
        self.env.goal_locations = set()
        return self.env.encode((x, y, self.env.get_observation(), self.is_cookie_in_the_room()))

    def step(self, action):
        abstract_obs, rewards, done, _ = self.env.step(action)
        player_x, player_y = self.env.player_location

        abstract_obs = self.env.decode(abstract_obs)

        env_state = self.env.encode((player_x, player_y, abstract_obs, self.is_cookie_in_the_room()))

        if abstract_obs == 'button':
            self.env.goal_locations = {random.choice(self.possible_cookies_locations)}

        # enforce correct behaviour
        if rewards == self.env.goal_reward and self.env.player_location not in self.env.goal_locations:
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
        self.env.state_2_one_hot_map = {}
        # hack to make reset work and abstract steps work
        x, y = self.env.player_location
        reset_tile = self.env.abstract_symbol_name_map[self.env.abstract_world[x][y]]
        self.env.state_2_one_hot_map[reset_tile] = 0
        counter += 1
        for ab_tile in list(set(self.env.abstract_symbol_name_map.values())):
            self.env.state_2_one_hot_map[ab_tile] = counter
            counter += 1

        world_to_process = self.env.world
        for x, row in enumerate(world_to_process):
            for y, tile in enumerate(row):
                if tile not in {'#', 'D', 'E'}:
                    if tile == ' ' or tile == 'G':
                        abstract_tile = self.env.abstract_symbol_name_map[self.env.abstract_world[x][y]]
                        # x, y, abstract output, is_cookie_in_the_room
                        self.env.state_2_one_hot_map[(x, y, abstract_tile, False)] = counter
                        self.env.state_2_one_hot_map[(x, y, abstract_tile, True)] = counter + 1
                        counter += 2

        self.env.one_hot_2_state_map = {v: k for k, v in self.env.state_2_one_hot_map.items()}

        return counter

    def is_cookie_in_the_room(self):
        if not self.env.goal_locations:
            return False
        x, y = self.env.player_location
        curr_room = self.env.abstract_symbol_name_map[self.env.abstract_world[x][y]]
        cookie_x, cookie_y = list(self.env.goal_locations)[0]
        cookie_room = self.env.abstract_symbol_name_map[self.env.abstract_world[cookie_x][cookie_y]]
        return curr_room == cookie_room


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


def get_initial_data(env, input_al, initial_sample_num=5000, min_seq_len=10, max_seq_len=50, incl_rewards=False,
                     is_smm=False):
    # Generate random initial samples
    random_samples = []
    for _ in range(initial_sample_num):
        sample = [] if is_smm else ['Init']
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


def add_statistics_to_file(experiment_name, statistics, statistic_interval_size, subfolder=''):
    import os
    import csv

    subfolder = subfolder + '/' if subfolder[-1] != '/' else subfolder

    if not os.path.exists(f'statistics/{subfolder}'):
        os.makedirs(f'statistics/{subfolder}')

    experiment_setup = statistics.pop(0)

    intervals, goal_reached, avg_reward, avg_step = [], [], [], []

    current_interval = statistic_interval_size
    for s in statistics:
        intervals.append(current_interval)
        goal_reached.append(s[0])
        avg_reward.append(s[1])
        avg_step.append(s[2])
        current_interval += statistic_interval_size

    with open(f'statistics/{subfolder}{experiment_name}.csv', 'a', newline='') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow([experiment_setup])
        writer.writerow(intervals)
        writer.writerow(goal_reached)
        writer.writerow(avg_reward)
        writer.writerow(avg_step)


def writeSamplesToFile(samples, path="alergiaSamples.txt"):
    isSMM = False
    if isinstance(samples[0][0], tuple):
        isSMM = True
    with open(path, 'a') as f:
        for sample in samples:
            s = "" if isSMM else f'{str(sample.pop(0))},'
            for i, o in sample:
                s += f'{i},{o},'
            s = s[:-1]
            f.write(s + '\n')

    f.close()
    samples.clear()


def deleteSampleFile(path="alergiaSamples.txt"):
    import os
    if os.path.exists(path):
        os.remove(path)
