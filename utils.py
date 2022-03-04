import random
from statistics import mean
import gym
import gym_partially_observable_grid

from aalpy.base import SUL

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

        self.button_location = None
        for y, line in enumerate(self.env.abstract_world):
            for x, tile in enumerate(line):
                if tile == '@':
                    self.button_location = y, x

        assert self.button_location

    def reset(self):
        return self.env.reset()

    def step(self, action):
        abstract_obs, rewards, done, _ = self.env.step(action)
        player_x, player_y = self.env.player_location

        abstract_obs = self.env.decode(abstract_obs)
        if abstract_obs == 'button':
            self.env.env.goal_locations = {random.choice(self.possible_cookies_locations)}

        return (player_x, player_y, abstract_obs), rewards, done, _

    def play(self):
        def render():
            from copy import deepcopy
            world_copy = deepcopy(self.env.world)
            for x,y in self.possible_cookies_locations:
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
            print(f'Output: {output, reward, done, info}')

if __name__ == '__main__':
    cd = CookieDomain()
    cd.play()

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
        output = f'{output[0]}_{output[1]}'

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
        world = deepcopy(env.world)
        world[xy[0]][xy[1]] = 'E'
        for line in world:
            print(f'{"".join(line)}')
        sleep(step_time)


def get_initial_data(env, input_al, initial_sample_num=5000, min_seq_len=10, max_seq_len=50):
    # Generate random initial samples
    random_samples = []
    for _ in range(initial_sample_num):
        sample = ['Init']
        env.reset()
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            i = random.choice(input_al)
            encoded_i = env.actions_dict[i]
            o, r, _, _ = env.step(encoded_i)
            o = process_output(env, o, r)
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
