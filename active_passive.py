import random
from statistics import mean

import aalpy.paths
import gym
import gym_partially_observable_grid

from aalpy.base import SUL
from aalpy.learning_algs import run_active_Alergia
from aalpy.learning_algs.stochastic_passive.ActiveAleriga import Sampler
from aalpy.utils import visualize_automaton

from prism_schedulers import PrismInterface

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"


class StochasticWorldSUL(SUL):
    def __init__(self, stochastic_world):
        super().__init__()
        self.world = stochastic_world
        self.goal_reached = False

    def pre(self):
        self.goal_reached = False
        self.world.reset()

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            return world.get_abstraction()
        output, reward, done, info = self.world.step(world.actions_dict[letter])
        if reward == self.world.goal_reward or self.goal_reached:
            self.goal_reached = True
            return "GOAL"
        if done:
            return "EP_END"
        output = self.world.decode(output)
        if str(output).isdigit():
            output = f's{output}'
        return output


class EpsGreedySampler(Sampler):
    def __init__(self, input_al, eps=0.9, num_new_samples=2000, min_seq_len=10, max_seq_len=50):
        self.eps = eps
        self.new_samples = num_new_samples
        self.input_al = input_al
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

    def sample(self, sul, model):
        new_data = []

        prism_interface = PrismInterface("GOAL", model)
        completely_random = True if prism_interface.property_val < 0.5 else False

        for _ in range(self.new_samples):
            sample = ['Init']
            sul.pre()
            prism_interface.reset()
            continue_random = completely_random

            for _ in range(random.randint(self.min_seq_len, self.max_seq_len)):
                if not continue_random and random.random() < self.eps:
                    i = prism_interface.get_input()
                    if i is None:
                        i = random.choice(self.input_al)
                else:
                    i = random.choice(self.input_al)

                o = sul.step(i)
                sample.append((i, o))

                continue_random = not prism_interface.step_to(i, o)

            sul.post()
            new_data.append(sample)

        return new_data


# Make environment deterministic even if it is stochastic
force_determinism = False
# Add slip to the observation set (action failed)
indicate_slip = False
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = True

world = gym.make('poge-v1', world_file_path='worlds/world1.txt',
                 force_determinism=force_determinism,
                 indicate_slip=indicate_slip,
                 is_partially_obs=is_partially_obs)

input_al = list(world.actions_dict.keys())

sul = StochasticWorldSUL(world)


def get_initial_data(sul, input_al, initial_sample_num=5000, min_seq_len=10, max_seq_len=50):
    random_samples = []
    for _ in range(initial_sample_num):
        sample = ['Init']
        sul.pre()
        for _ in range(random.randint(min_seq_len, max_seq_len)):
            i = random.choice(input_al)
            o = sul.step(i)
            sample.append((i, o))
        sul.post()
        random_samples.append(sample)
    return random_samples


def test_model(model, sul, input_al, num_episodes, max_ep_len=100):
    num_steps_per_ep = []
    goal_reached = 0

    prism_interface = PrismInterface("GOAL", model)

    for _ in range(num_episodes):
        step_counter = 0
        scheduler_step_valid = True
        sul.pre()
        prism_interface.reset()
        while True:
            if step_counter == max_ep_len:
                break
            i = prism_interface.get_input()
            if not scheduler_step_valid or i is None:
                i = random.choice(input_al)
            o = sul.step(i)
            step_counter += 1
            scheduler_step_valid = prism_interface.step_to(i, o)
            if o == 'GOAL':
                goal_reached += 1
                break

        num_steps_per_ep.append(step_counter)
        sul.post()

    print(f'Tested on {num_episodes} episodes:')
    print(f'Goal reached  : {goal_reached}')
    print(f'Avg. step count : {mean(num_steps_per_ep)}')


data = get_initial_data(sul, input_al)

sampler = EpsGreedySampler(input_al, num_new_samples=2000)

final_model = run_active_Alergia(data=data, sul=sul, sampler=sampler, n_iter=5)

prism_interface = PrismInterface("GOAL", final_model)
goal_reached = prism_interface.property_val

print(f'Goal Reachable in Final Model: {goal_reached}')

test_model(final_model, sul, input_al, num_episodes=100)