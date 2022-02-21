import random
from collections import Counter

import aalpy.paths
import gym
import gym_partially_observable_grid

from aalpy.learning_algs import run_active_Alergia
from aalpy.learning_algs.stochastic_passive.ActiveAleriga import Sampler
from aalpy.utils import visualize_automaton, save_automaton_to_file, load_automaton_from_file

from prism_schedulers import PrismInterface
from utils import StochasticWorldSUL, test_model, get_initial_data

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"


class EpsGreedySampler(Sampler):
    def __init__(self, input_al, eps=0.9, num_new_samples=2000, min_seq_len=10, max_seq_len=50):
        self.eps = eps
        self.new_samples = num_new_samples
        self.input_al = input_al
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.scheduler_freq_counter = Counter()
        self.sampling_round = 0

    def sample(self, sul, model):
        self.sampling_round += 1
        new_data = []

        reward_states = set()
        for s in model.states:
            if '_r_' in s.output or 'GOAL' in s.output:
                reward_states.add(s.output)
                if s.output not in self.scheduler_freq_counter.keys():
                    self.scheduler_freq_counter[s.output] += 1

        reward_states = list(reward_states)

        schedulers = {rs: PrismInterface(rs, model) for rs in reward_states}

        # print(reward_states)
        # print([s for s in schedulers.keys()])
        # print(self.scheduler_freq_counter)

        for _ in range(self.new_samples):
            # select a scheduler according to the inverse frequency distribution -> less used schedulers will be
            # sampled more

            ignore_scheduler = True if len(reward_states) == 0 else False

            scheduler = None
            if not ignore_scheduler:
                reward_state = random.choices(list(self.scheduler_freq_counter.keys()),
                                              [1 / v for v in self.scheduler_freq_counter.values()], k=1)[0]
                scheduler = schedulers[reward_state]
                self.scheduler_freq_counter[reward_state] += 1
                ignore_scheduler = True if scheduler.property_val < 0.5 else False

            sample = ['Init']
            sul.pre()
            if not ignore_scheduler:
                scheduler.reset()
            continue_random = ignore_scheduler
            for _ in range(random.randint(self.min_seq_len, self.max_seq_len)):
                if not continue_random and random.random() < self.eps:
                    i = scheduler.get_input()
                    if i is None:
                        i = random.choice(self.input_al)
                        continue_random = True
                else:
                    i = random.choice(self.input_al)

                o = sul.step(i)
                sample.append((i, o))

                if o == 'GOAL':
                    break

                # once reward state is reached, continue doing completely random sampling
                if not continue_random and not ignore_scheduler:
                    continue_random = True if o == scheduler.dest else not scheduler.step_to(i, o)

            sul.post()
            new_data.append(sample)

        return new_data


# Make environment deterministic even if it is stochastic
force_determinism = False
# Add slip to the observation set (action failed)
indicate_slip = False
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = True

min_seq_len, max_seq_len = 10, 50

world = gym.make(id='poge-v1',
                 world_file_path='worlds/world1.txt',
                 force_determinism=force_determinism,
                 indicate_slip=indicate_slip,
                 is_partially_obs=is_partially_obs,
                 one_time_rewards=True)

input_al = list(world.actions_dict.keys())

sul = StochasticWorldSUL(world)

data = get_initial_data(sul, input_al, initial_sample_num=10000, min_seq_len=min_seq_len, max_seq_len=max_seq_len)

sampler = EpsGreedySampler(input_al, eps=0.9, num_new_samples=2000, min_seq_len=min_seq_len, max_seq_len=max_seq_len)

final_model = run_active_Alergia(data=data, sul=sul, sampler=sampler, n_iter=5)
# final_model = load_automaton_from_file('passive_active.dot', automaton_type='mdp')
print(f'Final model size: {final_model.size}')
# save_automaton_to_file(final_model, 'passive_active')

test_model(final_model, sul, input_al, num_episodes=100)
