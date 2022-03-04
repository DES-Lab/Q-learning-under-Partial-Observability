import gym
import gym_partially_observable_grid
import random
from collections import Counter

import aalpy.paths
from aalpy.learning_algs.stochastic_passive.ActiveAleriga import Sampler
from aalpy.learning_algs import run_active_Alergia
from aalpy.utils import save_automaton_to_file

from prism_schedulers import PrismInterface
from utils import get_initial_data, test_model, process_output, CookieDomain

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"


class EpsGreedySampler(Sampler):
    def __init__(self, input_al, eps=0.1, num_new_samples=2000, min_seq_len=10, max_seq_len=50):
        self.eps = eps
        self.new_samples = num_new_samples
        self.input_al = input_al
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.scheduler_freq_counter = Counter()
        self.sampling_round = 0

    def sample(self, env, model):
        self.sampling_round += 1
        new_data = []

        reward_states = set()
        for s in model.states:
            if '_r_' in s.output and 'neg' not in s.output or 'GOAL' in s.output:
                reward_states.add(s.output)
                if s.output not in self.scheduler_freq_counter.keys():
                    self.scheduler_freq_counter[s.output] += 1

        reward_states = list(reward_states)

        schedulers = {rs: PrismInterface(rs, model) for rs in reward_states}
        # In some weird cases this could prevent crash
        schedulers = {k: v for k, v in schedulers.items() if v is not None}

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
            env.reset()
            if not ignore_scheduler:
                scheduler.reset()
            continue_random = ignore_scheduler
            for _ in range(random.randint(self.min_seq_len, self.max_seq_len)):
                if not continue_random and random.random() < 1 - self.eps:
                    i = scheduler.get_input()
                    if i is None:
                        i = random.choice(self.input_al)
                        continue_random = True
                else:
                    i = random.choice(self.input_al)

                encoded_i = env.actions_dict[i]
                o, r, _, _ = env.step(encoded_i)
                o = process_output(env, o, r)
                sample.append((i, o))

                if r == env.goal_reward:
                    break

                # once reward state is reached, continue doing completely random sampling
                if not continue_random and not ignore_scheduler:
                    continue_random = True if o == scheduler.destination else not scheduler.step_to(i, o)

            new_data.append(sample)

        return new_data


def active_passive_experiment(exp_name,
                              world,
                              force_determinism=False,
                              indicate_slip=False,
                              indicate_wall=False,
                              is_partially_obs=True,
                              one_time_rewards=True,
                              initial_sample_num=10000,
                              min_seq_len=10,
                              max_seq_len=50,
                              active_passive_iterations=10,
                              sampler_eps_value=0.1,
                              num_new_samples=2000,
                              test_episodes=100):
    if exp_name == 'cookie_domain':
        env = CookieDomain()
    else:
        env = gym.make(id='poge-v1',
                       world_file_path=world,
                       force_determinism=force_determinism,
                       indicate_slip=indicate_slip,
                       is_partially_obs=is_partially_obs,
                       indicate_wall=indicate_wall,
                       one_time_rewards=one_time_rewards)

    input_al = list(env.actions_dict.keys())

    data = get_initial_data(env,
                            input_al,
                            initial_sample_num=initial_sample_num,
                            min_seq_len=min_seq_len,
                            max_seq_len=max_seq_len)

    sampler = EpsGreedySampler(input_al,
                               eps=sampler_eps_value,
                               num_new_samples=num_new_samples,
                               min_seq_len=min_seq_len,
                               max_seq_len=max_seq_len)

    final_model = run_active_Alergia(data=data, sul=env, sampler=sampler, n_iter=active_passive_iterations)

    print(f'Final model size: {final_model.size}')
    save_automaton_to_file(final_model, f'passive_active_{exp_name}')

    test_model(final_model, env, input_al, num_episodes=test_episodes)


def experiments_setup(exp_name):
    if exp_name == 'confusing_big_gravity':
        active_passive_experiment(
            exp_name='confusing_big_gravity',
            world='worlds/confusing_big_gravity.txt',
            initial_sample_num=10000,
            active_passive_iterations=10,
            num_new_samples=4000)
    if exp_name == 'world1':
        active_passive_experiment(
            exp_name='world1',
            world='worlds/world1.txt',
            initial_sample_num=5000,
            active_passive_iterations=5,
            num_new_samples=2000)
    if exp_name == 'cookie_domain':
        active_passive_experiment(
            exp_name='cookie_domain',
            world='...',
            initial_sample_num=20000,
            active_passive_iterations=10,
            num_new_samples=2000,
            min_seq_len=20,
            max_seq_len=75)
    # TODO add other interesting experiments


if __name__ == '__main__':
    experiments_setup('cookie_domain')
