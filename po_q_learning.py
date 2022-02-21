import random

import gym
import gym_partially_observable_grid

from aalpy.learning_algs import run_Alergia
from aalpy.utils import load_automaton_from_file

import numpy as np

# Make environment deterministic even if it is stochastic
from utils import StochasticWorldSUL, get_initial_data

force_determinism = False
# Add slip to the observation set (action failed). Only necessary if is_partially_obs is set to True AND you want
# the underlying system to behave like deterministic MDP.
indicate_slip = True
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = True
# If one_time_rewards is set to True, reward in single location will be obtained only once per episode.
# Otherwise, reward will be given every time
one_time_rewards = True

env = gym.make(id='poge-v1',
               world_file_path='worlds/world2.txt',
               force_determinism=force_determinism,
               indicate_slip=indicate_slip,
               is_partially_obs=is_partially_obs,
               one_time_rewards=one_time_rewards,
               max_ep_len=100,
               step_penalty=0.1)

# Hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

sul = StochasticWorldSUL(env)

# static properties of environment
reverse_action_dict = dict([(v, k) for k, v in env.actions_dict.items()])
input_al = list(env.actions_dict.keys())
min_seq_len = 3
max_seq_len = 20
n_obs = env.observation_space.n


class PoRlData:
    def __init__(self, aut_model, aal_samples):
        self.aut_model = aut_model
        self.aal_samples = aal_samples
        self.model_state_ids = dict([(v, k) for k, v in enumerate(aut_model.states)])
        self.n_model_states = len(aut_model.states)
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([n_obs * self.n_model_states * 2, env.action_space.n])

    def update(self,rl_samples):
        new_model = run_Alergia(self.aal_samples, automaton_type="mdp")
        new_n_model_states = len(new_model.states)
        print(f"Learned MDP with {new_n_model_states} states")
        print("Updating Q table")
        self.aut_model = new_model
        self.n_model_states = new_n_model_states
        self.model_state_ids = dict([(v, k) for k, v in enumerate(self.aut_model.states)])
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([env.observation_space.n * self.n_model_states * 2, env.action_space.n])
        replay_traces(rl_samples, self.aut_model, self.model_state_ids, self.q_table)
        print("Replayed traces")


def initialize():
    print("Starting initial automata learning phase")
    aal_samples = get_initial_data(sul, input_al, initial_sample_num=10000, min_seq_len=min_seq_len, max_seq_len=max_seq_len)
    model = run_Alergia(aal_samples, automaton_type="mdp")
    po_rl_data = PoRlData(model,aal_samples)
    print(f"Learned MDP with {po_rl_data.n_model_states} states")
    return po_rl_data


def replay_traces(rl_samples, model, model_state_ids, q_table):
    for sample in rl_samples:
        model.reset_to_initial()
        model_state = model.current_state
        unknown_model_state = False
        for (state, action, next_state, reward, mdp_action, output) in sample:

            model_state_id = model_state_ids[model_state]
            extended_state = model_state_id * n_obs + state
            if unknown_model_state:
                extended_state += len(model_state_ids) * n_obs

            # MDP step
            step_possible = False
            if not unknown_model_state:
                step_possible = model.step_to(mdp_action, output) is not None
            if not step_possible and not unknown_model_state:
                # print(f"Unknown model states after {steps} steps")
                unknown_model_state = True
            elif step_possible:
                model_state = model.current_state

            model_state_id = model_state_ids[model_state]
            next_extended_state = model_state_id * n_obs + next_state
            if unknown_model_state:
                next_extended_state += len(model_state_ids) * n_obs

            old_value = q_table[extended_state, action]
            next_max = np.max(q_table[next_extended_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[extended_state, action] = new_value


def train(init_po_rl_data : PoRlData, num_training_episodes = 30000):
    rl_samples = []
    po_rl_data = init_po_rl_data
    goal_reached_frequency = 0
    for i in range(1, num_training_episodes + 1):
        state = env.reset()
        po_rl_data.aut_model.reset_to_initial()
        model_state = po_rl_data.aut_model.current_state
        unknown_model_state = False

        epochs, penalties, reward, = 0, 0, 0
        done = False
        steps = 0
        model_state_id = po_rl_data.model_state_ids[model_state]
        extended_state = model_state_id * n_obs + state
        if unknown_model_state:
            extended_state += po_rl_data.n_model_states * n_obs
        sample = ['Init']
        rl_sample = []
        while not done:
            # print(f"state {state},{model_state_id}: {extended_state}")
            action = env.action_space.sample() if random.random() < epsilon else np.argmax(po_rl_data.q_table[extended_state])
            steps += 1
            next_state, reward, done, info = env.step(action)
            if reward == 10 and done:
                goal_reached_frequency += 1
            # step in MDP
            output = env.decode(next_state)
            mdp_action = reverse_action_dict[action]
            # print(f"Performed {mdp_action}")
            step_possible = False
            if not unknown_model_state:
                step_possible = po_rl_data.aut_model.step_to(mdp_action, output) is not None
            if not step_possible and not unknown_model_state:
                # print(f"Unknown model states after {steps} steps")
                unknown_model_state = True
            elif step_possible:
                model_state = po_rl_data.aut_model.current_state

            sample.append((mdp_action, output))

            model_state_id = po_rl_data.model_state_ids[model_state]
            next_extended_state = model_state_id * n_obs + next_state
            if unknown_model_state:
                next_extended_state += po_rl_data.n_model_states * n_obs

            old_value = po_rl_data.q_table[extended_state, action]
            next_max = np.max(po_rl_data.q_table[next_extended_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            po_rl_data.q_table[extended_state, action] = new_value
            rl_sample.append((state, action, next_state, reward, mdp_action, output))

            if reward == -1:
                penalties += 1

            state = next_state
            extended_state = next_extended_state
            epochs += 1
        po_rl_data.aal_samples.append(sample)
        rl_samples.append(rl_sample)
        if i % 100 == 0:
            print(f"Episode: {i}")
        if i % 1000 == 0:
            print(f"Goal reached in {goal_reached_frequency / 10} percent of the cases in last 1000 ep.")
            if goal_reached_frequency / 10 > 95:
                break
            po_rl_data.update(rl_samples)
            goal_reached_frequency = 0
    print("Training finished.\n")
    return po_rl_data


def evaluate(po_rl_data : PoRlData, episodes=100):
    total_epochs = 0
    goals_reached = 0

    for eval_ep in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        po_rl_data.aut_model.reset_to_initial()
        model_state = po_rl_data.aut_model.current_state
        unknown_model_state = False

        done = False
        steps = 0
        while not done:
            steps += 1
            model_state_id = po_rl_data.model_state_ids[model_state]
            extended_state = model_state_id * n_obs + state
            if unknown_model_state:
                extended_state += po_rl_data.n_model_states * n_obs

            action = np.argmax(po_rl_data.q_table[extended_state])
            state, reward, done, info = env.step(action)
            # step in MDP
            output = env.decode(state)
            mdp_action = reverse_action_dict[action]
            # print(f"Performed {mdp_action}")
            step_possible = False
            if not unknown_model_state:
                step_possible = po_rl_data.aut_model.step_to(mdp_action, output) is not None
            if not step_possible and not unknown_model_state:
                print(f"{eval_ep}: Unknown model states after {steps} steps")
                print(output, mdp_action)
                unknown_model_state = True
            elif step_possible:
                model_state = po_rl_data.aut_model.current_state

            if reward == 10 and done:
                goals_reached += 1

            epochs += 1

        total_epochs += epochs

    print(f"Results after {episodes} episodes:")
    print(f"Total Number of Goal reached: {goals_reached}")
    print(f"Average timesteps per episode: {total_epochs / episodes}")


# TODO: I think it is better if we have first function, like train and evaluate defined, then enviroment and helper stuff
# then you call train(), eval()
initial_data = initialize()
trained_data = train(initial_data)
evaluate(trained_data)
