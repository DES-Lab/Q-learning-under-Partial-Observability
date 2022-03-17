import random

import gym
import gym_partially_observable_grid

from aalpy.learning_algs import run_Alergia
from aalpy.utils import load_automaton_from_file

import numpy as np

from utils import get_initial_data


class PoRLParameters:
    def __init__(self, epsilon, alpha, gamma, training_episodes, update_interval, early_stopping_threshold,
                 curiosity_reward, freeze_automaton_after):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.curiosity_reward = curiosity_reward
        self.training_episodes = training_episodes
        self.freeze_automaton_after = freeze_automaton_after
        self.update_interval = update_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.epsilon_update_scheme = lambda old_eps, ep: old_eps

    def update_eps(self, ep):
        self.epsilon = self.epsilon_update_scheme(self.epsilon, ep)


class PoRLConfig:
    def __init__(self, init_epsilon, init_curiosity_rew,
                 curiosity_rew_reduction, curiosity_rew_reduction_mode
                 ):
        self.init_epsilon = init_epsilon
        self.init_curiosity_rew = init_curiosity_rew
        self.curiosity_rew_reduction = curiosity_rew_reduction
        self.curiosity_rew_reduction_mode = curiosity_rew_reduction_mode

    def linear_decrease_to_freeze(self, start_epsilon, target_value, po_rl_parameters: PoRLParameters):
        start_value = start_epsilon
        decrement = (start_value - target_value) / min(po_rl_parameters.freeze_automaton_after,
                                                       po_rl_parameters.training_episodes)
        update_scheme = lambda old_eps, ep: old_eps - decrement if ep < po_rl_parameters.freeze_automaton_after else target_value
        return update_scheme


force_determinism = False
# Add slip to the observation set (action failed). Only necessary if is_partially_obs is set to True AND you want
# the underlying system to behave like deterministic MDP.
indicate_slip = False
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = True
# If one_time_rewards is set to True, reward in single location will be obtained only once per episode.
# Otherwise, reward will be given every time
one_time_rewards = True

env = gym.make(id='poge-v1',
               world_file_path='../worlds/world3.txt',
               force_determinism=force_determinism,
               indicate_slip=indicate_slip,
               is_partially_obs=is_partially_obs,
               one_time_rewards=one_time_rewards,
               max_ep_len=150,
               goal_reward=10,  # 60 for gravity, 10 for world2
               step_penalty=0.1,  # 0.5 for gravity, 0.1 for world 2
               indicate_wall=True)
# env.play()
# static properties of environment
reverse_action_dict = dict([(v, k) for k, v in env.actions_dict.items()])
input_al = list(env.actions_dict.keys())

n_obs = env.observation_space.n
min_seq_len = 10
max_seq_len = 50

# # Hyper parameters
# alpha = 0.1
# gamma = 0.9
# epsilon = 0.5

config = PoRLConfig(init_epsilon=0.9,
                    init_curiosity_rew=2,  # 2 for world2
                    curiosity_rew_reduction=0.9,
                    curiosity_rew_reduction_mode="mult")

parameters = PoRLParameters(epsilon=config.init_epsilon, alpha=0.1, gamma=0.9, training_episodes=30000,
                            update_interval=1000, early_stopping_threshold=None,
                            curiosity_reward=config.init_curiosity_rew, freeze_automaton_after=15000)

parameters.epsilon_update_scheme = config.linear_decrease_to_freeze(config.init_epsilon, 0.1, parameters)


class PoRlAgent:
    def __init__(self, aut_model, aal_samples):
        self.aut_model = aut_model
        self.aal_samples = aal_samples
        self.model_state_ids = dict([(v, k) for k, v in enumerate(aut_model.states)])
        self.n_model_states = len(aut_model.states)
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([n_obs * self.n_model_states * 2, env.action_space.n])
        self.model_state = None
        self.unknown_model_state = False

    def update(self, rl_samples, parameters: PoRLParameters):
        new_model = run_Alergia(self.aal_samples, automaton_type="mdp")
        new_n_model_states = len(new_model.states)
        print(f"Learned MDP with {new_n_model_states} states")
        print("Updating Q table")
        self.aut_model = new_model
        self.n_model_states = new_n_model_states
        self.model_state_ids = dict([(v, k) for k, v in enumerate(self.aut_model.states)])
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([env.observation_space.n * self.n_model_states * 2, env.action_space.n])
        self.replay_traces(rl_samples, parameters)
        print("Replayed traces")

    def reset_aut(self):
        self.unknown_model_state = False
        self.aut_model.reset_to_initial()
        self.model_state = self.aut_model.current_state

    def get_extended_state(self, rl_state):
        model_state_id = self.model_state_ids[self.model_state]
        extended_state = model_state_id * n_obs + rl_state
        if self.unknown_model_state:
            extended_state += self.n_model_states * n_obs
        return extended_state

    def perform_aut_step(self, mdp_action, output, curiosity_reward):
        step_possible = False
        additional_reward = 0
        if not self.unknown_model_state:
            step_possible = self.aut_model.step_to(mdp_action, output) is not None
        if not step_possible and not self.unknown_model_state:
            # print(f"Unknown model states after {steps} steps")
            # TODO remove me if curiosity does not make sense
            additional_reward += curiosity_reward
            self.unknown_model_state = True
        elif step_possible:
            self.model_state = self.aut_model.current_state
        return additional_reward

    def replay_traces(self, rl_samples, parameters: PoRLParameters):
        for sample in rl_samples:
            self.reset_aut()
            for (state, action, next_state, reward, mdp_action, output) in sample:
                extended_state = self.get_extended_state(state)

                # MDP step
                add_reward = self.perform_aut_step(mdp_action, output, parameters.curiosity_reward)
                next_extended_state = self.get_extended_state(next_state)
                # reward += add_reward
                reward += add_reward

                old_value = self.q_table[extended_state, action]
                next_max = np.max(self.q_table[next_extended_state])
                new_value = (1 - parameters.alpha) * old_value + parameters.alpha * (
                            reward + parameters.gamma * next_max)
                self.q_table[extended_state, action] = new_value


def initialize():
    print("Starting initial automata learning phase")
    aal_samples = get_initial_data(env, input_al, initial_sample_num=10000, min_seq_len=min_seq_len,
                                   max_seq_len=max_seq_len)
    model = run_Alergia(aal_samples, automaton_type="mdp")
    po_rl_data = PoRlAgent(model, aal_samples)
    print(f"Learned MDP with {po_rl_data.n_model_states} states")
    return po_rl_data


def train(init_po_rl_agent: PoRlAgent, po_rl_config: PoRLConfig, po_rl_parameters: PoRLParameters):
    rl_samples = []
    po_rl_agent = init_po_rl_agent
    goal_reached_frequency = 0
    for episode in range(1, po_rl_parameters.training_episodes + 1):
        # if episode > po_rl_config.freeze_automaton_after:
        #     update_interval = 100000000
        #     epsilon = 0.1
        #     curiosity_reward = 0
        state = env.reset()
        po_rl_agent.reset_aut()
        epochs, penalties, reward, = 0, 0, 0
        done = False
        steps = 0
        extended_state = po_rl_agent.get_extended_state(state)
        sample = ['Init']
        rl_sample = []
        while not done:
            # print(f"state {state},{model_state_id}: {extended_state}")
            action = env.action_space.sample() if random.random() < po_rl_parameters.epsilon else np.argmax(
                po_rl_agent.q_table[extended_state])
            steps += 1
            next_state, reward, done, info = env.step(action)
            if reward == env.goal_reward and done:
                goal_reached_frequency += 1
            # step in MDP
            output = env.decode(next_state)
            mdp_action = reverse_action_dict[action]
            # print(f"Performed {mdp_action}")
            add_reward = po_rl_agent.perform_aut_step(mdp_action, output, po_rl_parameters.curiosity_reward)
            reward += add_reward

            sample.append((mdp_action, output))
            next_extended_state = po_rl_agent.get_extended_state(next_state)

            old_value = po_rl_agent.q_table[extended_state, action]
            next_max = np.max(po_rl_agent.q_table[next_extended_state])
            new_value = (1 - po_rl_parameters.alpha) * old_value + po_rl_parameters.alpha * (
                        reward + po_rl_parameters.gamma * next_max)
            po_rl_agent.q_table[extended_state, action] = new_value
            # TODO maybe subtract curiosity reward here, so we don't add it twice
            # it seems to work better without subtracting, though
            reward -= add_reward
            rl_sample.append((state, action, next_state, reward, mdp_action, output))

            if reward == -1:
                penalties += 1

            state = next_state
            extended_state = po_rl_agent.get_extended_state(state)
            epochs += 1
        po_rl_parameters.update_eps(episode)
        if episode > po_rl_parameters.freeze_automaton_after:
            po_rl_parameters.curiosity_reward = 0
        po_rl_agent.aal_samples.append(sample)
        rl_samples.append(rl_sample)
        if episode % 100 == 0:
            print(f"Episode: {episode}")
        if episode <= po_rl_parameters.freeze_automaton_after and episode % po_rl_parameters.update_interval == 0:
            # rl_samples = rl_samples[-2000:]
            if po_rl_config.curiosity_rew_reduction_mode == "minus":
                po_rl_parameters.curiosity_reward -= po_rl_config.curiosity_rew_reduction
            else:
                po_rl_parameters.curiosity_reward *= po_rl_config.curiosity_rew_reduction
            if po_rl_parameters.curiosity_reward < 0:
                po_rl_parameters.curiosity_reward = 0

            print(
                f"Goal reached in {(goal_reached_frequency / po_rl_parameters.update_interval) * 100} percent of the cases in last 1000 ep.")
            if po_rl_parameters.early_stopping_threshold and goal_reached_frequency / 10 >= po_rl_parameters.early_stopping_threshold:
                break
            po_rl_agent.update(rl_samples, po_rl_parameters)
            goal_reached_frequency = 0
    print("Training finished.\n")
    print(po_rl_agent.q_table)
    return po_rl_agent


def evaluate(po_rl_agent: PoRlAgent, episodes=100):
    total_epochs = 0
    goals_reached = 0
    coordinate_list = []
    for eval_ep in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        po_rl_agent.reset_aut()

        done = False
        steps = 0
        coordinates = [env.player_location]
        while not done:
            steps += 1
            extended_state = po_rl_agent.get_extended_state(state)

            action = np.argmax(po_rl_agent.q_table[extended_state])

            state, reward, done, info = env.step(action)
            coordinates.append(env.player_location)
            # step in MDP
            output = env.decode(state)
            mdp_action = reverse_action_dict[action]

            print(f"{steps}: {mdp_action} : {output}")
            # print(f"Performed {mdp_action}")
            po_rl_agent.perform_aut_step(mdp_action, output, 0)

            if reward == env.goal_reward and done:
                goals_reached += 1

            epochs += 1
        coordinate_list.append(coordinates)
        total_epochs += epochs
    print(f"Results after {episodes} episodes:")
    print(f"Total Number of Goal reached: {goals_reached}")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    return coordinate_list


initial_agent = initialize()
trained_agent = train(initial_agent, po_rl_config=config, po_rl_parameters=parameters)
coordinates = evaluate(trained_agent)
from utils import visualize_episode

visualize_episode(env, coordinates[0])
