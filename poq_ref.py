import random

import gym
import gym_partially_observable_grid

from aalpy.learning_algs import run_Alergia

import numpy as np

from utils import get_initial_data


class PoRlAgent:
    def __init__(self,
                 aut_model,
                 aal_samples,
                 abstract_observation_space,
                 action_space,
                 update_interval=1000,
                 epsilon=0.9,
                 alpha=0.1,
                 gamma=0.9,
                 early_stopping_threshold=None,
                 freeze_after_ep=0,
                 verbose=False):
        self.aut_model = aut_model
        self.aal_samples = aal_samples
        # parameters
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.freeze_automaton_after = freeze_after_ep
        self.update_interval = update_interval
        self.early_stopping_threshold = early_stopping_threshold

        # curiosity params
        self.curiosity_enabled = False
        self.curiosity_reward = None
        self.init_curiosity_rew = None
        self.curiosity_rew_reduction = None
        self.curiosity_rew_reduction_mode = None

        # print
        self.verbose = verbose

        # constants
        self.abstract_observation_space = abstract_observation_space
        self.action_space = action_space

        self.model_state_ids = dict([(v, k) for k, v in enumerate(aut_model.states)])
        self.n_model_states = len(aut_model.states)
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([self.abstract_observation_space * self.n_model_states * 2, self.action_space])
        self.model_state = None
        self.unknown_model_state = False

    def update_model(self):
        new_model = run_Alergia(self.aal_samples, automaton_type="mdp", print_info=self.verbose)
        new_n_model_states = len(new_model.states)

        self.aut_model = new_model
        self.n_model_states = new_n_model_states
        self.model_state_ids = dict([(v, k) for k, v in enumerate(self.aut_model.states)])
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([self.abstract_observation_space * self.n_model_states * 2, self.action_space])

    def reset_aut(self):
        self.unknown_model_state = False
        self.aut_model.reset_to_initial()
        self.model_state = self.aut_model.current_state

    def get_extended_state(self, rl_state):
        model_state_id = self.model_state_ids[self.model_state]
        extended_state = model_state_id * self.abstract_observation_space + rl_state
        if self.unknown_model_state:
            extended_state += self.n_model_states * self.abstract_observation_space
        return extended_state

    def perform_aut_step(self, mdp_action, output, curiosity_reward):
        step_possible = False
        additional_reward = 0
        if not self.unknown_model_state:
            step_possible = self.aut_model.step_to(mdp_action, output) is not None
        if not step_possible and not self.unknown_model_state:
            if self.curiosity_enabled:
                additional_reward += curiosity_reward
            self.unknown_model_state = True
        elif step_possible:
            self.model_state = self.aut_model.current_state
        return additional_reward

    def replay_traces(self, rl_samples):
        if self.verbose:
            print('Replaying traces')
        for sample in rl_samples:
            self.reset_aut()

            for (state, action, next_state, reward, mdp_action, output) in sample:
                extended_state = self.get_extended_state(state)

                # MDP step
                curiosity_reward = self.perform_aut_step(mdp_action, output, self.curiosity_reward)
                next_extended_state = self.get_extended_state(next_state)

                reward += curiosity_reward

                old_value = self.q_table[extended_state, action]
                next_max = np.max(self.q_table[next_extended_state])
                new_value = (1 - self.alpha) * old_value + self.alpha * (
                        reward + self.gamma * next_max)
                self.q_table[extended_state, action] = new_value

    def set_curiosity_params(self, init_curiosity_rew=2, curiosity_rew_reduction=0.9,
                             curiosity_rew_reduction_mode='mult'):
        self.curiosity_enabled = True
        self.init_curiosity_rew = init_curiosity_rew
        self.curiosity_reward = init_curiosity_rew
        self.curiosity_rew_reduction = curiosity_rew_reduction
        self.curiosity_rew_reduction_mode = curiosity_rew_reduction_mode

    def update_epsilon(self, current_episode, num_training_episodes, target_value=0.1):
        if self.freeze_automaton_after:
            divisor = min(self.freeze_automaton_after, num_training_episodes)
        else:
            divisor = num_training_episodes

        decrement = (self.initial_epsilon - target_value) / divisor
        if self.freeze_automaton_after and current_episode < self.freeze_automaton_after:
            self.epsilon -= decrement
        else:
            self.epsilon = target_value


def train(env_data, agent, num_training_episodes, verbose=True):
    if verbose:
        print('Training started')

    env, input_al, reverse_action_dict, env.observation_space.n = env_data

    rl_samples = []
    goal_reached_frequency = 0
    for episode in range(1, num_training_episodes + 1):
        state = env.reset()
        agent.reset_aut()

        done = False
        steps = 0

        extended_state = agent.get_extended_state(state)
        sample = ['Init']
        rl_sample = []

        while not done:
            if random.random() < agent.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.q_table[extended_state])

            next_state, reward, done, info = env.step(action)
            steps += 1

            if reward == env.goal_reward and done:
                goal_reached_frequency += 1

            # step in MDP
            output = env.decode(next_state)
            mdp_action = reverse_action_dict[action]

            add_reward = agent.perform_aut_step(mdp_action, output, agent.curiosity_reward)
            reward += add_reward

            sample.append((mdp_action, output))

            next_extended_state = agent.get_extended_state(next_state)
            old_value = agent.q_table[extended_state, action]
            next_max = np.max(agent.q_table[next_extended_state])
            new_value = (1 - agent.alpha) * old_value + agent.alpha * (
                    reward + agent.gamma * next_max)

            agent.q_table[extended_state, action] = new_value

            # subtract curiosity reward here, so we don't add it twice
            # it seems to work better without subtracting, though
            reward -= add_reward

            # add step to the replay sample
            rl_sample.append((state, action, next_state, reward, mdp_action, output))

            state = next_state
            extended_state = agent.get_extended_state(state)

        # update epsilon value
        agent.update_epsilon(episode, num_training_episodes)

        if agent.freeze_automaton_after and episode > agent.freeze_automaton_after:
            agent.curiosity_reward = 0

        # add episode to memory for automata learning and q-table replaying
        agent.aal_samples.append(sample)
        rl_samples.append(rl_sample)

        if episode % agent.update_interval == 0:
            if agent.curiosity_enabled:
                if agent.curiosity_rew_reduction_mode == "minus":
                    agent.curiosity_reward -= agent.curiosity_rew_reduction
                else:
                    agent.curiosity_reward *= agent.curiosity_rew_reduction

                if agent.curiosity_reward < 0:
                    agent.curiosity_reward = 0

            if agent.early_stopping_threshold and goal_reached_frequency / 10 >= agent.early_stopping_threshold:
                break

            # if freezing is enabled do not update the model
            if agent.freeze_automaton_after is not None and episode > agent.freeze_automaton_after:
                pass
            else:
                if verbose:
                    print(f'============== Update Interval {episode} ==============')
                    print(f"Goal reached in {(goal_reached_frequency / agent.update_interval) * 100} "
                          f"percent of the cases in last 1000 ep.")

                    print('============== Updating model ==============')
                agent.update_model()
                agent.replay_traces(rl_samples)

            if verbose:
                print('============== Model updated ===============')

            goal_reached_frequency = 0

    print("Training finished.\n")
    return agent


def evaluate(env_data, po_rl_agent: PoRlAgent, episodes=100):
    env, input_al, reverse_action_dict, env.observation_space.n = env_data

    total_steps = 0
    goals_reached = 0

    for eval_ep in range(episodes):
        state = env.reset()
        po_rl_agent.reset_aut()

        done = False
        steps = 0
        while not done:
            steps += 1
            extended_state = po_rl_agent.get_extended_state(state)

            action = np.argmax(po_rl_agent.q_table[extended_state])

            state, reward, done, info = env.step(action)
            total_steps += 1

            # step in MDP
            output = env.decode(state)
            mdp_action = reverse_action_dict[action]

            po_rl_agent.perform_aut_step(mdp_action, output, 0)

            if reward == env.goal_reward and done:
                goals_reached += 1

    print(f"Results after {episodes} episodes:")
    print(f"Total Number of Goal reached: {goals_reached}")
    print(f"Average timesteps per episode: {total_steps / episodes}")


def experiment_setup(exp_name,
                     world,
                     is_partially_obs=True,
                     force_determinism=False,
                     goal_reward=10,
                     step_penalty=0.1,
                     max_ep_len=100,
                     indicate_slip=False,
                     indicate_wall=False,
                     one_time_rewards=True,
                     initial_sample_num=10000,
                     num_training_episodes=20000,
                     min_seq_len=10,
                     max_seq_len=50,
                     update_interval=1000,
                     epsilon=0.9,
                     alpha=0.1,
                     gamma=0.9,
                     early_stopping_threshold=None,
                     freeze_after_ep=None,
                     verbose=False,
                     test_episodes=100):

    env = gym.make(id='poge-v1',
                   world_file_path=world,
                   force_determinism=force_determinism,
                   indicate_slip=indicate_slip,
                   indicate_wall=indicate_wall,
                   is_partially_obs=is_partially_obs,
                   one_time_rewards=one_time_rewards,
                   max_ep_len=max_ep_len,
                   goal_reward=goal_reward,  # 60 for gravity, 10 for world2
                   step_penalty=step_penalty)  # 0.5 for gravity, 0.1 for world 2)

    input_al = list(env.actions_dict.keys())
    reverse_action_dict = dict([(v, k) for k, v in env.actions_dict.items()])
    env_data = (env, input_al, reverse_action_dict, env.observation_space.n)

    if verbose:
        print('Initial sampling and model construction started')
    initial_samples = get_initial_data(env, input_al, initial_sample_num=initial_sample_num,
                                       min_seq_len=min_seq_len, max_seq_len=max_seq_len)
    model = run_Alergia(initial_samples, automaton_type="mdp", print_info=verbose)

    agent = PoRlAgent(model,
                      initial_samples,
                      env.observation_space.n,
                      env.action_space.n,
                      update_interval=update_interval,
                      epsilon=epsilon,
                      alpha=alpha,
                      gamma=gamma,
                      early_stopping_threshold=early_stopping_threshold,
                      freeze_after_ep=freeze_after_ep,
                      verbose=verbose)

    trained_agent = train(env_data,
                          agent,
                          num_training_episodes=num_training_episodes)

    evaluate(env_data, trained_agent, test_episodes)


if __name__ == '__main__':
    experiment_setup(exp_name='test', world='worlds/world1+rew.txt', verbose=True)
