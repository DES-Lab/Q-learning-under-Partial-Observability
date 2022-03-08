import random

import gym
import gym_partially_observable_grid

from aalpy.learning_algs import run_Alergia

import numpy as np
from aalpy.utils import save_automaton_to_file

from utils import get_initial_data


class PartiallyObservableRlAgent:
    """
    Reinforcement learning agent that can make decisions in (extremely) partially observable environment.
    """
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

        self.automaton_model = aut_model
        self.automata_learning_samples = aal_samples
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

        # helper variables and q_table
        self.model_state_ids = dict([(v, k) for k, v in enumerate(aut_model.states)])
        self.n_model_states = len(aut_model.states)
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([self.abstract_observation_space * self.n_model_states * 2, self.action_space])
        self.model_state = None
        self.unknown_model_state = False

    def update_model(self):
        """
        With all observed samples constructs the new model with the ALERGIA algorithm.
        State space of learned model is used to extend the q-table.
        """
        new_model = run_Alergia(self.automata_learning_samples, automaton_type="mdp", print_info=self.verbose)
        new_n_model_states = len(new_model.states)

        self.automaton_model = new_model
        self.n_model_states = new_n_model_states
        self.model_state_ids = dict([(v, k) for k, v in enumerate(self.automaton_model.states)])
        # potentially have only n_states*2 x |actions|
        self.q_table = np.zeros([self.abstract_observation_space * self.n_model_states * 2, self.action_space])

    def reset_aut(self):
        """
        Reset automaton state to the initial_state.
        This is done so that we can trace the episode starting from the initial state in the model.
        """
        self.unknown_model_state = False
        self.automaton_model.reset_to_initial()
        self.model_state = self.automaton_model.current_state

    def get_extended_state(self, rl_state):
        """
        Given the state_obtained by the environment and current automaton state,
        return the extended state which combines them.
        """
        model_state_id = self.model_state_ids[self.model_state]
        extended_state = model_state_id * self.abstract_observation_space + rl_state
        if self.unknown_model_state:
            extended_state += self.n_model_states * self.abstract_observation_space
        return extended_state

    def automaton_step(self, mdp_action, output, curiosity_reward):
        """
        Perform a step on the learned automaton. Based on performed action and observed output.
        Returns the curiosity reward if action/output combination is not defined for the current state, and if
        the curiosity reward is defined.

        """
        step_possible = False
        additional_reward = 0
        if not self.unknown_model_state:
            step_possible = self.automaton_model.step_to(mdp_action, output) is not None
        if not step_possible and not self.unknown_model_state:
            if self.curiosity_enabled:
                additional_reward += curiosity_reward
            self.unknown_model_state = True
        elif step_possible:
            self.model_state = self.automaton_model.current_state
        return additional_reward

    def replay_traces(self, rl_samples):
        """
        Relays all episodes on the extended q-table. This way values in the extended
        q-table are updated without interaction with the environment.
        """
        if self.verbose:
            print('Replaying traces')
        for sample in rl_samples:
            self.reset_aut()

            for (state, action, next_state, reward, mdp_action, output) in sample:
                extended_state = self.get_extended_state(state)

                # MDP step
                curiosity_reward = self.automaton_step(mdp_action, output, self.curiosity_reward)
                next_extended_state = self.get_extended_state(next_state)

                reward += curiosity_reward

                old_value = self.q_table[extended_state, action]
                next_max = np.max(self.q_table[next_extended_state])
                new_value = (1 - self.alpha) * old_value + self.alpha * (
                        reward + self.gamma * next_max)
                self.q_table[extended_state, action] = new_value

    def set_curiosity_params(self, init_curiosity_rew=2, curiosity_rew_reduction=0.9,
                             curiosity_rew_reduction_mode='mult'):
        """
        Define values for curiosity parameter.
        If this function is called during training curiosity rewards will be added.
        """
        self.curiosity_enabled = True
        self.init_curiosity_rew = init_curiosity_rew
        self.curiosity_reward = init_curiosity_rew
        self.curiosity_rew_reduction = curiosity_rew_reduction
        self.curiosity_rew_reduction_mode = curiosity_rew_reduction_mode

    def update_epsilon(self, current_episode, num_training_episodes, target_value=0.1):
        """
        Updates the current value of epsilon. Value of epsilon decreases as the training progresses.
        This ensured more random sampling at the beginning, and as the training progresses sampling will become more
        and more optimal (up to target_value).
        """
        if self.freeze_automaton_after:
            divisor = min(self.freeze_automaton_after, num_training_episodes)
        else:
            divisor = num_training_episodes

        decrement = (self.initial_epsilon - target_value) / divisor
        if self.freeze_automaton_after:
            if current_episode < self.freeze_automaton_after:
                self.epsilon -= decrement
            else:
                self.epsilon = target_value
        else:
            self.epsilon -= decrement


def train(env_data, agent, num_training_episodes, verbose=True):
    """
    Trains a partially-observable q-agent.
    """
    if verbose:
        print('Training started')

    env, input_al, reverse_action_dict, env.observation_space.n = env_data

    rl_samples = []
    goal_reached_frequency = 0
    for episode in range(1, num_training_episodes + 1):

        # reset environment and agent(its automaton) state
        state = env.reset()
        agent.reset_aut()

        done = False
        steps = 0

        extended_state = agent.get_extended_state(state)
        sample = ['Init']
        rl_sample = []

        while not done:
            # Choose greedy or random action
            if random.random() < agent.epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.q_table[extended_state])

            # perform action on the env
            next_state, step_reward, done, info = env.step(action)
            steps += 1

            # is goal reached?
            if step_reward == env.goal_reward and done:
                goal_reached_frequency += 1

            output = env.decode(next_state)
            mdp_action = reverse_action_dict[action]

            # perform a step in the learned automaton (in the agent)
            add_reward = agent.automaton_step(mdp_action, output, agent.curiosity_reward)
            # if curiosity reward is present add it to the reward
            reward = step_reward + add_reward

            # append input/action and output in a automata learning sample
            sample.append((mdp_action, output))

            # update the extended q-table
            next_extended_state = agent.get_extended_state(next_state)
            old_value = agent.q_table[extended_state, action]
            next_max = np.max(agent.q_table[next_extended_state])
            new_value = (1 - agent.alpha) * old_value + agent.alpha * (
                    reward + agent.gamma * next_max)

            agent.q_table[extended_state, action] = new_value

            # add step to the replay sample
            # ignore curiosity reward to avid bias in later stages
            rl_sample.append((state, action, next_state, step_reward, mdp_action, output))

            state = next_state
            extended_state = agent.get_extended_state(state)

        # update epsilon value
        agent.update_epsilon(episode, num_training_episodes)

        # If freezing is present, remove the curiously reward
        if agent.freeze_automaton_after and episode > agent.freeze_automaton_after:
            agent.curiosity_reward = 0

        # add episode to memory for automata learning and q-table replaying
        agent.automata_learning_samples.append(sample)
        rl_samples.append(rl_sample)

        # Update interval (for model learning and q-table extension)
        if episode % agent.update_interval == 0:
            if agent.curiosity_enabled:
                # update curiosity values
                if agent.curiosity_rew_reduction_mode == "minus":
                    agent.curiosity_reward -= agent.curiosity_rew_reduction
                else:
                    agent.curiosity_reward *= agent.curiosity_rew_reduction

                if agent.curiosity_reward < 0:
                    agent.curiosity_reward = 0

            # Early stopping
            if agent.early_stopping_threshold and (
                    goal_reached_frequency / agent.update_interval) >= agent.early_stopping_threshold:
                print('Early stopping threshold exceeded, training stopped.')
                break

            # if freezing is enabled do not update the model
            if agent.freeze_automaton_after is not None and episode > agent.freeze_automaton_after:
                pass
            else:
                if verbose:
                    print(f'============== Update Interval {episode} ==============')
                    print(f"Goal reached in {round((goal_reached_frequency / agent.update_interval) * 100, 2)} "
                          f"percent of the cases in last {agent.update_interval} ep.")

                    print('============== Updating model ==============')

                # Update the model by running ALERGIA on all samples
                agent.update_model()
                # Based on the updated model extend the q-table
                agent.replay_traces(rl_samples)

            goal_reached_frequency = 0

    print("Training finished.\n")
    return agent


def evaluate(env_data, po_rl_agent: PartiallyObservableRlAgent, episodes=100):
    """
    Evaluates the partially-observable q-agent.
    """
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

            po_rl_agent.automaton_step(mdp_action, output, 0)

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
                     indicate_wall=True,
                     one_time_rewards=True,
                     initial_sample_num=10000,
                     num_training_episodes=30000,
                     min_seq_len=10,
                     max_seq_len=50,
                     update_interval=1000,
                     # initial epsilon value that will decrease to 0.1 during training
                     epsilon=0.9,
                     alpha=0.1,
                     gamma=0.9,
                     early_stopping_threshold=None,
                     freeze_after_ep=None,
                     verbose=False,
                     test_episodes=100,
                     curiosity_reward=None,
                     curiosity_reward_reduction=None,
                     curiosity_rew_reduction_mode=None):
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

    agent = PartiallyObservableRlAgent(model,
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

    if curiosity_reward:
        assert curiosity_reward_reduction is not None and curiosity_rew_reduction_mode is not None
        agent.set_curiosity_params(curiosity_reward, curiosity_reward_reduction, curiosity_rew_reduction_mode)

    trained_agent = train(env_data,
                          agent,
                          num_training_episodes=num_training_episodes)

    evaluate(env_data, trained_agent, test_episodes)

    if verbose:
        print(f'Final model constucted during learning saved to {exp_name}.dot')
    save_automaton_to_file(agent.automaton_model, exp_name)


def experiment(exp_name):
    if exp_name == 'world1':
        experiment_setup('world1',
                         'worlds/world1+rew.txt',
                         is_partially_obs=True,
                         force_determinism=False,
                         goal_reward=10,
                         step_penalty=0.1,
                         max_ep_len=100,
                         one_time_rewards=True,
                         initial_sample_num=4000,
                         num_training_episodes=10000,
                         update_interval=1000,
                         early_stopping_threshold=None,
                         freeze_after_ep=None,
                         verbose=True,
                         test_episodes=100)
    if exp_name == 'world2':
        experiment_setup('world2',
                         'worlds/world2.txt',
                         is_partially_obs=True,
                         force_determinism=False,
                         goal_reward=10,
                         step_penalty=0.1,
                         max_ep_len=150,
                         one_time_rewards=True,
                         initial_sample_num=10000,
                         num_training_episodes=30000,
                         min_seq_len=30,
                         max_seq_len=100,
                         update_interval=1000,
                         early_stopping_threshold=None,
                         freeze_after_ep=15000,
                         verbose=True,
                         test_episodes=100,
                         epsilon=0.5,
                         curiosity_reward=10,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult')
    if exp_name == 'gravity':
        experiment_setup('gravity',
                         'worlds/confusing_big_gravity.txt',
                         is_partially_obs=True,
                         force_determinism=False,
                         goal_reward=10,
                         step_penalty=0.1,
                         max_ep_len=100,
                         one_time_rewards=True,
                         initial_sample_num=10000,
                         num_training_episodes=20000,
                         update_interval=1000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=10000,
                         verbose=True,
                         test_episodes=100,
                         epsilon=0.5,
                         curiosity_reward=10,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )


if __name__ == '__main__':
    experiment('gravity')
