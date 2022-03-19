import random

from aalpy.learning_algs import run_Alergia

import numpy as np
from aalpy.utils import save_automaton_to_file

from utils import get_initial_data, add_statistics_to_file
from world_repository import get_world


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
                 initial_epsilon=0.9,
                 target_epsilon=0.1,
                 alergia_epsilon=0.005,
                 alpha=0.1,
                 gamma=0.9,
                 early_stopping_threshold=None,
                 freeze_after_ep=0,
                 verbose=False,
                 re_init_epsilon=True,
                 linear_epsilon=False):

        self.automaton_model = aut_model
        self.automata_learning_samples = aal_samples
        # parameters
        self.initial_epsilon = initial_epsilon
        self.target_epsilon = target_epsilon
        self.epsilon = initial_epsilon
        self.alergia_epsilon = alergia_epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.freeze_automaton_after = freeze_after_ep
        self.update_interval = update_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.linear_epsilon = linear_epsilon

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
        self.re_init_epsilon = re_init_epsilon

    def update_model(self):
        """
        With all observed samples constructs the new model with the ALERGIA algorithm.
        State space of learned model is used to extend the q-table.
        """
        new_model = run_Alergia(self.automata_learning_samples,
                                automaton_type="mdp",
                                eps=self.alergia_epsilon,
                                print_info=self.verbose)
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

    def update_epsilon(self, current_episode, num_training_episodes, target_value):
        """
        Updates the current value of epsilon. Value of epsilon decreases as the training progresses.
        This ensured more random sampling at the beginning, and as the training progresses sampling will become more
        and more optimal (up to target_value).
        """
        if self.freeze_automaton_after:
            if self.re_init_epsilon and current_episode == self.freeze_automaton_after:
                self.epsilon = self.initial_epsilon
            if self.re_init_epsilon and current_episode > self.freeze_automaton_after:
                divisor = (num_training_episodes - self.freeze_automaton_after)
            else:
                divisor = min(self.freeze_automaton_after, num_training_episodes)
        else:
            divisor = num_training_episodes

        if self.linear_epsilon:
            decrement = (self.initial_epsilon - target_value) / divisor
        else:
            decrement = (target_value / self.initial_epsilon) ** (1 / divisor)
        if self.freeze_automaton_after and not self.re_init_epsilon:
            if current_episode < self.freeze_automaton_after:
                if self.linear_epsilon:
                    self.epsilon -= decrement
                else:
                    self.epsilon *= decrement
            else:
                self.epsilon = target_value
        else:
            if self.linear_epsilon:
                self.epsilon -= decrement
            else:
                self.epsilon *= decrement


def train(env_data, agent, num_training_episodes, verbose=True, statistics_interval=1000):
    """
    Trains a partially-observable q-agent.
    """
    statistics = [f'POQL, {num_training_episodes}, Update Interval:{agent.update_interval}'
                  f'InitEps:{agent.initial_epsilon}, TargetEps:{agent.target_epsilon},'
                  f'AlergiaEps:{agent.alergia_epsilon},Curiosity:{agent.curiosity_reward},'
                  f'Freeze:{agent.freeze_automaton_after}']

    if verbose:
        print('Training started')

    env, input_al, reverse_action_dict, env.observation_space.n = env_data
    frozen = False
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

            # # is goal reached?
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
        agent.update_epsilon(episode, num_training_episodes, agent.target_epsilon)

        # If freezing is present, remove the curiously reward
        if agent.freeze_automaton_after and episode > agent.freeze_automaton_after:
            agent.curiosity_reward = 0

        # add episode to memory for automata learning and q-table replaying
        agent.automata_learning_samples.append(sample)
        rl_samples.append(rl_sample)

        # For statistics
        if episode % statistics_interval == 0:
            statistics.append(evaluate(env_data, agent, verbose=False))

        # Update interval (for model learning and q-table extension)
        if episode % agent.update_interval == 0:
            # Early stopping
            goal_reached, _, _ = evaluate(env_data, agent, verbose=True)
            if agent.early_stopping_threshold:
                if goal_reached / 100 >= agent.early_stopping_threshold:
                    print('Early stopping threshold exceeded, training stopped.')
                    break

            print(f"Eps: {agent.epsilon}")
            if agent.curiosity_enabled:
                # update curiosity values
                if agent.curiosity_rew_reduction_mode == "minus":
                    agent.curiosity_reward -= agent.curiosity_rew_reduction
                else:
                    agent.curiosity_reward *= agent.curiosity_rew_reduction

                if agent.curiosity_reward < 0:
                    agent.curiosity_reward = 0
            print(f"Goal reached in {round((goal_reached_frequency / agent.update_interval) * 100,2)} "
                  f"percent of the cases during training.")

            # if freezing is enabled do not update the model
            if frozen:
                pass
            else:
                if verbose:
                    print(f'============== Update Interval {episode} ==============')
                    print(f"Goal reached in {goal_reached} % of test episodes.")
                    print('============== Updating model ==============')

                # Update the model by running ALERGIA on all samples
                agent.update_model()
                # Based on the updated model extend the q-table
                agent.replay_traces(rl_samples)

                if agent.freeze_automaton_after is not None and episode >= agent.freeze_automaton_after:
                    if verbose:
                        print('Freezing automaton.')
                    frozen = True

            goal_reached_frequency = 0

    print("Training finished.\n")
    return agent, statistics


def evaluate(env_data, po_rl_agent: PartiallyObservableRlAgent, episodes=100, verbose=True):
    """
    Evaluates the partially-observable q-agent.
    """
    env, input_al, reverse_action_dict, env.observation_space.n = env_data

    total_steps = 0
    goals_reached = 0
    calculative_reward = 0

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
            calculative_reward += reward

            # step in MDP
            output = env.decode(state)
            mdp_action = reverse_action_dict[action]

            po_rl_agent.automaton_step(mdp_action, output, 0)

            if reward == env.goal_reward and done:
                goals_reached += 1

    avg_rew = round(calculative_reward / episodes, 2)
    avg_step = round(total_steps / episodes, 2)
    if verbose:
        print(f"Evaluation performed on {episodes} episodes.")
        print(f"Total Number of Goal reached  : {goals_reached}")
        print(f"Average reward per episode    : {avg_rew}")
        print(f"Average timesteps per episode : {avg_step}")

    return goals_reached, avg_rew, avg_step


def experiment_setup(exp_name,
                     env,
                     initial_sample_num=10000,
                     num_training_episodes=30000,
                     min_seq_len=10,
                     max_seq_len=50,
                     update_interval=1000,
                     initial_epsilon=0.9,
                     target_epsilon=0.1,
                     alergia_epsilon=0.005,
                     alpha=0.1,
                     gamma=0.9,
                     early_stopping_threshold=None,
                     freeze_after_ep=None,
                     re_init_epsilon=False,
                     verbose=False,
                     test_episodes=100,
                     curiosity_reward=None,
                     curiosity_reward_reduction=None,
                     curiosity_rew_reduction_mode=None):

    input_al = list(env.actions_dict.keys())
    reverse_action_dict = dict([(v, k) for k, v in env.actions_dict.items()])
    env_data = (env, input_al, reverse_action_dict, env.observation_space.n)

    if verbose:
        print('Initial sampling and model construction started')
    initial_samples = get_initial_data(env, input_al, initial_sample_num=initial_sample_num,
                                       min_seq_len=min_seq_len, max_seq_len=max_seq_len)
    model = run_Alergia(initial_samples, eps=alergia_epsilon, automaton_type="mdp", print_info=verbose)

    env.training_episode = 0
    agent = PartiallyObservableRlAgent(model,
                                       initial_samples,
                                       env.observation_space.n,
                                       env.action_space.n,
                                       update_interval=update_interval,
                                       initial_epsilon=initial_epsilon,
                                       target_epsilon=target_epsilon,
                                       alergia_epsilon=alergia_epsilon,
                                       alpha=alpha,
                                       gamma=gamma,
                                       early_stopping_threshold=early_stopping_threshold,
                                       freeze_after_ep=freeze_after_ep,
                                       verbose=verbose,
                                       re_init_epsilon=re_init_epsilon)

    if curiosity_reward:
        assert curiosity_reward_reduction is not None and curiosity_rew_reduction_mode is not None
        agent.set_curiosity_params(curiosity_reward, curiosity_reward_reduction, curiosity_rew_reduction_mode)

    trained_agent, statistics = train(env_data,
                                      agent,
                                      num_training_episodes=num_training_episodes)

    evaluate(env_data, trained_agent, test_episodes)

    if verbose:
        print(f'Final model constructed during learning saved to {exp_name}.dot')
    save_automaton_to_file(agent.automaton_model, f'learned_models/{exp_name}')

    add_statistics_to_file(exp_name, statistics, statistic_interval_size=1000, subfolder='poql')


def experiment(exp_name):
    env = get_world(exp_name)
    if env is None:
        print(f'Environment {exp_name} not found.')
        return
    if exp_name == 'world1':
        experiment_setup('world1',
                         env=env,
                         initial_sample_num=4000,
                         num_training_episodes=10000,
                         update_interval=1000,
                         early_stopping_threshold=None,
                         freeze_after_ep=None,
                         verbose=True,
                         test_episodes=100)
    if exp_name == 'world2+rew':
        experiment_setup('world2+rew',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=120000,
                         min_seq_len=30,
                         max_seq_len=100,
                         update_interval=1000,
                         early_stopping_threshold=None,
                         freeze_after_ep=40000,
                         verbose=True,
                         test_episodes=100,
                         re_init_epsilon=True,
                         initial_epsilon=0.9,
                         curiosity_reward=None,
                         curiosity_reward_reduction=0.99,
                         curiosity_rew_reduction_mode='mult')
    if exp_name == 'world2':
        experiment_setup('world2',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=80000,
                         min_seq_len=30,
                         max_seq_len=100,
                         update_interval=2000,
                         early_stopping_threshold=None,
                         freeze_after_ep=20000,
                         verbose=True,
                         test_episodes=100,
                         initial_epsilon=0.9,
                         re_init_epsilon=True,
                         curiosity_reward=None,
                         curiosity_reward_reduction=0.95,
                         curiosity_rew_reduction_mode='mult')
    if exp_name == 'gravity':
        experiment_setup('gravity',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=20000,
                         update_interval=1000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=10000,
                         verbose=True,
                         test_episodes=100,
                         re_init_epsilon=True,
                         initial_epsilon=0.9,
                         curiosity_reward=None,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'gravity2':
        experiment_setup('gravity2',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=20000,
                         update_interval=1000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=10000,
                         verbose=True,
                         test_episodes=100,
                         initial_epsilon=0.9,
                         re_init_epsilon=True,
                         curiosity_reward=None,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'big_office_one_time_rew' or exp_name == 'big_office_permanent_rew':
        experiment_setup(exp_name,
                         env=env,
                         num_training_episodes=30000,
                         update_interval=2000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=16000,
                         verbose=True,
                         test_episodes=100,
                         initial_epsilon=0.9,
                         re_init_epsilon=True,
                         curiosity_reward=10,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'misleading_office_one_time':
        experiment_setup('misleading_office_one_time',
                         env=env,
                         num_training_episodes=40000,
                         update_interval=2000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=20000,
                         verbose=True,
                         test_episodes=100,
                         alergia_epsilon=0.1,
                         initial_epsilon=0.9,
                         re_init_epsilon=True,
                         curiosity_reward=10,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'corridor-rew':
        experiment_setup('corridor-rew',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=30000,
                         update_interval=1000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=12000,
                         verbose=True,
                         test_episodes=100,
                         initial_epsilon=0.9,
                         re_init_epsilon=True,
                         curiosity_reward=None,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'corridor_one_time_rew':
        experiment_setup('corridor_one_time_rew',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=30000,
                         update_interval=1000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=12000,
                         verbose=True,
                         test_episodes=100,
                         initial_epsilon=0.9,
                         re_init_epsilon=True,
                         curiosity_reward=None,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'corner':
        experiment_setup('corner',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=30000,
                         update_interval=1000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=12000,
                         verbose=True,
                         test_episodes=100,
                         initial_epsilon=0.9,
                         re_init_epsilon=False,
                         curiosity_reward=5,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'thin_maze':
        experiment_setup('thin_maze',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=30000,
                         update_interval=1000,
                         early_stopping_threshold=0.98,
                         freeze_after_ep=12000,
                         verbose=True,
                         test_episodes=100,
                         initial_epsilon=0.9,
                         re_init_epsilon=False,
                         curiosity_reward=5,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )


if __name__ == '__main__':
    experiment('thin_maze')
