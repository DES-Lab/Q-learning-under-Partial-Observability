import random
import sys
from collections import defaultdict

import numpy as np
from aalpy.learning_algs import run_JAlergia
from aalpy.utils import save_automaton_to_file

from utils import get_initial_data, add_statistics_to_file, writeSamplesToFile, deleteSampleFile
from world_repository import get_world


def get_sorted_outputs(mdp):
    return sorted(list(set([s.output for s in mdp.states])))

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
                 alergia_model_type='mdp',
                 alpha=0.1,
                 gamma=0.9,
                 early_stopping_threshold=None,
                 freeze_after_ep=0,
                 verbose=False,
                 re_init_epsilon=True,
                 linear_epsilon=False):

        # self.automaton_model = aut_model
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
        self.alergia_model_type = alergia_model_type

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
        self.sorted_actions = sorted(list(aut_model.get_input_alphabet()))
        self.sorted_obs = get_sorted_outputs(aut_model)

        # helper variables and q_table
        # self.model_state_ids = dict([(v, k) for k, v in enumerate(aut_model.states)])
        # self.n_model_states = len(aut_model.states)
        # self.acc_seq_ids = dict()
        # self.state_to_acc_seq = self.shortest_seqs_to_states()
        # self.extend_acc_seq_ids()
        # potentially have only n_states*2 x |actions|
        # self.q_table = np.zeros([self.abstract_observation_space * len(self.acc_seq_ids) * 2, self.action_space])
        self.last_automaton_model = None
        self.automaton_model = None
        self.state_to_acc_seq = None
        self.acc_seq_ids = dict()
        self.q_table = np.zeros([1,self.action_space])
        self.unknown_q_table = np.zeros([1,self.action_space])
        self.update_model(aut_model)
        self.model_state = None
        self.unknown_model_state = False
        self.re_init_epsilon = re_init_epsilon


    def shortest_seqs_to_states(self):
        seq_map = dict()
        states_todo = list(self.automaton_model.states)
        existing_seq_map = defaultdict(list)
        for existing_seq in self.acc_seq_ids.keys():

            reached_state = self.get_state_after_seq(self.automaton_model,existing_seq)
            if reached_state in states_todo:
                existing_seq_map[reached_state].append(existing_seq)
                # states_todo.remove(reached_state)
        for s,seqs in existing_seq_map.items():
            seqs = sorted(seqs,key=len)
            seq_map[s] = seqs[0]
            states_todo.remove(s)

        self.automaton_model.reset_to_initial()
        for s in states_todo:
            seq = self.shortest_seq_to_state(s)
            if seq is not None:
                seq_map[s] = seq # TODO check why we may have unreachable states
        return seq_map

    def shortest_seq_to_state(self, target_state):
        def state_for_obs(transitions, obs):
            for t in transitions:
                if t[0].output == obs:
                    return t[0]
            return None

        explored = []
        queue = [(self.automaton_model.initial_state,[self.automaton_model.initial_state.output])]

        if self.automaton_model.initial_state == target_state:
            return tuple([self.automaton_model.initial_state.output])

        while queue:
            (state,path) = queue.pop(0)
            # node = path[-1]
            if state not in explored:
                # neighbours = node.transitions.values()
                # for neighbour in neighbours:
                for action in self.sorted_actions:
                    for obs in self.sorted_obs:
                    # sorted_neighbors = sorted(list(state.transitions[action]), key=lambda t : t[1])
                    # for neighbor in sorted_neighbors:
                    #     neighbor = neighbor[0]
                        neighbor = state_for_obs(state.transitions[action],obs)
                        # obs = neighbor.output
                        if neighbor:
                            new_path = list(path)
                            new_path.extend([action,obs])
                            # return path if neighbour is goal
                            if neighbor == target_state:
                                return tuple(new_path) #tuple(new_path[1::2])
                            else:
                                queue.append((neighbor, new_path))
                                # acc_seq = new_path[:-1]
                                # inputs = []
                                # for ind, state in enumerate(acc_seq):
                                #     inputs.append(next(key for key, value in state.transitions.items()
                                #                        if value == new_path[ind + 1]))
                                # return tuple(inputs)

                # mark node as explored
                explored.append(state)
        print("Unreachable state")
        return None

    def extend_acc_seq_ids(self):
        for acc_seq in self.state_to_acc_seq.values():
            if acc_seq not in self.acc_seq_ids.keys():
                if acc_seq is None:
                    print("FOO")
                    raise Exception("FOO")
                self.acc_seq_ids[acc_seq] = len(self.acc_seq_ids)

    def get_old_or_neighbor_ids(self, new_id):
        acc_seq = self.acc_seq_for_id(new_id)
        if self.last_automaton_model is None:
            return None
        state_after_acc_seq = self.get_state_after_seq(self.last_automaton_model,acc_seq)
        if state_after_acc_seq:
            shortest_seq = self.shortest_seq_to_state(state_after_acc_seq)
            if shortest_seq in self.acc_seq_ids.keys():
                return self.acc_seq_ids[shortest_seq]
        # find neighbors
        state_after_acc_seq = self.get_state_after_seq(self.automaton_model,acc_seq)
        pre_neighbors = defaultdict(list)
        for state in self.automaton_model.states:
            for (action,transitions) in state.transitions.items():
                for t in transitions:
                    if t[0] == state_after_acc_seq:
                        pre_neighbors[action].append((action,state))
        return pre_neighbors

    def get_state_after_seq(self, automaton_model, seq):
        automaton_model.reset_to_initial()
        # print(existing_seq)
        # print(list(range(1,len(existing_seq),2)))
        for i in range(1, len(seq), 2):
            action = seq[i]
            obs = seq[i + 1]
            self.automaton_model.step_to(action, obs)
        reached_state = self.automaton_model.current_state
        return reached_state

    def acc_seq_for_id(self, new_id):
        return [(seq, identifier) for (seq, identifier) in self.acc_seq_ids.items() if new_id == identifier][0][0]
    def update_model(self, new_model = None):
        """
        With all observed samples constructs the new model with the ALERGIA algorithm.
        State space of learned model is used to extend the q-table.
        """
        if new_model is None:
            writeSamplesToFile(self.automata_learning_samples)
            new_model = run_JAlergia('alergiaSamples.txt',
                                     automaton_type=self.alergia_model_type,
                                     eps=self.alergia_epsilon,
                                     path_to_jAlergia_jar='alergia.jar',
                                     heap_memory='-Xmx12g')

        # if self.alergia_model_type == 'smm':
        #     new_model = smm_to_mdp_conversion(new_model)

        # new_n_model_states = len(new_model.states)
        self.last_automaton_model = self.automaton_model
        self.automaton_model = new_model
        self.sorted_obs = get_sorted_outputs(self.automaton_model)
        self.state_to_acc_seq = self.shortest_seqs_to_states()
        self.extend_acc_seq_ids()
        # self.n_model_states = new_n_model_states
        # self.model_state_ids = dict([(v, k) for k, v in enumerate(self.automaton_model.states)])
        # potentially have only n_states*2 x |actions|
        if self.q_table.shape[0] < len(self.acc_seq_ids):
            new_q_table = np.zeros([len(self.acc_seq_ids), self.action_space])
            new_unknown_q_table = np.zeros([len(self.acc_seq_ids) * self.abstract_observation_space, self.action_space])
            new_q_table[0:self.q_table.shape[0], :] = self.q_table[:]
            new_unknown_q_table[0:self.unknown_q_table.shape[0], :] = self.unknown_q_table[:]

            mean_q = np.mean(self.q_table)
            q_std = np.std(self.q_table)
            mean_unknown_q = np.mean(self.unknown_q_table)
            q_std = np.std(self.unknown_q_table)
            old_size = self.q_table.shape[0] if self.last_automaton_model is not None else 0
            new_size = new_q_table.shape[0]
            self.q_table = new_q_table
            self.unknown_q_table = new_unknown_q_table

            # for i in range(old_size,self.q_table.shape[0]):
            #     for j in range(self.action_space):
            for new_id in range(new_size-old_size):
                old_or_neighbor_ids = self.get_old_or_neighbor_ids(new_id)
                unknown_ids = range(new_id * self.abstract_observation_space, new_id * self.abstract_observation_space + self.abstract_observation_space)
                if old_or_neighbor_ids is None:
                    self.q_table[new_id, :] = np.ones([1,self.action_space]) * mean_q
                    self.unknown_q_table[unknown_ids, :] = np.ones([1,self.action_space]) * mean_unknown_q
                elif isinstance(old_or_neighbor_ids,int):
                    self.q_table[new_id, :] = self.q_table[old_or_neighbor_ids,:]
                    unknown_old_ids = range(old_or_neighbor_ids * self.abstract_observation_space,
                                        old_or_neighbor_ids * self.abstract_observation_space + self.abstract_observation_space)
                    self.unknown_q_table[unknown_ids, :] = self.unknown_q_table[unknown_old_ids,:]
                else:
                    for action in range(self.action_space):
                        if action in old_or_neighbor_ids.keys():
                            old_ids_for_action = old_or_neighbor_ids[action]
                            self.q_table[new_id, action] = -1e100
                            self.unknown_q_table[unknown_ids, action] = -1e100
                            assert (len(old_ids_for_action) > 0)
                            for old_id in old_ids_for_action:
                                unknown_old_ids = range(old_id * self.abstract_observation_space,
                                                        old_id * self.abstract_observation_space + self.abstract_observation_space)
                                if self.q_table[old_id, action] > self.q_table[new_id, action]:
                                    self.q_table[new_id, action] = self.q_table[old_id, action]
                                self.unknown_q_table[unknown_ids, :] = np.max(self.unknown_q_table[unknown_ids, :],
                                                                              self.unknown_q_table[unknown_old_ids, :])

                                # if self.unknown_q_table[old_id, action] > self.unknown_q_table[new_id, action]:
                                #     self.unknown_q_table[new_id, action] = self.unknown_q_table[old_id, action]
                        else:
                            self.q_table[new_id,action] = mean_q
                            self.unknown_q_table[unknown_ids,action] = mean_unknown_q

            # self.q_table[old_size:,:] = np.ones([new_size-old_size,self.action_space]) * mean_q
            # self.unknown_q_table[old_size:,:] = np.ones([new_size-old_size,self.action_space]) * mean_unknown_q
            # self.q_table[old_size:,:] = np.random.normal(mean_q,q_std,[new_size-old_size,self.action_space])


        # self.q_table = np.zeros([self.abstract_observation_space * self.n_model_states * 2, self.action_space])

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
        # model_state_id = self.model_state_ids[self.model_state]
        # extended_state = model_state_id * self.abstract_observation_space + rl_state
        # if self.unknown_model_state:
        #     extended_state += self.n_model_states * self.abstract_observation_space
        model_state_id = self.acc_seq_ids[self.state_to_acc_seq[self.model_state]]
        extended_state = model_state_id #
        if self.unknown_model_state:
            extended_state = extended_state * self.abstract_observation_space + rl_state

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

    def replay_traces_(self, rl_samples):
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
    statistics = [f'POQL, {num_training_episodes}, ModelType:{agent.alergia_model_type},'
                  f'Update Interval:{agent.update_interval}'
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
        sample = [] if agent.alergia_model_type == 'smm' else ['Init']
        rl_sample = []

        while not done:
            # Choose greedy or random action
            if random.random() < agent.epsilon:
                action = env.action_space.sample()
            else:
                if agent.unknown_model_state:
                    action = np.argmax(agent.unknown_q_table[extended_state])
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
            unknown_model_state_pre = agent.unknown_model_state
            add_reward = agent.automaton_step(mdp_action, output, agent.curiosity_reward)
            # if curiosity reward is present add it to the reward
            reward = step_reward + add_reward

            # append input/action and output in a automata learning sample
            sample.append((mdp_action, output))

            # update the extended q-table
            next_extended_state = agent.get_extended_state(next_state)
            q_table_pre = agent.q_table
            q_table_post = agent.q_table
            if unknown_model_state_pre:
                q_table_pre = agent.unknown_q_table
            if agent.unknown_model_state:
                q_table_post = agent.unknown_q_table

            old_value = q_table_pre[extended_state, action]
            next_max = np.max(q_table_post[next_extended_state])
            new_value = (1 - agent.alpha) * old_value + agent.alpha * (
                    reward + agent.gamma * next_max)

            q_table_pre[extended_state, action] = new_value

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
            print(f"Goal reached in {round((goal_reached_frequency / agent.update_interval) * 100, 2)} "
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
                # agent.replay_traces(rl_samples)

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

            q_table = po_rl_agent.q_table
            if po_rl_agent.unknown_model_state:
                q_table = po_rl_agent.unknown_q_table

            action = np.argmax(q_table[extended_state])

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
                     alergia_model_type='mdp',
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
    initial_samples = get_initial_data(env, input_al,
                                       initial_sample_num=initial_sample_num,
                                       min_seq_len=min_seq_len,
                                       max_seq_len=max_seq_len,
                                       is_smm=alergia_model_type == 'smm')

    deleteSampleFile()
    writeSamplesToFile(initial_samples)
    model = run_JAlergia('alergiaSamples.txt', eps=alergia_epsilon, automaton_type=alergia_model_type,
                         path_to_jAlergia_jar='alergia.jar', heap_memory='-Xmx12g')

    # if alergia_model_type == 'smm':
    #     model = smm_to_mdp_conversion(model)

    env.training_episode = 0
    agent = PartiallyObservableRlAgent(model,
                                       initial_samples,
                                       env.observation_space.n,
                                       env.action_space.n,
                                       update_interval=update_interval,
                                       initial_epsilon=initial_epsilon,
                                       target_epsilon=target_epsilon,
                                       alergia_epsilon=alergia_epsilon,
                                       alergia_model_type=alergia_model_type,
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
                                      num_training_episodes=num_training_episodes,
                                      verbose=verbose)

    evaluate(env_data, trained_agent, test_episodes, verbose=verbose)

    if verbose:
        print(f'Final model constructed during learning saved to {exp_name}.dot')
    save_automaton_to_file(agent.automaton_model, f'learned_models/{exp_name}')

    add_statistics_to_file(exp_name, statistics, statistic_interval_size=1000, subfolder='poql')

    deleteSampleFile()


def poql_experiment(exp_name, early_stopping_acc=1.01, model_type='mdp', verbose=True):
    env = get_world(exp_name)

    if env is None:
        print(f'Environment {exp_name} not found.')
        return
    alergia_eps = 0.05
    curiosity_reward = 0
    if exp_name == 'officeWorld':
        experiment_setup(exp_name,
                         env=env,
                         initial_sample_num=4000,
                         num_training_episodes=10000,
                         update_interval=1000,
                         early_stopping_threshold=early_stopping_acc,
                         freeze_after_ep=None,
                         verbose=verbose,
                         alergia_model_type=model_type,
                         test_episodes=100)
    if exp_name == 'confusingOfficeWorld':
        experiment_setup(exp_name,
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=30000,
                         update_interval=1000,
                         early_stopping_threshold=early_stopping_acc,
                         freeze_after_ep=15000,
                         re_init_epsilon=True,
                         alergia_epsilon=alergia_eps,
                         initial_epsilon=0.5,
                         curiosity_reward=curiosity_reward,
                         curiosity_reward_reduction=0.99,
                         curiosity_rew_reduction_mode='mult',
                         verbose=verbose,
                         alergia_model_type=model_type,
                         test_episodes=100)
    if exp_name == 'gravity':
        experiment_setup('gravity',
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=30000,
                         update_interval=1000,
                         early_stopping_threshold=early_stopping_acc,
                         freeze_after_ep=15000,
                         verbose=verbose,
                         test_episodes=100,
                         initial_epsilon=0.5,
                         re_init_epsilon=True,
                         alergia_model_type=model_type,
                         alergia_epsilon=alergia_eps,
                         curiosity_reward=curiosity_reward,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )
    if exp_name == 'thinMaze':
        experiment_setup(exp_name,
                         env=env,
                         initial_sample_num=10000,
                         num_training_episodes=60000,
                         update_interval=1000,
                         early_stopping_threshold=early_stopping_acc,
                         freeze_after_ep=20000,
                         verbose=verbose,
                         test_episodes=100,
                         initial_epsilon=0.4,
                         alergia_epsilon=alergia_eps,
                         re_init_epsilon=False,
                         alergia_model_type=model_type,
                         curiosity_reward=curiosity_reward,
                         curiosity_reward_reduction=0.9,
                         curiosity_rew_reduction_mode='mult'
                         )


if __name__ == '__main__':
    if len(sys.argv) == 2:
        poql_experiment(sys.argv[1], early_stopping_acc=1, model_type='mdp', verbose=True)
    else:
        print("Pass one of the following arguments to the script to run the experiment:\n  "
              "{gravity, officeWorld, confusingOfficeWorld, thinMaze}")