import gym
from gym import spaces

import sys
from random import choices
from copy import deepcopy


class StochasticTile:
    def __init__(self, rule_id):
        self.rule_id = rule_id
        self.behaviour = dict()

    def add_stochastic_action(self, action, new_action_probabilities):
        self.behaviour[action] = new_action_probabilities
        assert round(sum([p[1] for p in new_action_probabilities]), 5) == 1.0

    def get_action(self, action):
        if action not in self.behaviour.keys():
            return action
        actions = [a[0] for a in self.behaviour[action]]
        prob_dist = [a[1] for a in self.behaviour[action]]

        new_action = choices(actions, prob_dist, k=1)[0]

        return new_action


class PartiallyObservableWorld(gym.Env):
    def __init__(self, world_file_path, force_determinism=False, indicate_slip=False, is_partially_obs=True, max_ep_len=100):
        self.actions_dict = { 'up':0, 'down':1, 'left':2, 'right':3}
        self.actions = [0,1,2,3]

        self.world, self.abstract_world = None, None
        self.get_world_from_file(world_file_path)

        # Map of stochastic tiles, where each tile is identified by rule_id
        self.rules = dict()
        # Map of locations to rule_ids, that is, tile has stochastic behaviour
        self.stochastic_tile = dict()

        # If true, once the executed action is not the same as the desired action,
        # 'slip' will be added to abstract output
        self.indicate_slip = indicate_slip
        self.last_action_slip = False

        # Indicate whether observations will be abstracted or will they be x-y coordinates
        self.is_partially_obs = is_partially_obs

        if not force_determinism:  # This option exist if you want to make a stochastic env. deterministic
            self.get_rules(world_file_path)

        self.initial_location = None
        self.player_location = None
        self.goal_location = None

        # Episode lenght
        self.max_ep_len = max_ep_len
        self.step_counter = 0

        self.construct_world()

        # Action and Observation Space
        self.one_hot_2_state_map, self.one_hot_2_state_map = None, None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self._get_obs_space())

    def get_world_from_file(self, world_file_path):
        file = open(world_file_path, 'r')
        self.world = [list(l.strip()) for l in file.readlines()]
        file.close()
        abstraction_path = world_file_path.split('.txt')
        abstraction_path.extend('_abstraction.txt')
        abstraction_path = ''.join(abstraction_path)
        file = open(abstraction_path, 'r')
        self.abstract_world = [list(l.strip()) for l in file.readlines()]
        file.close()

    def get_rules(self, world_file_path):
        rules_path = world_file_path.split('.txt')
        rules_path.extend('_rules.txt')
        rules_path = ''.join(rules_path)
        try:
            file = open(rules_path, 'r')
        except OSError:
            # Environment is deterministic as no rules file is present
            return
        rules_lines = [list(l.strip()) for l in file.readlines()]
        file.close()

        # Extract the rules layout
        rule_world = []
        for ind, l in enumerate(rules_lines):
            if not l:
                continue
            if ind <= len(self.world):
                rule_world.append(l)
            else:
                self._parse_rule(l)

        for i, x in enumerate(rule_world):
            for y in x:
                if y in self.rules.keys():
                    tile_xy = (i, x.index(y))
                    self.stochastic_tile[tile_xy] = y

    def _parse_rule(self, rule):
        rule = ''.join(rule)
        rule = rule.replace(" ", '')
        rule_parts = rule.split('-')
        rule_id = rule_parts[0]
        rule_action = self.actions_dict[rule_parts[1]]
        rule_mappings = rule_parts[2]
        rule_mappings = rule_mappings.lstrip('[').rstrip(']')
        if rule_id not in self.rules.keys():
            self.rules[rule_id] = StochasticTile(rule_id)

        action_prob_pairs = []
        for action_prob in rule_mappings.split(','):
            ap = action_prob.split(':')
            action = self.actions_dict[ap[0]]
            prob = float(ap[1])
            action_prob_pairs.append((action, prob))

        self.rules[rule_id].add_stochastic_action(rule_action, action_prob_pairs)

    def _get_obs_space(self):
        self.state_2_one_hot_map = {}
        counter = 0
        abstract_symbols = set()
        world_to_process = self.world if not self.is_partially_obs else self.abstract_world
        for x, row in enumerate(world_to_process):
            for y, tile in enumerate(row):
                if tile not in {'#', 'D'}:
                    if self.is_partially_obs and tile not in abstract_symbols:
                        abstract_symbols.add(tile)
                        self.state_2_one_hot_map[tile] = counter
                        counter += 1
                    elif not self.is_partially_obs:
                        self.state_2_one_hot_map[(x,y)] = counter
                        counter += 1
        #for rule in self.rules:
        #    for item.
        self.one_hot_2_state_map = {v:k for k, v in self.state_2_one_hot_map.items()}
        return counter

    def encode(self, state):
        return self.state_2_one_hot_map[state]

    def decode(self, one_hot_enc):
        return self.one_hot_2_state_map[one_hot_enc]

    def construct_world(self):
        for i, l in enumerate(self.world):
            if 'E' in l:
                self.player_location = (i, l.index('E'))
                self.initial_location = (i, l.index('E'))
                self.world[self.player_location[0]][self.player_location[1]] = ' '
            if 'G' in l:
                self.goal_location = (i, l.index('G'))

        assert self.player_location and self.goal_location

    def get_abstraction(self):
        return self.abstract_world[self.player_location[0]][self.player_location[1]]

    def step(self, action):
        assert action in self.actions

        self.step_counter += 1
        new_location = self._get_new_location(action)
        if self.world[new_location[0]][new_location[1]] == '#':
            observation = self.get_observation()
            done = True if self.step_counter >= self.max_ep_len else False
            return self.encode(observation), 0, done, {}
            #return '#', 0, False, {}

        if self.world[new_location[0]][new_location[1]] == '*':
            # TODO update like #
            return self.encode((new_location[0], new_location[1])), -1, True, {}
            #return '*', -1, True, {}

        # If you open the door, perform that step once more and enter new room
        if self.world[new_location[0]][new_location[1]] == 'D':
            self.player_location = new_location
            new_location = self._get_new_location(action)

        # Update player location
        self.player_location = new_location

        # Reward is reached if goal is reached. This terminates the episode.
        reward = 1 if self.player_location == self.goal_location else 0
        done = 1 if reward or self.step_counter >= self.max_ep_len else 0
        observation = self.get_observation()

        return self.encode(observation), reward, done, {}

    def render(self, mode='human'):
        world_copy = deepcopy(self.world)
        world_copy[self.player_location[0]][self.player_location[1]] = 'E'
        for l in world_copy:
            print("".join(l))

    def get_observation(self):
        if self.is_partially_obs:
            if self.indicate_slip and self.last_action_slip:
                observation = self.get_abstraction() + '_slip'
            else:
                observation = self.get_abstraction()
        else:
            observation = self.player_location
        return observation

    def reset(self):
        self.step_counter = 0
        self.player_location = self.initial_location[0], self.initial_location[1]
        return self.encode(self.get_observation())

    def _get_new_location(self, action):
        old_action = action
        self.last_action_slip = False
        if self.player_location in self.stochastic_tile.keys():
            action = self.rules[self.stochastic_tile[self.player_location]].get_action(action)
        if old_action != action:
            self.last_action_slip = True
        if action == 0: # up
            return self.player_location[0] - 1, self.player_location[1]
        if action == 1: # down
            return self.player_location[0] + 1, self.player_location[1]
        if action == 2: #left
            return self.player_location[0], self.player_location[1] - 1
        if action == 3: #right
            return self.player_location[0], self.player_location[1] + 1

    def play(self):
        while True:
            sys.stdout.flush()
            action = input('Action: ')
            o = self.step(action)
            self.render()
            print(o)
