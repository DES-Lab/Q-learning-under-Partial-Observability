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


class PartiallyObservableWorld:
    def __init__(self, world_file_path, force_determinism=False, indicate_slip=False):
        self.actions = ['up', 'down', 'left', 'right']

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

        if not force_determinism:  # This option exist if you want to make a stochastic env. deterministic
            self.get_rules(world_file_path)

        self.initial_location = None
        self.player_location = None
        self.goal_location = None

        self.construct_world()

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
        rule_action = rule_parts[1]
        rule_mappings = rule_parts[2]
        rule_mappings = rule_mappings.lstrip('[').rstrip(']')
        if rule_id not in self.rules.keys():
            self.rules[rule_id] = StochasticTile(rule_id)

        action_prob_pairs = []
        for action_prob in rule_mappings.split(','):
            ap = action_prob.split(':')
            action = ap[0]
            prob = float(ap[1])
            action_prob_pairs.append((action, prob))

        self.rules[rule_id].add_stochastic_action(rule_action, action_prob_pairs)

    def construct_world(self):
        for i, l in enumerate(self.world):
            if 'E' in l:
                self.player_location = (i, l.index('E'))
                self.initial_location = (i, l.index('E'))
                self.world[self.player_location[0]][self.player_location[1]] = ' '
            if 'G' in l:
                self.goal_location = [i, l.index('G')]

        assert self.player_location and self.goal_location

    def get_abstraction(self):
        return self.abstract_world[self.player_location[0]][self.player_location[1]]

    def step(self, action, abstract_output=True):
        assert action in self.actions

        new_location = self._get_new_location(action)
        if self.world[new_location[0]][new_location[1]] == '#':
            return '#'

        # If you open the door, perform that step once more and enter new room
        if self.world[new_location[0]][new_location[1]] == 'D':
            self.player_location = new_location
            new_location = self._get_new_location(action)

        self.player_location = new_location
        if abstract_output:
            if self.indicate_slip and self.last_action_slip:
                return self.get_abstraction() + '_slip'
            return self.get_abstraction()
        return self.player_location

    def render(self):
        world_copy = deepcopy(self.world)
        world_copy[self.player_location[0]][self.player_location[1]] = 'E'
        for l in world_copy:
            print("".join(l))

    def reset(self):
        self.player_location = self.initial_location[0], self.initial_location[1]

    def _get_new_location(self, action):
        old_action = action
        self.last_action_slip = False
        if self.player_location in self.stochastic_tile.keys():
            action = self.rules[self.stochastic_tile[self.player_location]].get_action(action)
        if old_action != action:
            self.last_action_slip = True
        if action == 'up':
            return self.player_location[0] - 1, self.player_location[1]
        if action == 'down':
            return self.player_location[0] + 1, self.player_location[1]
        if action == 'left':
            return self.player_location[0], self.player_location[1] - 1
        if action == 'right':
            return self.player_location[0], self.player_location[1] + 1

    def play(self):
        while True:
            sys.stdout.flush()
            action = input('Action: ')
            o = self.step(action, abstract_output=True)
            self.render()
            print(o)
