import gym
import gym_partially_observable_grid
import aalpy.paths
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion

from aalpy.base import SUL
from aalpy.oracles import RandomWordEqOracle
from aalpy.learning_algs import run_Lstar, run_stochastic_Lstar
from aalpy.utils import visualize_automaton, save_automaton_to_file, load_automaton_from_file

from prism_schedulers import PrismInterface

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"

class StochasticWorldSUL(SUL):
    def __init__(self, stochastic_world):
        super().__init__()
        self.world = stochastic_world
        self.goal_reached = False

    def pre(self):
        self.goal_reached = False
        self.world.reset()

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            return world.get_abstraction()
        output, reward, done, info = self.world.step(world.actions_dict[letter])
        if done or self.goal_reached:
            self.goal_reached = True
            return "GOAL"
        return self.world.decode(output)


# Make environment deterministic even if it is stochastic
force_determinism = False
# Add slip to the observation set (action failed)
indicate_slip = False
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = True

world = gym.make('poge-v1', world_file_path='worlds/world1.txt',
                 force_determinism=force_determinism,
                 indicate_slip=indicate_slip,
                 is_partially_obs=is_partially_obs)

input_al = list(world.actions_dict.keys())

sul = StochasticWorldSUL(world)
eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=1000, min_walk_len=5, max_walk_len=30)

learn = False

if learn:
    # learned_model = run_Lstar(input_al, sul, eq_oracle, 'mealy')
    learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle, automaton_type='smm', max_rounds=15)
    #visualize_automaton(learned_model)
    save_automaton_to_file(learned_model, 'approximate_model')
else:
    learned_model = load_automaton_from_file('approximate_model.dot', automaton_type='smm')

learned_model_mdp = smm_to_mdp_conversion(learned_model)

for s in learned_model_mdp.states:
    if str(s.output).isdigit():
        s.output = f's_{s.output}'

prism_interface = PrismInterface("GOAL", learned_model_mdp, num_steps=17)

print(prism_interface.create_mc_query())

print(prism_interface.property_val)