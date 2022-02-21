import gym
import gym_partially_observable_grid

import aalpy.paths
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import save_automaton_to_file, load_automaton_from_file

from prism_schedulers import PrismInterface
from utils import test_model, StochasticWorldSUL

aalpy.paths.path_to_prism = "C:/Program Files/prism-4.7/bin/prism.bat"

# Make environment deterministic even if it is stochastic
force_determinism = True
# Add slip to the observation set (action failed)
indicate_slip = True
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = True  # prism will not work with False (need to fix touple type)

world = gym.make(id='poge-v1',
                 world_file_path='worlds/world0.txt',
                 force_determinism=force_determinism,
                 indicate_slip=indicate_slip,
                 is_partially_obs=is_partially_obs)

input_al = list(world.actions_dict.keys())

sul = StochasticWorldSUL(world)
eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=1000, min_walk_len=10, max_walk_len=30)

# learned_model = run_Lstar(input_al, sul, eq_oracle, 'mealy')
learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle, automaton_type='smm', max_rounds=15)
# visualize_automaton(learned_model)
# save_automaton_to_file(learned_model, 'approximate_model')

learned_model_mdp = smm_to_mdp_conversion(learned_model)

prism_interface = PrismInterface("GOAL", learned_model_mdp, num_steps=17)

test_model(learned_model_mdp, sul, input_al, num_episodes=100)
