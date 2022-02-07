from aalpy.base import SUL
from aalpy.oracles import RandomWordEqOracle
from aalpy.learning_algs import run_Lstar, run_stochastic_Lstar
from aalpy.utils import visualize_automaton

from PartiallyObservableEnvs import PartiallyObservableWorld


class StochasticWorldSUL(SUL):
    def __init__(self, stochastic_world):
        super().__init__()
        self.world = stochastic_world

    def pre(self):
        self.world.reset()

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            return world.get_abstraction()
        return self.world.step(letter)


world = PartiallyObservableWorld('worlds/world0.txt', force_determinism=False, indicate_slip=True)
input_al = world.actions

sul = StochasticWorldSUL(world)
eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=1000, min_walk_len=5, max_walk_len=30)

#learned_model = run_Lstar(input_al, sul, eq_oracle, 'mealy')
learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle, automaton_type='smm')
visualize_automaton(learned_model)
