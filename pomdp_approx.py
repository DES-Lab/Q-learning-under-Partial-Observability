from aalpy.SULs import MdpSUL
from aalpy.automata import MdpState, Mdp
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_stochastic_Lstar
from aalpy.utils import visualize_automaton


def get_small_pomdp():
    q0 = MdpState("q0", "init")
    q1 = MdpState("q1", "beep")
    q2 = MdpState("q2", "beep")
    q3 = MdpState("q3", "coffee")
    q4 = MdpState("q4", "tea")

    q0.transitions['but'].append((q0, 1))
    q0.transitions['coin'].append((q1, 0.8))
    q0.transitions['coin'].append((q2, 0.2))

    q1.transitions['coin'].append((q1, 1))
    q1.transitions['but'].append((q3, 1))

    q2.transitions['coin'].append((q2, 0.3))
    q2.transitions['coin'].append((q1, 0.7))
    q2.transitions['but'].append((q4, 1))

    q3.transitions['coin'].append((q3, 1))
    q3.transitions['but'].append((q3, 1))

    q4.transitions['coin'].append((q4, 1))
    q4.transitions['but'].append((q4, 1))

    return Mdp(q0, [q0, q1, q2, q3, q4])


mdp = get_small_pomdp()
input_al = mdp.get_input_alphabet()
sul = MdpSUL(mdp)
eq_oracle = RandomWalkEqOracle(input_al, sul)

learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle, automaton_type='mdp', min_rounds=200, strategy='chi2')

visualize_automaton(learned_model)
