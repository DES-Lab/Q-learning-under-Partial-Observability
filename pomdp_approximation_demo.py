import random

from aalpy.SULs import MdpSUL
from aalpy.automata import MdpState, Mdp
from aalpy.automata.StochasticMealyMachine import smm_to_mdp_conversion
from aalpy.oracles import RandomWalkEqOracle
from aalpy.learning_algs import run_stochastic_Lstar, run_Alergia
from aalpy.utils import visualize_automaton


def get_counter_pomdp():
    q0 = MdpState("q0", "1")
    q1 = MdpState("q1", "1")
    q2 = MdpState("q2", "1")
    q3 = MdpState("q3", "1")
    q4 = MdpState("q4", "2")

    q0.transitions['a'].append((q0, 1))
    q0.transitions['b'].append((q1, 0.8))
    q0.transitions['b'].append((q2, 0.2))

    q1.transitions['a'].append((q2, 1))
    q1.transitions['b'].append((q2, 1))

    q2.transitions['a'].append((q3, 0.3))
    q2.transitions['a'].append((q4, 0.7))
    q2.transitions['b'].append((q4, 1))

    q3.transitions['a'].append((q4, 1))
    q3.transitions['b'].append((q4, 1))

    q4.transitions['a'].append((q4, 1))
    q4.transitions['b'].append((q4, 1))

    return Mdp(q0, [q0, q1, q2, q3, q4])


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


pomdp = get_counter_pomdp()
visualize_automaton(pomdp)
input_al = pomdp.get_input_alphabet()
sul = MdpSUL(pomdp)


def passive():
    data = []
    for _ in range(10000):
        sample = []
        sul.pre()
        for _ in range(20):
            i = random.choice(input_al)
            o = sul.step(i)
            sample.append((i, o))
        data.append(sample)

    model = run_Alergia(data, automaton_type='smm')
    model = smm_to_mdp_conversion(model)
    visualize_automaton(model, path="PassivePOMDPApprix")


def active():
    eq_oracle = RandomWalkEqOracle(input_al, sul)

    learned_model = run_stochastic_Lstar(input_al, sul, eq_oracle, automaton_type='mdp', min_rounds=100,
                                         strategy='chi2')

    visualize_automaton(learned_model, path="ActivePOMDPApprox")


if __name__ == '__main__':
    passive()
    # active()
