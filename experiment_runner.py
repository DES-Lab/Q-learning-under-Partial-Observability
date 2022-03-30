from stable_baselines import DQN, A2C, ACKTR, ACER, PPO2

from partially_observable_q_learning import poql_experiment
from reccurent_policy_comp import lstm_experiment
from stacked_frames_comp import stacked_experiment
from world_repository import get_world, get_all_world_ids

# world_ids = get_all_world_ids()
world_ids = ['world1', 'world1_confusing', 'simple_showcase2', 'gravity2']
# repeat each experiment repeat_each times
repeat_each = 2
# print additional info during each experiment
verbose = False
# put to 0.98 if you want to stop when goal is reached in 98% of test episodes
early_stopping_acc = 1.0
#
num_ep = 20000

def run_poql_experiments():
    num_runs = len(world_ids) * repeat_each * 2
    i = 0
    for w in world_ids:
        for _ in range(repeat_each):
            for model_type in ['mdp', 'smm']:
                poql_experiment(w, early_stopping_acc=early_stopping_acc, model_type=model_type, verbose=verbose)
                i += 1
                print(f"Percentage of complete POQL experiments: {round(i / num_runs, 2)}%")


def run_stacked_experiments():
    algs = [DQN, A2C, ACKTR]
    num_runs = len(algs) * len(world_ids) * repeat_each  # * 2 # for frame size of 5 and 10
    i = 0
    frame_size = 5
    for w in world_ids:
        for alg in algs:
            # for frame_size in [5,10]:
            for _ in range(repeat_each):
                poge = get_world(w)
                # 30k episodes as default, should be more than enough for all then cut in statistics
                stacked_experiment(w, poge, alg, num_ep * poge.max_ep_len, num_frames=frame_size,
                                   early_stopping_acc=early_stopping_acc, verbose=verbose)
                i += 1
                print(f"Percentage of complete Stacked experiments: {round(i / num_runs, 2) * 100}%")


def run_lstm_experiments():
    algs = [ACER, A2C, ACKTR] # PPO2
    num_runs = len(algs) * len(world_ids) * repeat_each
    i = 0
    for w in world_ids:
        for alg in algs:
            for _ in range(repeat_each):
                poge = get_world(w)
                # 30k episodes as default, should be more than enough for all then cut in statistics
                lstm_experiment(w, poge, alg, num_ep * poge.max_ep_len, early_stopping_acc=early_stopping_acc,
                                verbose=verbose)
                i += 1
                print(f"Percentage of complete LSTM experiments: {round(i / num_runs, 2) * 100}%")


if __name__ == '__main__':
    # run_poql_experiments()
    run_stacked_experiments()
    run_lstm_experiments()
