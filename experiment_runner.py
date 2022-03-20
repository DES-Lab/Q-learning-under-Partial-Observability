from stable_baselines import DQN, A2C, ACKTR, ACER, PPO2
from stable_baselines.common.callbacks import StopTrainingOnRewardThreshold

from partially_observable_q_learning import poql_experiment
from reccurent_policy_comp import lstm_experiment
from stacked_frames_comp import stacked_experiment
from world_repository import get_world, get_all_world_ids

world_ids = get_all_world_ids()

early_stopping_acc = 1.01  # put to 0.98 if you want to stop when goal is reached in 98% of test episodes


def run_poql_experiments():
    repeat_each = 5
    num_runs = len(world_ids) * repeat_each
    i = 0
    for w in world_ids:
        for _ in range(repeat_each):
            poql_experiment(w, early_stopping_acc=early_stopping_acc, verbose=False)
            i += 1
            print(f"Percentage of complete POQL experiments: {round(i / num_runs, 2)}%")


def run_stacked_experiments():
    algs = [DQN, A2C, ACKTR]
    repeat_each = 5
    num_runs = len(algs) * len(world_ids) * repeat_each
    i = 0
    for w in world_ids:
        for alg in algs:
            for _ in range(repeat_each):
                poge = get_world(w)
                # 30k episodes as default, should be more than enough for all then cut in statistics
                stacked_experiment(w, poge, alg, 30000 * poge.max_ep_len, num_frames=5,
                                   early_stopping_acc=early_stopping_acc)
                i += 1
                print(f"Percentage of complete Stacked experiments: {round(i / num_runs, 2)}%")


def run_lstm_experiments():
    algs = [A2C, ACER, PPO2, ACKTR]
    repeat_each = 5
    num_runs = len(algs) * len(world_ids) * repeat_each
    i = 0
    for w in world_ids:
        for alg in algs:
            for _ in range(repeat_each):
                poge = get_world(w)
                # 30k episodes as default, should be more than enough for all then cut in statistics
                lstm_experiment(w, poge, alg, 30000 * poge.max_ep_len, early_stopping_acc=early_stopping_acc)
                i += 1
                print(f"Percentage of complete Stacked experiments: {round(i / num_runs, 2)}%")
