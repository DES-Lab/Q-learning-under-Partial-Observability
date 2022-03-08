import sys
from partially_observable_q_learning import experiment_setup

args = sys.argv[1:]

if len(args) == 0 or args[0] in {'-h', '--help'}:
    print('Mandatory params:\n\t-exp_name <name> - name of the experiment\n\t-world <path_to_world.txt> - layout file')
    print('All other params are optional. Each param is a pair of -<variable name> <value>, for example:')
    print('\t-is_partially_obs True')
    exit()

variable_map = dict()
for i in range(0, len(args), 2):
    variable_map[args[i][1:]] = args[i + 1]

# mandatory values
assert variable_map['exp_name']
assert variable_map['world']

# if not defined, default value will be obtained
is_partially_obs = variable_map.get('is_partially_obs', True)
force_determinism = variable_map.get('force_determinism', False)
goal_reward = variable_map.get('goal_reward', 10)
step_penalty = variable_map.get('step_penalty', 0.1)
max_ep_len = variable_map.get('max_ep_len', 100)
indicate_slip = variable_map.get('indicate_slip', False)
indicate_wall = variable_map.get('indicate_wall', False)
one_time_rewards = variable_map.get('one_time_rewards', True)
initial_sample_num = variable_map.get('initial_sample_num', 10000)
num_training_episodes = variable_map.get('num_training_episodes', 20000)
min_seq_len = variable_map.get('min_seq_len', 15)
max_seq_len = variable_map.get('max_seq_len', 50)
update_interval = variable_map.get('update_interval', 1000)
# initial epsilon value that will decrease to 0.1 during training
epsilon = variable_map.get('epsilon', 0.9)
alpha = variable_map.get('alpha', 0.1)
gamma = variable_map.get('gamma', 0.9)
early_stopping_threshold = variable_map.get('early_stopping_threshold', None)
freeze_after_ep = variable_map.get('freeze_after_ep', None)
verbose = True
test_episodes = variable_map.get('test_episodes', 100)
curiosity_reward = variable_map.get('curiosity_reward', None)
curiosity_reward_reduction = variable_map.get('curiosity_reward_reduction', None)
curiosity_rew_reduction_mode = variable_map.get('curiosity_rew_reduction_mode', None)

experiment_setup(exp_name=variable_map['exp_name'],
                 world=variable_map['world'],
                 is_partially_obs=is_partially_obs,
                 force_determinism=force_determinism,
                 goal_reward=goal_reward,
                 step_penalty=step_penalty,
                 max_ep_len=max_ep_len,
                 indicate_slip=indicate_slip,
                 indicate_wall=indicate_wall,
                 one_time_rewards=one_time_rewards,
                 initial_sample_num=initial_sample_num,
                 num_training_episodes=num_training_episodes,
                 min_seq_len=min_seq_len,
                 max_seq_len=max_seq_len,
                 update_interval=update_interval,
                 epsilon=epsilon,
                 alpha=alpha,
                 gamma=gamma,
                 early_stopping_threshold=early_stopping_threshold,
                 freeze_after_ep=freeze_after_ep, verbose=verbose,
                 test_episodes=test_episodes,
                 curiosity_reward=curiosity_reward,
                 curiosity_reward_reduction=curiosity_reward_reduction,
                 curiosity_rew_reduction_mode=curiosity_rew_reduction_mode,
                 )
