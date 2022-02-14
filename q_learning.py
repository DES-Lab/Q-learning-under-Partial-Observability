import random

import gym
import gym_partially_observable_grid
import numpy as np

# Make environment deterministic even if it is stochastic
force_determinism = False
# Add slip to the observation set (action failed)
indicate_slip = True
# Use abstraction/partial observability. If set to False, (x,y) coordinates will be used as outputs
is_partially_obs = True

env = gym.make('poge-v1', world_file_path='worlds/world0.txt',
               force_determinism=force_determinism,
               indicate_slip=indicate_slip,
               is_partially_obs=is_partially_obs)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

num_training_episodes = 10000

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, num_training_episodes + 1):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    while not done:

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]

        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -1:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        print(f"Episode: {i}")

print("Training finished.\n")

total_epochs, total_penalties = 0, 0
episodes = 100

goals_reached = 0
for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -1:
            penalties += 1
        if reward and done:
            goals_reached += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Total Number of Goal reached: {goals_reached}")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
