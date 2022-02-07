import random

import gym
import gym_partially_observable_grid
import numpy as np

env = gym.make('poge-v1', world_file_path='worlds/world0.txt', force_determinism=False, indicate_slip=False,
               is_partially_obs=True)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 10000):
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
        if reward == 1:
            goals_reached += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Total Number of Goal reached: {goals_reached}")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
