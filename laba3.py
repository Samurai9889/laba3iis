import numpy as np
import random


num_states = 25
num_actions = 4


Q_table = np.zeros((num_states, num_actions))


learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
start_state = 0
goal_state = num_states - 1


def choose_action(state):
    if random.uniform(0, 1) < exploration_prob:
        return random.randint(0, num_actions - 1)
    else:

        return np.argmax(Q_table[state, :])


num_episodes = 1000

for episode in range(num_episodes):
    state = start_state
    while state != goal_state:

        action = choose_action(state)


        new_state = state + 1 \
            if action == 3 else state - 1 if action == 2 else state + 5 if action == 1 else state - 5


        if 0 <= new_state < num_states:
            reward = 1 if new_state == goal_state else 0


            Q_table[state, action] = (1 - learning_rate) * Q_table[state, action] + \
                                     learning_rate * (reward + discount_factor * np.max(Q_table[new_state, :]))


            state = new_state
        else:

            continue


state = start_state
path = [state]

while state != goal_state:
    action = np.argmax(Q_table[state, :])
    new_state = state + 1 if action == 3 else state - 1 if action == 2 else state + 5 if action == 1 else state - 5
    path.append(new_state)
    state = new_state

print("Знайдений шлях:", path)
