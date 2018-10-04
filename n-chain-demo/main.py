import numpy as np
import gym
import keras

env = gym.make('NChain-v0')
env.reset()
# NChain environment comprises two actions: 0 and 1, move forward and move back
# A Naive agent that just sums up the rewards earned in running through all the episodes
def naive_sum_reward_agent(env, num_episodes=500):
    # Since there are two possible state / action combinations, store an index for each one.
    # So the matrix / table will look like the following:
    # [[s0a0, s0a1],
    #  [s1a0, s1a1],
    #  ...
    #  ]
    r_table = np.zeros((5, 2))
    for g in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if np.sum(r_table[state, :]) == 0:
                # if the table has no rewards for any actions, 
                # Select actions randomly between 0 and 1
                action = np.random.randint(0, 2)
            else:
                # select the action with the highest reward
                action = np.argmax(r_table[state, :])
            new_state, reward, done, _ = env.step(action)
            r_table[state, action] += reward
            state = new_state
    return r_table

print(naive_sum_reward_agent(env))
