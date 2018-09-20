import gym
import numpy as np

env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])
learn_rate = .8
y = .95

num_episodes = 2000

rList = []
for i in range(num_episodes):
    new_observation = env.reset()
    r_all = 0
    d = False
    j = 0
    # QTable Learning Algorithm
    while j < 99:
        j += 1
        # Choose action (with noise) from Q table
        action = np.argmax(Q[new_observation, :] + np.random.randn(1, env.action_space.n) * (1./(i + 1)))
        print("j: ",  j)
        print("action: ", action)
        
        state1, reward, end, _ = env.step(action)
        # Update Q-Table
        Q[new_observation, action] = Q[new_observation, action] + learn_rate * (reward + y * np.max(Q[state1, :]) - Q[new_observation, action])
        if reward != 0:
            env.render()

        r_all += reward
        new_observation = state1
        if end == True:
            break
    rList.append(r_all)

print("Score over time: " + str(sum(rList) / num_episodes))  
print("Times won: ", len(list(filter(lambda x: x > 0, rList))))
