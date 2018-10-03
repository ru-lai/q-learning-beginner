import gym
import universe
import random

env = gym.make('flashgames.NeonRace-v0')
env.configure()

# Move left
left = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', True), ('KeyEvent', 'ArrowRight', False)]

# Move right
right = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', True)]

# Move forward
forward = [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowLeft', False), ('KeyEvent', 'ArrowRight', False)]

# Variable to see if we should turn or not
shouldTurn = 0

# store all of the rewards in a list
rewards = []

# set a threshold to the amount of rewards we store in our rewards list
bufffer_size = 100

# set the current action; initialize with a default: forward
action = forward

while True:
    turn -= 1
    if shouldTurn <= 0:
        action = forward
        shouldTurn = 0

    action_n = [action for ob in observation_n]
    # in each step we record the: 
    # state of the car (observation_n),
    # reward from the previous action (reward_n) ## This implies that the car is not stuck.
    # boolean that is true or false if the game has ended, (done_n)
    # a var used to help debug (info_n)
    observation_n, reward_n, done_n, info_n = env.step(action_n)
    print(f"reward for this turn: {reward_n}")
    rewards += [reward_n[0]]

    if len(rewards) >= bufffer_size:
        # take the average of rewards
        mean = sum(rewards) / len(rewards)

        # if it is 0, we probably were stuck somewhere and need to break out
        if mean == 0:
            turn = 20
            if random.random() < 0.5:
                action = right
            else:
                action = left


        rewards = []

    env.render()


