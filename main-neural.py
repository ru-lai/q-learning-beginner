# Neural Net solution to the frozenlake game
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

# establish the feed-forward part of the network Resource: https://towardsdatascience.com/deep-learning-feedforward-neural-network-26a6705dbdc7
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Obtain loss by taking the sum of squared difference between target and Prediction values
# https://en.wikipedia.org/wiki/Residual_sum_of_squares
# Simple explanation for the sum of squared errors (SSE):
#   Take the actual value outputted by the final nodes,
#   subtract that value by the label / target value
#   Square it to make it positive and amplify larger errors in order to reflect outliers
#   Add all of them up to see how the network did as a whole.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# train the network
init = tf.initialize_all_variables()

# Set learning parameters
y = 0.99
e = 0.1
num_episodes = 2000
# Lists for total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset the environment and get the new observation
        s = env.reset()
        rAll = 0
        end = False
        j = 0
        # The Q network
        while j < 99:
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s + 1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get a new state and reward from the environment
            state1, reward, end, _ = env.step(a[0])
            # Get the Q value by feeding it through the network
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[state1:state1 + 1]})
            # Get the max Q value and set the target value for our agents chosen action
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = reward + y * maxQ1

            # train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s+1], nextQ:targetQ})
            rAll += reward
            s = state1
            if end == True:
                # Reduce the probability of random action as training continues
                e = 1./((i / 50) + 10)
                break

        jList.append(j)
        rList.append(rAll)

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
