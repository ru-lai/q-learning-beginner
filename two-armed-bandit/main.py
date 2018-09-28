import tensorflow as tf
import numpy as np

bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)

def pullBandit(bandit):
    # Return a random number
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

tf.reset_default_graph()

weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)

responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

update = optimizer.minimize(loss)

# number of episodes to train the agent on
total_episodes = 1000

total_reward = np.zeros(num_bandits) # set up scoreboard for bandits, starts at a 1 x n matrix of zeros, where n is the number of bandits
e = 0.1 # the chance of taking a random action

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)

    reward = pullBandit(bandits[action]) # choose one bandit to pick the reward from

    # update the network weights
    _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={reward_holder: [reward], action_holder: action})

    # update our tally of scores (to see which bandit has been pulled the most times)
    total_reward[action] += reward
    if i % 50 == 0:
        print('Running reward for the {num_bandits} bandits: {total_reward}'.format(num_bandits = num_bandits, total_reward = total_reward))

