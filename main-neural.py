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
