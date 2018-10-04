# n-chain-demo

> Three implementations of policy gradients in one demo!

Source and Inspiration: http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/

## Game Definition
In NChain, The agent can be in one of 5 states and can have 2 actions, either move forward or move back.
The most rewarding state for the agent is in the transition from state 4 to state 5.
When the agent transitions from state 4 to state 5, a reward of 10 is attributed to that move.
However, minus that particular transition, the "most valuable" move an agent can make is to move backward at any state, when it will receive 2 points and be replaced in state 0.
