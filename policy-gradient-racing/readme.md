# policy-gradient-racing

> A demonstration of using policy gradients in reinforcement learning

Inspiration and example found from the book Hands-On Reinforcement Learning in Python

## Install
I encountered an issue installing the universe package to my OSX computer.  This occurred because I didn't have the libjpeg library installed to system (plus some others).
I resolved this dependency issue by running the following command: 
```
$ brew install libjpeg-turbo
```

The python output recommended running the following on Ubuntu:
```
$ sudo apt-get install libjpeg-turbo8-dev
```

However, I didn't run this on an Ubuntu system, so I can't verify whether or not it will work.

## Stuck in Dependency Hell i.e. Just a Step Too Slow
I guess one of the biggest issues with a quickly moving field is that the environments move quick as well.
Utilizing a specific book, I learned how the policy gradient and some RL environment programming works.  However, by the time I actually implemented (copied) and learned (understood) the code, the universe import and environemnts were already deprecated.
Because of this, I needed to actually look for a comparable game that can be used to train the agent.

