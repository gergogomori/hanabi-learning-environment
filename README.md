## Introduction to the project

The original Hanabi repository from DeepMind was modified in order to test how a learning agent and rule-based agents play Hanabi togeher.
Currently the focus is only on two players: one learning agent and one rule-based agent.
The first agent is always the learning agent.
Learning agents can be modifications of agents implemented in [tf-agents](https://www.tensorflow.org/agents/overview).

tf-agents has to be installed in order to run "examples/rl_env_example.py".
This is the file controlling the games and running basic metrics.

A DQN agent was added to the pool of available agents. Currently a DQN agent is playing together with the original "SimpleAgent", which is a purely rule-based agent.
The DQN agent utilizes a RNN as its Q-Network.
It also has a module called "intention classifier". The aim of this unit is to learn the intentions of its teammate, thus playing cards when it can.
This is a seperate NN, it is trained from the feedback it receives after each game. (During the game the agents only communicate via hints, obeying the rules of the game.)


## Original README

This is not an officially supported Google product.

hanabi\_learning\_environment is a research platform for Hanabi experiments. The file rl\_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py for non-RL methods like Monte Carlo tree search.

### Getting started
Install the learning environment:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
```
Run the examples:
```
pip install numpy                   # game_example.py uses numpy
python examples/rl_env_example.py   # Runs RL episodes
python examples/game_example.py     # Plays a game using the lower level interface
```
