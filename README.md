## Introduction to the project

The original Hanabi repository from DeepMind was modified in order to analyze how a learning agent and rule-based agents play togeher.
Currently the focus is only on two players: one learning agent and one rule-based agent.
The first agent is always the learning agent.
Learning agents can be modifications of agents implemented in [tf-agents](https://www.tensorflow.org/agents/overview).

The file for controlling the games and running basic metrics is: "examples/rl_env_example.py".

Currently a DQN agent is playing together with the original "SimpleAgent", which is a purely rule-based agent.
The DQN agent utilizes a RNN as its Q-Network.
It also has a module called "intention recognizer". The aim of this unit is to learn the intentions of its teammate, thus playing cards when it can.
This is a separate neural network and is trained with the feedback it receives after each game. (During the game the agents only communicate via hints, obeying the rules of the game.)

Below is a comparison of two simulations of 6000 games. The horizontal axis indicates the percentage of total games played. In the first scenario, the additional module was utilized, thus the agent noticed the moments of the game when it can play a card. In the second scenario, the "intention recognizer" was bypassed during the action selection step. Every other aspect of the two scenarios remained the same. An increase of the overall score can be observed when making a comparison.

 With "intention recognizer"   | Without "intention recognizer"
:-----------------------------:|:-----------------------------:
![With "intention recognizer"](https://github.com/gergogomori/hanabi-learning-environment/blob/master/with_intention_recognizer)      |![Without "intention recognizer"](https://github.com/gergogomori/hanabi-learning-environment/blob/master/without_intention_recognizer.png)


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
