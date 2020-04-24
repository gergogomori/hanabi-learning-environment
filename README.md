## Introduction

The original Hanabi repository from Deepmind was modified in order to test how rule-based agents and learning agents play Hanabi togeher.
Currently the focus is on only 2 players: one rule-based agent and one learning agent.
Learning agents are planned to be modifications of agents implemented in [tf-agents](https://www.tensorflow.org/agents/overview).
tf-agents has to be installed in order to "run examples/rl_env_example.py".


## Original Readme

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
