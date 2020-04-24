"""Random TF Agent."""

from hanabi_learning_environment.rl_env import Agent

import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import random_tf_policy


class RandomTFAgent(Agent):
  """Agent that takes random actions."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.action_spec = config['action_spec']
    self.observation_spec = config['observation_spec']
    self.time_step_spec = ts.time_step_spec(self.observation_spec)
    self.policy = random_tf_policy.RandomTFPolicy(action_spec=self.action_spec, time_step_spec=self.time_step_spec)

  def act(self, observation):
    """Act based on an observation."""
    time_step = ts.restart(observation)
    return self.policy.action(time_step).action
