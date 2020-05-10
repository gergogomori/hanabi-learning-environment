"""DQN Agent."""

import tensorflow as tf
from tf_agents.trajectories import trajectory, time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import random
from tf_agents.policies import random_tf_policy

class DQNAgent:

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.observation_spec = config['observation_spec']
    self.time_step_spec = ts.time_step_spec(self.observation_spec)
    self.action_spec = config['action_spec']

    self.q_net = q_network.QNetwork(self.observation_spec, self.action_spec)
    self.optimizer = tf.keras.optimizers.Adam()

    self.agent = dqn_agent.DqnAgent(
      self.time_step_spec,
      self.action_spec,
      q_network=self.q_net,
      optimizer=self.optimizer)

    self.agent.initialize()

    self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self.agent.collect_data_spec,
      batch_size=1,
      dataset_drop_remainder=True)

    self.dataset = self.replay_buffer.as_dataset(
      sample_batch_size=1,
      num_steps=2,
      single_deterministic_pass=True)

    self.epsilon = 0.3

    self.random_policy = random_tf_policy.RandomTFPolicy(self.time_step_spec, self.action_spec)

  def act(self, time_step):
    if random.random() > self.epsilon:
      action_step = self.agent.policy.action(time_step)
    else:
      time_step = ts.restart(tf.ones(self.time_step_spec.observation.shape))
      action_step = self.random_policy.action(time_step)
    return action_step

  def collect(self, last_time_step, action_step, next_time_step):
    traj = trajectory.from_transition(last_time_step, action_step, next_time_step)
    self.replay_buffer.add_batch(traj)

  def learn(self):
    if self.replay_buffer.num_frames() > 1:
      for batched_experience in self.dataset:
        experience, _ = batched_experience
        self.agent.train(experience)

    self.replay_buffer.clear()
