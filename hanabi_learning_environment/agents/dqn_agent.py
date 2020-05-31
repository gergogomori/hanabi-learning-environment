"""DQN Agent."""

import tensorflow as tf
import numpy as np
from tf_agents.trajectories import trajectory, time_step as ts
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import random
from tf_agents.policies import random_tf_policy
from tf_agents.policies import fixed_policy

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

    self.epsilon = 1.0

    self.random_policy = random_tf_policy.RandomTFPolicy(self.time_step_spec, self.action_spec)

    self.intention_classifier = tf.keras.models.Sequential(
      [tf.keras.layers.Dense(1, input_shape=self.observation_spec.shape, activation='sigmoid')])

    self.intention_classifier.compile(optimizer=self.optimizer, loss=tf.keras.losses.BinaryCrossentropy())

    self.intention_dataset_input = []
    self.intention_dataset_true = []

  def set_epsilon(self, counter):
    if counter > 1e6:
      self.epsilon = 1e6 / counter

  def act(self, time_step, episode_counter, prev_knowledge, current_knowledge, card_playable):
    self.set_epsilon(episode_counter)

    hinted = []
    candidate_action = -1
    for i, cards in enumerate(zip(prev_knowledge, current_knowledge)):
      if cards[0] != cards[1]:
        hinted.append(i)

    if len(hinted) == 1:
      candidate_action = 5 + hinted[0]

    self.intention_dataset_input.append(time_step.observation[0])
    self.intention_dataset_true.append(int(card_playable))

    confidence_of_hint = self.intention_classifier(time_step.observation).numpy()[0,0]

    if (confidence_of_hint > 0.6) and (self.epsilon < 0.7) and (candidate_action != -1):
      candidate_action = tf.convert_to_tensor(candidate_action, dtype=tf.int32)
      policy = fixed_policy.FixedPolicy(candidate_action, self.time_step_spec, self.action_spec)
      time_step = ts.restart(tf.ones(self.time_step_spec.observation.shape))
      print("Hinted card played!")
      return policy.action(time_step)

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

    self.intention_classifier.fit(np.asarray(self.intention_dataset_input), np.asarray(self.intention_dataset_true), batch_size=1)
    self.intention_dataset_input = []
    self.intention_dataset_true = []
