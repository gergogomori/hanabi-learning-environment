"""DQN Agent."""

import tensorflow as tf
import numpy as np
from tf_agents.trajectories import trajectory, time_step as ts
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
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
    self.environment_batch_size = config['environment_batch_size']

    self.q_net = q_rnn_network.QRnnNetwork(
      self.observation_spec,
      self.action_spec,
      input_fc_layer_params=(256, ),
      lstm_size=(256, 256))

    self.optimizer = tf.keras.optimizers.Adam()

    self.agent = dqn_agent.DqnAgent(
      self.time_step_spec,
      self.action_spec,
      q_network=self.q_net,
      optimizer=self.optimizer,
      epsilon_greedy=0.3)

    self.agent.initialize()

    self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=self.agent.collect_data_spec,
      batch_size=1,
      dataset_drop_remainder=True)

    self.dataset = self.replay_buffer.as_dataset(
      sample_batch_size=1,
      num_steps=2,
      single_deterministic_pass=True)

    self.intention_classifier = tf.keras.models.Sequential(
      [tf.keras.layers.Dense(1, input_shape=self.observation_spec.shape, activation='sigmoid')])

    self.intention_classifier.compile(optimizer=self.optimizer, loss=tf.keras.losses.BinaryCrossentropy())

    self.intention_dataset_input = []
    self.intention_dataset_true = []

    self.num_hinted = 0

  def initial_policy_state(self):
    return self.agent.policy.get_initial_state(self.environment_batch_size)

  def time_step_converter(self, time_step):
    if time_step.is_first():
      tf_time_step = ts.restart(
        tf.convert_to_tensor(time_step.observation, dtype=tf.int32),
        self.environment_batch_size)
    elif time_step.is_last():
      tf_time_step = ts.termination(
        tf.convert_to_tensor(time_step.observation, dtype=tf.int32),
        tf.convert_to_tensor(time_step.reward, dtype=tf.float32))
    else:
      tf_time_step = ts.transition(
        tf.convert_to_tensor(time_step.observation, dtype=tf.int32),
        tf.convert_to_tensor(time_step.reward, dtype=tf.float32))

    return tf_time_step

  def act(self, time_step, policy_state, prev_knowledge, current_knowledge, card_playable):

    time_step = self.time_step_converter(time_step)

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

    if (confidence_of_hint > 0.85) and (candidate_action != -1):
      print("Hinted card is played!")
      print(current_knowledge)
      self.num_hinted += 1
      candidate_action = tf.convert_to_tensor(candidate_action, dtype=tf.int32)
      policy = fixed_policy.FixedPolicy(candidate_action, self.time_step_spec, self.action_spec)
      time_step = ts.restart(tf.ones(self.time_step_spec.observation.shape, dtype=tf.int32))
      action_step = policy.action(time_step)
    else:
      action_step = self.agent.collect_policy.action(time_step, policy_state)

    return action_step

  def collect(self, last_time_step, action_step, next_time_step):
    last_time_step = self.time_step_converter(last_time_step)
    next_time_step = self.time_step_converter(next_time_step)
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

    self.num_hinted = 0
