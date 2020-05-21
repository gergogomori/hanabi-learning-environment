# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment."""

from __future__ import print_function

import sys
import math
import matplotlib.pyplot as plt

from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent

from tf_agents.environments import batched_py_environment
from hanabi_learning_environment.agents.dqn_agent import DQNAgent

# Currently available agents
AGENT_CLASSES = {'DQNAgent' : DQNAgent, 'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent}


class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.num_episodes = flags['num_episodes']
    self.environment = batched_py_environment.BatchedPyEnvironment([rl_env.make('Hanabi-Full', num_players=flags['players'])])
    self.agent_config = {'max_information_tokens' : self.environment.envs[0].game.max_information_tokens(),
                         'action_spec' : self.environment.action_spec(),
                         'observation_spec' : self.environment.observation_spec()}
    self.agent_1 = AGENT_CLASSES[flags['agent_1']](self.agent_config)
    self.agent_2 = AGENT_CLASSES[flags['agent_2']](self.agent_config)

  def run(self):
    """Run episodes."""
    rewards = []
    shouldAgentLearn = False
    for episode in range(self.num_episodes):
      time_step = self.environment.reset()
      agents = [self.agent_1, self.agent_2]
      done = False
      episode_reward = 0

      while not done:
        for agent_id, agent in enumerate(agents):
          if shouldAgentLearn:
            agent.learn()
            shouldAgentLearn = False

          # Make observation and select an action based on the type of the agent
          if agent_id == 0:
            last_time_step = time_step
            action_step = agent.act(time_step)
            action = action_step.action
          else:
            observation = self.environment.envs[0]._make_observation_all_players()['player_observations'][agent_id]
            action = agent.act(observation)

          # Make an environment step
          print('Agent: {} action: {}'.format(agent_id, action))
          time_step = self.environment.step(action)

          if agent_id == 0:
            next_time_step = time_step
            agent.collect(last_time_step, action_step, next_time_step)

          # Increase the reward of the episode
          episode_reward += time_step.reward[0].tolist()

          # Check for end of the episode
          done = time_step.is_last()
          if done:
            shouldAgentLearn = True
            break

      rewards.append(episode_reward)
      print('Episode %d ended.' % (episode+1))
      print('Max reward in the current run: %.3f' % max(rewards))
    return rewards

if __name__ == "__main__":
  flags = {'players': 2, 'num_episodes': 100, 'num_eval_episodes': 10, 'agent_1': 'DQNAgent', 'agent_2': 'SimpleAgent'}
  # Only 2 player games where the first agent is the learning agent are supported.
  if flags['players'] != 2 or flags['agent_1'] != 'DQNAgent':
    sys.exit("Currently this setup is not supported.")
  runner = Runner(flags)
  result = runner.run()
  print('Reward(s) of the episode(s): {}'.format(result))

  # Metrics - Average return
  average_return = []
  num_eval_episodes = flags['num_eval_episodes']
  for i in range(0, len(result), num_eval_episodes):
    average_return.append(math.fsum(result[i : i + num_eval_episodes]) / num_eval_episodes)
  plt.plot(average_return)
  plt.title("Average return")
  plt.show()
