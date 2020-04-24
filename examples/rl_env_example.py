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
import getopt
from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent

from hanabi_learning_environment.agents.random_tf_agent import RandomTFAgent

# Currently available agents
AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'RandomTFAgent' : RandomTFAgent}


class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.num_episodes = flags['num_episodes']
    self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
    self.agent_config = {'max_information_tokens' : self.environment.game.max_information_tokens(),
                         'action_spec' : self.environment.action_spec(),
                         'observation_spec' : self.environment.observation_spec()}
    self.agent_1 = AGENT_CLASSES[flags['agent_1']](self.agent_config)
    self.agent_2 = AGENT_CLASSES[flags['agent_2']](self.agent_config)

  def run(self):
    """Run episodes."""
    rewards = []
    for episode in range(self.num_episodes):
      time_step = self.environment.reset()
      agents = [self.agent_1, self.agent_2]
      done = False
      episode_reward = 0

      while not done:
        for agent_id, agent in enumerate(agents):

          # Make observations based on the type of agents
          if isinstance(agent, RandomTFAgent):
            observation = time_step.observation
          else:
            observation = self.environment._make_observation_all_players()['player_observations'][agent_id]

          # Action selection
          action = agent.act(observation)

          # Make an environment step
          print('Agent: {} action: {}'.format(agent_id, action))
          time_step = self.environment.step(action)

          # Increase the reward of the episode
          episode_reward += time_step.reward.numpy()

          # Check for end of the episode
          done = time_step.is_last()
          if done:
            break

      rewards.append(episode_reward)
      print('Running episode: %d' % episode)
      print('Max reward in the current run: %.3f' % max(rewards))
    return rewards

if __name__ == "__main__":
  flags = {'players': 2, 'num_episodes': 5, 'agent_1': 'RandomTFAgent', 'agent_2': 'SimpleAgent'}
  if flags['players'] != 2:
    sys.exit("Only 2 player games are supported currently.")
  runner = Runner(flags)
  print('Rewards of the episode(s): {}'.format(runner.run()))
