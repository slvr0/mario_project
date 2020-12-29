
from env import Environment
from conf import Config
from logger import Logger

import numpy as np

from PPO_network import ActorNetwork, CriticNetwork
from agent import Agent

from cartpole_test_env import run_cartpole_test

if __name__ == '__main__':
  config = Config()

  run_cartpole_test()


  # config = Config()
  # net = ActorNetwork((64,64),7)
  # ppo_mem = PPOAgentMemory(config)
  #
  # env_name = 'SuperMarioBros-v0'
  #
  # logger = Logger(config)
  #
  # agent_name = 'ppo_net_cartpole'
  # env = Environment(logger=logger, env_name=env_name, config=config, agent_name=agent_name)
  #
  #
  #
  # # config = Config()
  # # env_name = 'SuperMarioBros-v0'
  # # logger = Logger(config)
  # #
  # # agent_name = 'ppo_net_w0_l0_0'
  # # env = Environment(logger=logger, env_name=env_name, config=config, agent_name=agent_name)
  # #
  #
  #
  # nr_episodes = 1
  # average_reward = []
  # save_interval = 10
  #
  # for i in range(1, nr_episodes + 1):
  #  episode_reward = env.run_episode()
  #  average_reward.append(episode_reward)
  #
  #  if (i % save_interval) == 0 :
  #    env.agent.save_model('models/' + agent_name)
  #
  #  print("episode {} done, reward : {}, average reward : {}".format(i, episode_reward, np.mean(average_reward)))





