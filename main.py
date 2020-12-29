
from env import Environment
from conf import Config
from logger import Logger

import numpy as np

from PPO_network import ActorNetwork, CriticNetwork
from agent import Agent

from cartpole_test_env import run_cartpole_test

if __name__ == '__main__':
  config = Config()

  #run_cartpole_test()

  env_name = 'SuperMarioBros-v0'

  logger = Logger(config)

  agent_name = 'ppo_net_cartpole'
  env = Environment(logger=logger, env_name=env_name, config=config, agent_name=agent_name)

  nr_episodes = 1000
  average_reward = []
  save_interval = 10

  env.train(n_episodes=nr_episodes)





