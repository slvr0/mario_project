
from conf import Config
from agent import Agent
from env import Environment
from logger import Logger

import gym

import numpy as np

def run_cartpole_test():
  env = gym.make('CartPole-v0')
  N = 20
  config = Config()
  logger = Logger(config)

  agent = Agent(input_dims=env.observation_space.shape, output_dims=env.action_space.n, config=config)

  n_games = 300

  best_score = env.reward_range[0]
  score_history = []

  learn_iters = 0
  avg_score = 0
  n_steps = 0

  for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
      action, prob, val = agent.predict(observation)
      observation_, reward, done, info = env.step(action)
      n_steps += 1
      score += reward
      agent.remember(observation, action, prob, val, reward, done)
      if n_steps % N == 0:
        agent.train(logger)
        learn_iters += 1
      observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
      best_score = avg_score
      #agent.save_net()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
          'time_steps', n_steps, 'learning_steps', learn_iters)


