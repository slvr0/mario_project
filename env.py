import gym_super_mario_bros
from env_preprocessor import EnvPreprocessor
import numpy as np
from agent import Agent
from PPO_network import PPONetwork

import gym
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from gym.spaces import Box

#environment wrapping of super mario bros, we inject a logger to extract data at runtime
#we also use an agent that helps decide actions and collects training/testing data


class CustomRewardWrapper(Wrapper):
  def __init__(self, env=None, env_preprocessor=EnvPreprocessor):
    super(CustomRewardWrapper, self).__init__(env)
    self.preprocessor = env_preprocessor
    self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
    self.current_score = 0

  def step(self, action):
    #action = action.detach().cpu().numpy()
    state, reward, done, info = self.env.step(action)

    state = self.preprocessor.process_img(state)
    reward += (info["score"] - self.curr_score) / 40.

    self.curr_score = info["score"]
    if done:
      if info["flag_get"]:
        reward += 50
      else:
        reward -= 50
    return state, reward / 10., done, info

  def reset(self):
    self.curr_score = 0
    return self.preprocessor.process_img(self.env.reset())

class CustomSkipFrame(Wrapper):
  def __init__(self, env, skip=4):
    super(CustomSkipFrame, self).__init__(env)
    self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
    self.skip = skip
    self.states = np.zeros((skip, 84, 84), dtype=np.float32)

  def step(self, action):
    total_reward = 0
    last_states = []
    for i in range(self.skip):
      state, reward, done, info = self.env.step(action)
      total_reward += reward
      if i >= self.skip / 2:
        last_states.append(state)
      if done:
        self.reset()
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
    max_state = np.max(np.concatenate(last_states, 0), 0)
    self.states[:-1] = self.states[1:]
    self.states[-1] = max_state
    return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

  def reset(self):
    state = self.env.reset()
    self.states = np.concatenate([state for _ in range(self.skip)], 0)
    return self.states[None, :, :, :].astype(np.float32)

class Environment :
  def __init__(self, logger, env_name, config, agent_name):
    self.logger = logger
    self.env_name = env_name
    self.config = config
    self.preprocessor = EnvPreprocessor(self.config.preprocessing_preset)

    self.agent_name = agent_name

    if self.config.testing_environment :
      self.env = gym.make('CartPole-v0')
    else :
      self.env = self.create_environment()
      #convert to joypadspace , then apply custom wrappers
      self.env = JoypadSpace(self.env, actions=SIMPLE_MOVEMENT)
      self.env = CustomRewardWrapper(self.env, self.preprocessor)
      self.env = CustomSkipFrame(self.env, 4) #this is a frame buffer

    self.input_dims = self.env.observation_space.shape
    self.output_dims = self.env.action_space.n

    self.agent = Agent(self.input_dims, self.output_dims, self.config)
    self.agent.load_net()

  def create_environment(self):
    return gym_super_mario_bros.make(self.env_name)

  def run_cartpole_episode(self):

    env = gym.make('CartPole-v0')
    state = env.reset()
    done = False

    self.agent.memory.clear_memory()
    step = 0
    total_reward = 0

    while not done:

      action, prob, value = self.agent.predict(state)
      new_state, reward, done, info = env.step(action)

      total_reward += reward

      self.agent.remember(state, action, reward, value, prob, done)

      step += 1
      if step % self.config.learning_interval == 0  :
        self.agent.train(self.logger)

      #self.env.render()

    return total_reward

  def train(self, n_episodes):
    n_steps = 0
    learn_iters = 0
    best_score = 0
    score_history = []
    warmup_session = 30

    for i in range(n_episodes):
      observation = self.env.reset()
      done = False
      score = 0
      while not done:
        action, prob, val = self.agent.predict(observation)
        observation_, reward, done, info = self.env.step(action)
        n_steps += 1
        score += reward
        self.agent.remember(observation, action, prob, val, reward, done)
        if n_steps % self.config.learning_interval == 0:
          self.agent.train(self.logger)
          learn_iters += 1
        observation = observation_

        #self.env.render()

      score_history.append(score)
      avg_score = np.mean(score_history[-100:])

      if avg_score > best_score :
        best_score = avg_score

        if i > warmup_session : self.agent.save_net()


      print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)












