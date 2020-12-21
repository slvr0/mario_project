import gym_super_mario_bros
from env_preprocessor import EnvPreprocessor
import numpy as np
from agent import Agent
from PPO_network import PPONetwork

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
    action = action.detach().cpu().numpy()
    state, reward, done, info = self.env.step(*action)

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
    self.env = self.create_environment()
    self.agent_name = agent_name

    #convert to joypadspace , then apply custom wrappers
    self.env = JoypadSpace(self.env, actions=SIMPLE_MOVEMENT)
    self.env = CustomRewardWrapper(self.env, self.preprocessor)
    self.env = CustomSkipFrame(self.env, 4) #this is a frame buffer

    self.input_dims = self.env.observation_space.shape
    self.output_dims = self.env.action_space.n

    self.ppo_net = PPONetwork(self.input_dims, self.output_dims, self.config.network_preset)
    self.agent = Agent(self.ppo_net, self.input_dims, self.output_dims, self.config)
    self.agent.load_model('models/' + self.agent_name)
    self.ppo_net.eval()

    self.step_idx = 0

  def create_environment(self):
    return gym_super_mario_bros.make(self.env_name)

  def run_episode(self):
    state = self.env.reset()
    done = False
    self.agent.clear_memory()

    total_reward = 0

    while not done:
      self.step_idx += 1

      action = self.agent.predict(state)
      new_state, reward, done, info = self.env.step(action)

      total_reward += reward

      self.agent.baseline_buffer.append(reward)

      self.agent.store(state, action, reward, done)

      #self.env.render()

    self.agent.learn(state, self.step_idx)
    return total_reward













