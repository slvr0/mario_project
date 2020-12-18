import gym_super_mario_bros
import env_preprocessor

#environment wrapping of super mario bros, we inject a logger to extract data at runtime
#we also feed an agent that helps decide actions and collects training/testing data

class Environment :
  def __init__(self, agent, logger, env_name, preprocessing_method):
    self.agent = agent
    self.logger = logger
    self.env_name = env_name
    self.preprocessor = env_preprocessor.EnvPreprocessor(preprocessing_method)
    self.env = self.create_environment()

  def create_environment(self):
    return gym_super_mario_bros.make(self.env_name)












