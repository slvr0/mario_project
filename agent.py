from collections import deque
import numpy as np
import torch.functional as F
from torch.distributions import Categorical
import torch
import os
from conf import Config
from logger import Logger

#this class stores rewards and retrieves the instant average of reward
class BaselineBuffer:
  def __init__(self, size=50000):
    self.size = size
    self.storage = deque(maxlen=self.size)
    self.N = 0
    self.sum = 0

  #very unprobable that this actually works
  def average(self):
    if self.N > 0 :
      return self.sum / self.N
    else:
      return 0.0

    return np.mean(self.storage)

  def append(self, v):
    self.storage.append(v)
    self.sum += v
    self.N = self.N + 1 if self.N < self.size else self.N

class Agent :
  def __init__(self,ppo_net, input_dims, output_dims, config = Config()):
    self.ppo_net = ppo_net
    self.input_dims = input_dims
    self.output_dims = output_dims

    self.config = config

    self.logger = Logger(config=config)
    #hyperparams, move to config and argparser soon
    self.gamma = 0.9
    self.tau = 1.0
    self.beta = 0.01
    self.epsilon = 0.2
    self.batch_size = 1000
    self.save_interval = 10
    self.learning_rate = 1e-4
    self.local_steps = 512 # steps we take in learning algorithm chain
    self.batch_size = 8
    self.learning_epochs = 8 #change to ten
    self.global_steps = 5e6

    self.optimizer = torch.optim.Adam(self.ppo_net.parameters(), lr=self.learning_rate)

    self.baseline_buffer = BaselineBuffer(size=1000)

    self.observations = []
    self.actions = []
    self.rewards = []
    self.terminals = []
    self.values = []
    self.old_actions = []
    self.batch_scales = []

  def predict(self, observation):

    obs_tensor = torch.from_numpy(observation)

    logits, value = self.ppo_net(obs_tensor)
    self.values.append(value)

    policy = torch.nn.functional.softmax(logits, dim=1)
    old_m = Categorical(policy)
    action = old_m.sample()
    self.old_actions.append(action)

    return action

  def store(self, observation, action, reward, done):
    self.observations.append(observation)
    self.actions.append(action)
    self.rewards.append(torch.as_tensor(reward))
    self.terminals.append(torch.as_tensor(done))
    self.batch_scales.append(reward - self.baseline_buffer.average())

  def clear_memory(self):
    self.observations = []
    self.actions = []
    self.rewards = []
    self.terminals = []
    self.old_actions = []
    self.values = []
    self.batch_scales = []

  def save_model(self, path):
    torch.save(self.ppo_net.state_dict(), path)
    print("saving model...")

  def load_model(self, path):
    if os.path.isfile(path):
      print("loading model...")
      self.ppo_net.load_state_dict(torch.load(path))

  def learn(self, last_state, step_idx):
    actions = torch.LongTensor(self.actions).to(self.ppo_net.device)
    states = torch.FloatTensor(self.observations).to(self.ppo_net.device)
    scales = torch.FloatTensor(self.batch_scales).to(self.ppo_net.device)

    self.optimizer.zero_grad()

    logits,_ = self.ppo_net(states[:,0,:,:,:])
    logits_prob_log_softmaxed = torch.nn.functional.log_softmax(logits, dim=1)

    batch_size = self.config.batch_size

    logits_prob_action_scaled = scales * logits_prob_log_softmaxed[:, actions]

    loss = -logits_prob_action_scaled.mean()

    #now include entropy...
    logits_prob_softmaxed = torch.nn.functional.softmax(logits, dim=1)

    #entropy H(p) = - average sum of log p * p
    entropy = -(logits_prob_softmaxed * logits_prob_log_softmaxed).sum(dim=1).mean()
    entropy_loss = -self.config.entropy_beta * entropy

    final_loss = loss + entropy_loss

    final_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.ppo_net.parameters(), 0.5)

    self.optimizer.step()

    #output on tensorboard X (i.e.s do new calculations with weights updated and see the change
    newlogits, _ = self.ppo_net(states[:,0,:,:,:])
    gradient_max, gradient_means, gradient_counts = 0.0, 0.0, 0

    for param in self.ppo_net.parameters():

      gradient_max = max(gradient_max, param.data.abs().max().item())
      gradient_means += (param.data **2).mean().sqrt().item()
      gradient_counts += 1

    self.logger.writer.add_scalar("baseline reward", self.baseline_buffer.average(), step_idx)
    self.logger.writer.add_scalar("entropy(uncertainty in probability distribution)", entropy.item(), step_idx)
    self.logger.writer.add_scalar("scales (expected reward vs each action reward in training session)", scales.mean(), step_idx)
    self.logger.writer.add_scalar("loss entropy", entropy_loss.item(), step_idx)

    self.logger.writer.add_scalar("loss policy", loss.item(), step_idx)
    self.logger.writer.add_scalar("total loss", final_loss.item(), step_idx)
    self.logger.writer.add_scalar("gradient change average ", gradient_means / gradient_counts, step_idx)
    self.logger.writer.add_scalar("max gradient change", gradient_max, step_idx)

    self.clear_memory()












