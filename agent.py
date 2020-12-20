from collections import deque
import numpy as np
import torch.functional as F
from torch.distributions import Categorical
import torch
import os

#this class stores rewards and retrieves the instant average of reward
class BaselineBuffer:
  def __init__(self, size=50000):
    self.size = size
    self.storage = deque(maxlen=self.size)
    self.N = 0

  #very unprobable that this actually works
  def average(self):
    return np.mean(self.storage)

  def append(self, v):
    self.storage.append(v)
    self.N = self.N + 1 if self.N < self.size else self.N

class Agent :
  def __init__(self,ppo_net, input_dims, output_dims):
    self.ppo_net = ppo_net
    self.input_dims = input_dims
    self.output_dims = output_dims

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
    self.observations.append(torch.from_numpy(observation))
    self.actions.append(action)
    self.rewards.append(torch.as_tensor(reward))
    self.terminals.append(torch.as_tensor(done))

  def clear_memory(self):
    self.observations = []
    self.actions = []
    self.rewards = []
    self.terminals = []
    self.old_actions = []
    self.values = []

  def save_model(self, path):
    torch.save(self.ppo_net.state_dict(), path)
    print("saving model...")

  def load_model(self, path):
    if os.path.isfile(path):
      print("loading model...")
      self.ppo_net.load_state_dict(torch.load(path))

  def learn(self, last_state, logger):
    _, next_value, = self.ppo_net(torch.from_numpy(last_state))
    next_value = next_value.squeeze()

    old_log_policies = torch.cat(self.old_actions).detach()
    actions = torch.cat(self.actions)
    values = torch.cat(self.values).detach()
    states = torch.cat(self.observations)
    self.local_steps = len(states)
    gae = 0
    R = []
    for value, reward, done in list(zip(self.values, self.rewards, self.terminals))[::-1]:
      gae = gae * self.gamma * self.tau
      gae = gae + reward + self.gamma * next_value.detach() * (1 - done.item()) - value.detach()
      next_value = value
      R.append(gae + value)
    R = R[::-1]
    R = torch.cat(R).detach()
    advantages = R - values
    for i in range(self.learning_epochs):
      indice = torch.randperm(self.local_steps)
      for j in range(self.batch_size):

        batch_idx_start = int(j * (self.local_steps / self.batch_size))
        batch_idx_end = int((j + 1) * (
                            self.local_steps / self.batch_size))
        batch_indices = indice[batch_idx_start : batch_idx_end]

        logits, value = self.ppo_net(states[batch_indices])
        new_policy = torch.nn.functional.softmax(logits, dim=1)
        new_m = Categorical(new_policy)
        new_log_policy = new_m.log_prob(actions[batch_indices])
        ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
        actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                           torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) *
                                           advantages[
                                             batch_indices]))

        critic_loss = torch.nn.functional.smooth_l1_loss(torch.as_tensor(R[batch_indices]).squeeze(), value.squeeze())

        logger.writer.add_scalar("critic_loss", critic_loss, i)

        entropy_loss = torch.mean(new_m.entropy())
        total_loss = actor_loss + critic_loss - self.beta * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ppo_net.parameters(), 0.5)
        self.optimizer.step()









