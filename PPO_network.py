import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.optim as optim

import torch as T
from torch.distributions import Categorical

#network configuration are predefined preset configuration types which specifies layer structure

linear_dim0 = 256
linear_dim1 = 256
import numpy as np

import os

class ActorNetwork(nn.Module) :
  def __init__(self, input_dims, output_dims, lr=1e-4, network_name = 'test'):
    super(ActorNetwork, self).__init__()

    self.model_fp = os.path.join('models/actor', network_name)

    self.input_dims = input_dims[0]
    self.output_dims = output_dims

    self.conv1 = nn.Conv2d(self.input_dims, 32, 3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.linear = nn.Linear(32 * 6 * 6, 512)
    self.actor = nn.Linear(512, output_dims)
    self.softmax = nn.Softmax(dim=-1)



    # self.actor = nn.Sequential(
    #   nn.Conv2d(self.input_dims, 32, 3, stride=2, padding=1),
    #   nn.Conv2d(32, 32, 3, stride=2, padding=1),
    #   nn.Conv2d(32, 32, 3, stride=2, padding=1),
    #   nn.Conv2d(32, 1152, 3, stride=2, padding=1),
    #   nn.Linear(1152, self.output_dims),
    #   nn.Softmax(dim=-1)
    # )
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.device = T.device('cpu')
    self.to(self.device)

  def forward(self, state):
    x = F.relu(self.conv1(state))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.linear(x.view(x.size(0), -1))
    probs = self.actor(x)
    probs = self.softmax(probs)

    return Categorical(probs)

  def save_network(self):
    print("saving network...")
    T.save(self.state_dict() , self.model_fp)

  def load_network(self):
    if os.path.isfile(self.model_fp):
      print("loading actor network")
      self.load_state_dict(T.load(self.model_fp))

class CriticNetwork(nn.Module) :
  def __init__(self, input_dims, output_dims, lr=1e-4, network_name = 'test'):
    super(CriticNetwork, self).__init__()

    self.model_fp = os.path.join('models/critic', network_name)

    self.input_dims = input_dims[0]
    self.output_dims = output_dims

    self.conv1 = nn.Conv2d(self.input_dims, 32, 3, stride=2, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
    self.linear = nn.Linear(32 * 6 * 6, 512)
    self.critic = nn.Linear(512, 1)


    # self.actor = nn.Sequential(
    #   nn.Conv2d(self.input_dims, 32, 3, stride=2, padding=1),
    #   nn.Conv2d(32, 32, 3, stride=2, padding=1),
    #   nn.Conv2d(32, 32, 3, stride=2, padding=1),
    #   nn.Conv2d(32, 1152, 3, stride=2, padding=1),
    #   nn.Linear(1152, 1)
    # )
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.device = T.device('cpu')
    self.to(self.device)

  def forward(self, state):
    x = F.relu(self.conv1(state))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.linear(x.view(x.size(0), -1))
    value = self.critic(x)
    return value

  def save_network(self):
    T.save(self.state_dict() , self.model_fp)

  def load_network(self):
    if os.path.isfile(self.model_fp) :
      print("loading critic network")
      self.load_state_dict(T.load(self.model_fp))

class PPONetwork(nn.Module):
  def __init__(self, input_dims, output_dims, network_configuration):
    super(PPONetwork, self).__init__()

    self.input_dims = input_dims
    self.output_dims = output_dims
    self.network_configuration = network_configuration

    self.create_network()

  def create_network(self):
    if self.network_configuration == 1 :
      self.conv1 = nn.Conv2d(self.input_dims[0], 32, 3, stride=2, padding=1)
      self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
      self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
      self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
      self.linear = nn.Linear(32 * 6 * 6, 512)
      self.critic_linear = nn.Linear(512, 1)
      self.actor_linear = nn.Linear(512, self.output_dims)

  def init_weights(self):
      for module in self.modules():
          if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
              nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
              nn.init.constant_(module.bias, 0)

  def forward(self, state):
    if self.network_configuration == 1 :
      state = nn.functional.relu(self.conv1(state))
      state = nn.functional.relu(self.conv2(state))
      state = nn.functional.relu(self.conv3(state))
      state= nn.functional.relu(self.conv4(state))
      state = self.linear(state.view(state.size(0), -1))
      return self.actor_linear(state), self.critic_linear(state)