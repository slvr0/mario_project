from torch import nn
from torch import functional as F
import torch

#network configuration are predefined preset configuration types which specifies layer structure
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