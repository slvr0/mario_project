from collections import deque
import numpy as np
import torch.functional as F
from torch.distributions import Categorical
import torch as T
import os

from PPO_network import ActorNetwork, CriticNetwork

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

#state, action , value, probs(logged probs), (entropy later),

class PPOMemory:
  def __init__(self, batch_size):
    self.states = []
    self.probs = []
    self.vals = []
    self.actions = []
    self.rewards = []
    self.dones = []

    self.batch_size = batch_size

  def generate_batches(self):
    n_states = len(self.states)
    batch_start = np.arange(0, n_states, self.batch_size)
    indices = np.arange(n_states, dtype=np.int64)
    np.random.shuffle(indices)
    batches = [indices[i:i + self.batch_size] for i in batch_start]

    return np.array(self.states), \
           np.array(self.actions), \
           np.array(self.probs), \
           np.array(self.vals), \
           np.array(self.rewards), \
           np.array(self.dones), \
           batches

  def store_memory(self, state, action, probs, vals, reward, done):
    self.states.append(state)
    self.actions.append(action)
    self.probs.append(probs)
    self.vals.append(vals)
    self.rewards.append(reward)
    self.dones.append(done)

  def clear_memory(self):
    self.states = []
    self.probs = []
    self.actions = []
    self.rewards = []
    self.dones = []
    self.vals = []

class Agent :
  def __init__(self,input_dims, output_dims, config):

    self.input_dims = input_dims
    self.output_dims = output_dims
    self.config = config

    self.gamma            = self.config.gamma
    self.gamma_gae        = self.config.gamma_gae
    self.learning_rate    = self.config.learning_rate
    self.batch_size       = self.config.batch_size
    self.policy_grad_clip = self.config.policy_grad_clip
    self.entropy_scaling = self.config.entropy_scaling
    self.memory = PPOMemory(self.config.batch_size)

    self.actor = ActorNetwork(self.input_dims, self.output_dims, self.learning_rate)
    self.critic = CriticNetwork(self.input_dims, self.output_dims, self.learning_rate)

  def save_net(self):
    self.actor.save_network()
    self.critic.save_network()

  def load_net(self):
    self.actor.load_network()
    self.critic.load_network()

  def predict(self, state):

    state = T.tensor([state], dtype=T.float32).to(self.actor.device)

    probs = self.actor(state[0])
    value = self.critic(state[0])

    action = probs.sample()

    prob = T.squeeze(probs.log_prob(action)).item()
    action = T.squeeze(action).item()
    value = T.squeeze(value).item()

    return action, prob, value

  def remember(self, state, action, probs, vals, reward, done):
    self.memory.store_memory(state, action, probs, vals, reward, done)

  def train(self, logger):
    for epoch in range(self.config.epochs):

      states, actions, probs, values, rewards, terminals, batches = self.memory.generate_batches()

      advantage = np.zeros(len(rewards), dtype=np.float32)
      for t in range(len(rewards) - 1):
        adv_decay = 1.0
        adv_t = 0
        for k in range(t, len(rewards) - 1):
          adv_t += adv_decay * (rewards[k] + self.gamma * values[k + 1]
                                * (1 - int(terminals[k])) - values[k])



          adv_decay *= self.gamma * self.gamma_gae
        advantage[t] = adv_t

      advantage = T.tensor([advantage]).to(self.actor.device)

      for batch in batches :

        #now fetch batch data
        state_t = T.tensor(states[batch], dtype=T.float32 ).to(self.actor.device)
        prev_probs_t = T.tensor(probs[batch]).to(self.actor.device)
        actions_t = T.tensor(actions[batch]).to(self.actor.device)

        #calculate new poliy values, this is being looped so at this state policy will change,

        new_probs_t   = self.actor(state_t[:,0,:,:,:])

        entropy_loss = T.mean(new_probs_t.entropy())

        new_probs_t = new_probs_t.log_prob(actions_t)


        new_critics_v_t = self.critic(state_t[:,0,:,:,:])

        probability_ratio = new_probs_t.exp() / prev_probs_t.exp()

        weight_prob_ratio = advantage[0][batch] * probability_ratio

        weight_prob_ratio_clipped = T.clamp(weight_prob_ratio, 1 - self.policy_grad_clip, 1 + self.policy_grad_clip) *\
          advantage[0][batch]

        a_loss = -T.min(weight_prob_ratio, weight_prob_ratio_clipped).mean()

        state_prop = advantage[0][batch] + values[batch]
        c_loss = ((state_prop - new_critics_v_t) ** 2).mean()

        total_loss = a_loss + .5 * c_loss - (entropy_loss * self.entropy_scaling)

        #print(total_loss, (entropy_loss * self.entropy_scaling))

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        total_loss.backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

    self.memory.clear_memory()































