
#arguments should be passed in directly from the argparset to the constructor here
class Config :
  def __init__(self, *args, **kwarg):
    self.network_preset = 1
    self.preprocessing_preset = 1

    self.testing_environment = True # if true, sets up CartPole as testing environment no preprocessing/env wrapping for mario
    #going on

    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    self.batch_size = batch_size
    self.learning_rate = alpha
    self.gamma = 0.99
    self.gamma_gae = 0.95
    self.policy_grad_clip = 0.1

    self.epochs = n_epochs

    self.learning_interval = 20

