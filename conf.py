
#arguments should be passed in directly from the argparset to the constructor here
class Config :
  def __init__(self, *args, **kwarg):
    self.network_preset = 1
    self.preprocessing_preset = 1
    self.gamma = 0.99
    self.learning_rate = 1e-4
    self.batch_size = 32
    self.entropy_beta = 0.01
    self.bellman_steps = 10
    