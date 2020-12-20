
#arguments should be passed in directly from the argparset to the constructor here
class Config :
  def __init__(self, *args, **kwarg):
    self.network_preset = 1
    self.preprocessing_preset = 1