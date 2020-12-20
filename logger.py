import tensorboardX

from conf import Config

class Logger:
  def __init__(self, config):
    self.writer = tensorboardX.SummaryWriter('logs')
    self.config = config

