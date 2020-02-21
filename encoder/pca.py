import sys
sys.path.append('../')

import torch
import torch.nn as nn

from base import framework

class Config(framework.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.dim_ft = 1024
    self.dim_output = 512 # dim of feature layer output


class Encoder(nn.Module):
  def __init__(self, config):
    super(Encoder, self).__init__()

    self._config = config
    self.pca = nn.Linear(self._config.dim_ft, self._config.dim_output)
    nn.init.xavier_uniform_(self.pca.weight)

  def forward(self, ft):
    embed = self.pca(ft)
    return embed
