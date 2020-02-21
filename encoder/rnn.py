import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *

from base import framework


class Config(framework.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.cell = 'gru'
    self.dim_embed = 256
    self.dim_hidden = 256
    self.num_word = 10000
    self.dropin = .5


class Encoder(nn.Module):
  def __init__(self, config):
    super(Encoder, self).__init__()

    self._config = config

    self.embed = nn.Embedding(self._config.num_word, self._config.dim_embed)
    scale = 1.0 / (self._config.num_word**0.5)
    nn.init.uniform_(self.embed.weight, -scale, scale)

    if self._config.cell == 'gru':
      self.rnn = nn.GRU(self._config.dim_embed, self._config.dim_hidden)
    elif self._config.cell == 'lstm':
      self.rnn = nn.LSTM(self._config.dim_embed, self._config.dim_hidden)

    self.dropin = nn.Dropout(p=self._config.dropin)

  def forward(self, x, lens):
    """
    x: (t, b)
    lens: [int]
    """
    t, b = x.size()

    input = self.embed(x) # (t, b, dim_embed)
    input = self.dropin(input)

    output, _ = self.rnn(input)
    last_hiddens = []
    for i in range(b):
      j = lens[i]-1
      last_hidden = output[j, i]
      last_hiddens.append(last_hidden)
    last_hiddens = torch.stack(last_hiddens, 0) # (b, dim_hidden)
    return last_hiddens
