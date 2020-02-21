import sys
import math
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *
import numpy as np

from base import framework
import decoder.rnn

class Config(framework.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.cell = 'lstm'
    self.dim_embed = 256
    self.dim_hidden = 256
    self.num_word = 10000
    self.dropin = .5
    self.dropout = .5
    self.max_step = 20
    self.beam_width = 5
    self.num_sample = 1

    self.tied = False
    self.init_fg = False


class Decoder(decoder.rnn.Decoder):
  def forward(self, mode, init_state, **kwargs):
    if mode == 'sample':
      sample_out_wids, log_probs = self.sample_decode(init_state)
      greedy_out_wids = self.greedy_decode(init_state)
      return sample_out_wids, log_probs, greedy_out_wids
    elif mode == 'val':
      return self.greedy_decode(init_state)
    elif mode == 'tst':
      if kwargs['strategy'] == 'beam':
        return self.beam_decode(init_state)
      elif kwargs['strategy'] == 'greedy':
        return self.greedy_decode(init_state)
