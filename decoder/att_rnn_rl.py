import sys
import math
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base import framework
import decoder.att_rnn

CELL = 'cell'
ATT = 'attention'

class Config(framework.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.num_word = 10000
    self.dim_embed = 256
    self.dim_att_ft = 1024
    self.dropin = .5
    self.dropout = .5
    self.max_step = 15
    self.beam_width = 5
    self.tied_key_val = True
    self.val_proj = True

    self.num_sample = 1

    self.subcfgs[CELL] = att_rnn.CellConfig()

  def _assert(self):
    assert self.dim_embed == self.subcfgs[CELL].dim_embed


class Decoder(decoder.att_rnn.Decoder):
  def forward(self, mode, init_state, att_fts, att_masks, **kwargs):
    if mode == 'sample':
      sample_out_wids, log_probs = self.sample_decode(init_state, att_fts, att_masks)
      greedy_out_wids = self.greedy_decode(init_state, att_fts, att_masks)
      return sample_out_wids, log_probs, greedy_out_wids
    elif mode == 'val':
      return self.greedy_decode(init_state)
    elif mode == 'tst':
      if kwargs['strategy'] == 'beam':
        return self.beam_decode(init_state, att_fts, att_masks)
      elif kwargs['strategy'] == 'greedy':
        return self.greedy_decode(init_state, att_fts, att_masks)
