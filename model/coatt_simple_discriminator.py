import sys
import os
import json
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics

from base import framework
import encoder.rnn
import encoder.pca
import model.util

SE = 'sentence_encoder'

class ModelConfig(framework.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[SE] = encoder.rnn.Config()

    self.dim_kernel = 5
    self.num_kernel = 50
    self.noise = .5
    self.dim_ft = 2048
    self.num_ft = 36


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.trn_batch_size = 64
  cfg.tst_batch_size = 64
  cfg.num_epoch = kwargs['num_epoch']
  cfg.val_iter = -1
  cfg.monitor_iter = 100
  cfg.base_lr = kwargs['lr']
  cfg.dim_kernel = kwargs['dim_kernel']
  cfg.num_kernel = kwargs['num_kernel']
  cfg.noise = kwargs['discriminator_noise']
  cfg.dim_ft = kwargs['dim_ft']
  cfg.num_ft = kwargs['num_ft']

  sent_enc_cfg = cfg.subcfgs[SE]
  sent_enc_cfg.cell = kwargs['cell']
  sent_enc_cfg.dim_embed = kwargs['dim_input']
  sent_enc_cfg.dim_hidden = kwargs['dim_hidden']
  sent_enc_cfg.dropin = kwargs['dropin']

  return cfg


class Model(nn.Module):
  def __init__(self, config):
    super(Model, self).__init__()

    self._config = config
    sent_enc_cfg = self._config.subcfgs[SE]

    self.sent_encoder = encoder.rnn.Encoder(sent_enc_cfg)

    dim_all_kernels = self._config.dim_kernel * self._config.num_kernel
    self.img_key = nn.Linear(self._config.dim_ft, dim_all_kernels, bias=False)
    self.sent_img_key = nn.Linear(sent_enc_cfg.dim_hidden, dim_all_kernels, bias=False)
    self.