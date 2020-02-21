import sys
import math
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import framework
import encoder.rnn

SE = 'sentence_encoder'

class Config(framework.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.subcfgs[SE] = encoder.rnn.Config()

    self.dim_kernel = 5
    self.num_kernel = 50
    self.noise = .5
    self.dim_ft = 2048


class Discriminator(nn.Module):
  def __init__(self, config):
    super(Discriminator, self).__init__()

    self._config = config
    sent_enc_cfg = self._config.subcfgs[SE]

    self.sent_encoder = encoder.rnn.Encoder(sent_enc_cfg)

    dim_all_kernels = self._config.dim_kernel * self._config.num_kernel
    self.img_embed = nn.Linear(self._config.dim_ft, dim_all_kernels, bias=False)
    self.sent_img_embed = nn.Linear(sent_enc_cfg.dim_hidden, dim_all_kernels, bias=False)

    self.ff = nn.Linear(self._config.num_kernel, 2)

    self.op2monitor = {}

  def forward(self, mode, fts, sents, lens, **kwargs):
    """
    fts: (b, dim_ft)
    sents: (max_step, b)
    lens: (b)
    y: (b,)
    """
    if mode == 'trn':
      logits, dist2img = self.predict(fts, sents, lens)
      y = kwargs['y']
      loss = F.cross_entropy(logits, y)
      return loss, dist2img
    elif mode == 'eval':
      logits, dist2img = self.predict(fts, sents, lens)
      log_p = F.log_softmax(logits)
      return log_p[:, 1]
    else:
      logits, dist2img = self.predict(fts, sents, lens)
      predicts = F.softmax(logits, dim=-1)
      return predicts

  def predict(self, fts, sents, lens):
    b = fts.size(0)
    dim_kernel = self._config.dim_kernel
    num_kernel = self._config.num_kernel

    img_embed = self.img_embed(fts)
    img_embed = F.dropout(img_embed, p=self._config.noise) # (b,  num_kernel*dim_kernel)

    hidden = self.sent_encoder(sents, lens) # (b, dim_hidden)

    # (b, num_kernel*dim_kernel)
    sent2img_embed = self.sent_img_embed(hidden)
    sent2img_embed = F.dropout(sent2img_embed, p=self._config.noise)

    dist2img = (img_embed * sent2img_embed).view(b, num_kernel, dim_kernel)
    dist2img = torch.sum(dist2img, -1) / (dim_kernel**.5)
    dist2img = F.tanh(dist2img)

    out = self.ff(dist2img) # (b, 2)

    return out, dist2img
