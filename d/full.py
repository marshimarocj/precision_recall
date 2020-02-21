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
    self.num_sentence = 5


class Discriminator(nn.Module):
  def __init__(self, config):
    super(Discriminator, self).__init__()

    self._config = config
    sent_enc_cfg = self._config.subcfgs[SE]

    self.sent_encoder = encoder.rnn.Encoder(sent_enc_cfg)

    dim_all_kernels = self._config.dim_kernel * self._config.num_kernel
    self.img_embed = nn.Linear(self._config.dim_ft, dim_all_kernels, bias=False)
    self.sent_img_embed = nn.Linear(sent_enc_cfg.dim_hidden, dim_all_kernels, bias=False)
    self.sent_sent_embed = nn.Linear(sent_enc_cfg.dim_hidden, dim_all_kernels, bias=False)

    self.ff_img = nn.Linear(self._config.num_kernel, 2)
    self.ff_sent = nn.Linear(self._config.num_kernel, 2)

    self.op2monitor = {}

  def forward(self, mode, fts, sents, lens, **kwargs):
    """
    fts: (b, dim_ft)
    sents: (b*num_sentence, t)
    lens: (b*num_sentence)
    """
    if mode == 'trn':
      logits, logits_img, logits_sent, dist2img, dist2sent = self.predict(fts, sents, lens)
      y = kwargs['y']
      loss = F.cross_entropy(logits, y)
      loss_img = F.cross_entropy(logits_img, y)
      loss_sent = F.cross_entropy(logits_sent, y)
      return loss, loss_img, loss_sent, dist2img, dist2sent
    elif mode == 'eval':
      logits, logits_img, logits_sent, dist2img, dist2sent = self.predict(fts, sents, lens)
      log_p = F.log_softmax(logits)
      log_p_img = F.log_softmax(logits_img)
      log_p_sent = F.log_softmax(logits_sent)
      return log_p[:, 1], log_p_img[:, 1], log_p_sent[:, 1]
    else:
      logits, logits_img, logits_sent, dist2img, dist2sent = self.predict(fts, sents, lens)
      predicts = F.softmax(logits, dim=-1)
      predicts_img = F.softmax(logits_img, dim=-1)
      predicts_sent = F.softmax(logits_sent, dim=-1)
      return predicts, predicts_img, predicts_sent

  def predict(self, fts, sents, lens):
    b = fts.size(0)
    num_sentence = self._config.num_sentence
    dim_kernel = self._config.dim_kernel
    num_kernel = self._config.num_kernel

    img_embed = self.img_embed(fts)
    img_embed = F.dropout(img_embed, p=self._config.noise).unsqueeze(1) # (b, 1, num_kernel*dim_kernel)

    hidden = self.sent_encoder(sents, lens)

    # (b, num_sentence, num_kernel*dim_kernel)
    sent2img_embed = self.sent_img_embed(hidden)
    sent2img_embed = F.dropout(sent2img_embed, p=self._config.noise).view(b, num_sentence, -1)
    sent2sent_embed = self.sent_sent_embed(hidden)
    sent2sent_embed = F.dropout(sent2sent_embed, p=self._config.noise).view(b, num_sentence, -1)

    # print sent2img_embed.size(), img_embed.size()
    dist2img = (img_embed * sent2img_embed).view(b, num_sentence, num_kernel, dim_kernel)
    dist2img = torch.sum(dist2img, -1) / (dim_kernel**.5)
    dist2img = F.tanh(dist2img) # (b, num_sentence, num_kernel)

    sent2sent = (sent2sent_embed.unsqueeze(2) * sent2sent_embed.unsqueeze(1)).view(b, num_sentence, num_sentence, num_kernel, dim_kernel)
    sent2sent = torch.sum(sent2sent, -1) / (dim_kernel**.5)
    sent2sent = F.tanh(sent2sent) # (b, num_sentence, num_sentence, num_kernel)
    dist2sent = []
    for i in range(num_sentence):
      dist = (torch.sum(sent2sent[:, i], 1) - sent2sent[:, i, i]) / (num_sentence-1) # (b, num_kernel)
      dist2sent.append(dist)
    dist2sent = torch.stack(dist2sent, 1) # (b, num_sentence, num_kernel)

    out_img = self.ff_img(dist2img).view(b*num_sentence, 2)
    out_sent = self.ff_sent(dist2sent).view(b*num_sentence, 2)
    out = (out_img + out_sent).view(b*num_sentence, 2)

    return out, out_img, out_sent, dist2img, dist2sent
