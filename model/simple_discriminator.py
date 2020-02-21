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
    self.img_embed = nn.Linear(self._config.dim_ft, dim_all_kernels, bias=False)
    self.sent_img_embed = nn.Linear(sent_enc_cfg.dim_hidden, dim_all_kernels, bias=False)

    self.ff = nn.Linear(self._config.num_kernel, 2)

    self.op2monitor = {}

  def forward(self, mode, fts, sents, lens, **kwargs):
    """
    fts: (b, dim_ft)
    sents: (b, max_step)
    lens: (b)
    y: (b,)
    """
    if mode == 'trn':
      b = fts.size(0)

      logits, dist2img = self.predict(fts, sents, lens)
      y = kwargs['y']
      loss = F.cross_entropy(logits, y)
      return loss, dist2img
      # predicts = F.softmax(logits, dim=-1)
      # return predicts, loss
    elif mode == 'val':
      b = fts.size(0)
      
      logits, dist2img = self.predict(fts, sents, lens)
      predicts = F.softmax(logits, dim=-1)
      return predicts
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

  def trainable_params(self):
    params = []
    for name, param in self.named_parameters():
      if param.requires_grad:
        print(name)
        params.append(param)
    return params


PathCfg = model.util.PathCfg


class PretrainTrnTst(framework.TrnTst):
  def __init__(self, model_cfg, path_cfg, gpuids):
    super(PretrainTrnTst, self).__init__(model_cfg, path_cfg, gpuids)

  def feed_data_forward_backward(self, data):
    fts = torch.Tensor(data['fts']).cuda()
    b, dim_ft = fts.size()
    fts = fts.unsqueeze(1).expand(b, 5, dim_ft).contiguous().view(-1, dim_ft)

    pos_captionids = data['pos_captionids']
    pos_captionids = [
      torch.LongTensor(d) for d in pos_captionids
    ]
    pos_captionids = pad_sequence(pos_captionids).cuda()
    pos_caption_lens = torch.LongTensor(data['pos_caption_lens']).cuda()

    b = pos_captionids.size(1)
    y = torch.ones(b, dtype=torch.long).cuda()
    pos_loss, _ = self.model('trn', fts, pos_captionids, pos_caption_lens, y=y)

    neg_captionids = data['neg_captionids']
    neg_captionids = [
      torch.LongTensor(d) for d in neg_captionids
    ]
    neg_captionids = pad_sequence(neg_captionids).cuda()
    neg_caption_lens = torch.LongTensor(data['neg_caption_lens']).cuda()

    b = neg_captionids.size(1)
    y = torch.zeros(b, dtype=torch.long).cuda()
    neg_loss, _ = self.model('trn', fts, neg_captionids, neg_caption_lens, y=y)

    loss = pos_loss + neg_loss
    loss.backward()

    return loss

  def validation(self):
    predicts = []
    gts = []
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda()
      b, dim_ft = fts.size()
      fts = fts.unsqueeze(1).expand(b, 5, dim_ft).contiguous().view(-1, dim_ft)
      
      pos_captionids = data['pos_captionids']
      pos_captionids = [
        torch.LongTensor(d) for d in pos_captionids
      ]
      pos_captionids = pad_sequence(pos_captionids).cuda()
      pos_caption_lens = torch.LongTensor(data['pos_caption_lens']).cuda()

      pos_predicts = self.model('val', fts, pos_captionids, pos_caption_lens)
      pos_predicts = pos_predicts[:, 1].data.cpu().tolist()
      predicts += pos_predicts
      gts += [1 for _ in pos_predicts]

      neg_captionids = data['neg_captionids']
      neg_captionids = [
        torch.LongTensor(d) for d in neg_captionids
      ]
      neg_captionids = pad_sequence(neg_captionids).cuda()
      neg_caption_lens = torch.LongTensor(data['neg_caption_lens']).cuda()

      neg_predicts = self.model('val', fts, neg_captionids, neg_caption_lens)
      neg_predicts = neg_predicts[:, 1].data.cpu().tolist()
      predicts += neg_predicts
      gts += [0 for _ in neg_predicts]

    ap = sklearn.metrics.average_precision_score(gts, predicts)
    return {
      'ap': float(ap),
    }

  def predict_in_tst(self):
    # cum_accuracy = 0.
    # cnt = 0
    predicts = []
    gts = []
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda()
      b, dim_ft = fts.size()
      fts = fts.unsqueeze(1).expand(b, 5, dim_ft).contiguous().view(-1, dim_ft)
      
      pos_captionids = data['pos_captionids']
      pos_captionids = [
        torch.LongTensor(d) for d in pos_captionids
      ]
      pos_captionids = pad_sequence(pos_captionids).cuda()
      pos_caption_lens = torch.LongTensor(data['pos_caption_lens']).cuda()

      pos_predicts = self.model('tst', fts, pos_captionids, pos_caption_lens)
      pos_predicts = pos_predicts[:, 1].data.cpu().tolist()
      predicts += pos_predicts
      gts += [1 for _ in pos_predicts]

      neg_captionids = data['neg_captionids']
      neg_captionids = [
        torch.LongTensor(d) for d in neg_captionids
      ]
      neg_captionids = pad_sequence(neg_captionids).cuda()
      neg_caption_lens = torch.LongTensor(data['neg_caption_lens']).cuda()

      neg_predicts = self.model('tst', fts, neg_captionids, neg_caption_lens)
      # pp += torch.sum(neg_predicts[:, 1] < neg_predicts[:, 0]).data.cpu().numpy()
      neg_predicts = neg_predicts[:, 1].data.cpu().tolist()
      predicts += neg_predicts
      gts += [0 for _ in neg_predicts]

      # cum_accuracy += pp
      # cnt += b*2
    # print cum_accuracy / cnt
    ap = sklearn.metrics.average_precision_score(gts, predicts)
    print ap


def pad_sequence(seq):
  b = len(seq)
  max_len = max([len(d) for d in seq])
  y = torch.zeros(max_len, b, dtype=torch.long)
  for i, s in enumerate(seq):
    for j, d in enumerate(s):
      y[j, i] = d
  return y
