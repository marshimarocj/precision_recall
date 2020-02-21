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
    
    self.num_sentence = 5
    self.dim_kernel = 5
    self.num_kernel = 50
    self.noise = .5
    self.dim_ft = 2048
    self.tied_sentence_encoder = True


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
  cfg.num_sentence = kwargs['num_sentence']
  cfg.tied_sentence_encoder = kwargs['tied_sentence_encoder']

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

    if self._config.tied_sentence_encoder:
      self.sent_encoder = encoder.rnn.Encoder(sent_enc_cfg)
    else:
      self.sent2img_encoder = encoder.rnn.Encoder(sent_enc_cfg)
      self.sent2sent_encoder = encoder.rnn.Encoder(sent_enc_cfg)

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
    sents: (b*num_sentence, max_step)
    lens: (b*num_sentence)
    y: (b*num_sentence,)
    """
    if mode == 'trn':
      b = fts.size(0)
      num_sentence = self._config.num_sentence

      logits, logits_img, logits_sent, dist2img, dist2sent = self.predict(fts, sents, lens)
      y = kwargs['y']
      loss = F.cross_entropy(logits.view(b*num_sentence, 2), y.view(b*num_sentence))
      loss_img = F.cross_entropy(logits_img.view(b*num_sentence, 2), y.view(b*num_sentence))
      loss_sent = F.cross_entropy(logits_sent.view(b*num_sentence, 2), y.view(b*num_sentence))
      self.op2monitor['loss_img'] = loss_img
      self.op2monitor['loss_sent'] = loss_sent
      return loss, dist2img, dist2sent
    elif mode == 'val':
      b = fts.size(0)
      num_sentence = self._config.num_sentence

      logits, logits_img, logits_sent, dist2img, dist2sent = self.predict(fts, sents, lens)
      predicts = F.softmax(logits, dim=-1).view(b*num_sentence, 2)
      img_predicts = F.softmax(logits_img, dim=-1).view(b*num_sentence, 2)
      sent_predicts = F.softmax(logits_sent, dim=-1).view(b*num_sentence, 2)
      return predicts, img_predicts, sent_predicts
    else:
      logits, logits_img, logits_sent, dist2img, dist2sent = self.predict(fts, sents, lens)
      predicts = F.softmax(logits, dim=-1).view(b*num_sentence, 2) # (b*num_sentence, 2)
      return predicts

  def predict(self, fts, sents, lens):
    b = fts.size(0)
    num_sentence = self._config.num_sentence
    dim_kernel = self._config.dim_kernel
    num_kernel = self._config.num_kernel

    img_embed = self.img_embed(fts)
    img_embed = F.dropout(img_embed, p=self._config.noise).unsqueeze(1) # (b, 1, num_kernel*dim_kernel)

    if self._config.tied_sentence_encoder:
      hidden = self.sent_encoder(sents, lens)
    else:
      sent2img_hidden = self.sent2img_encoder(sents, lens) # (b*num_sentence, dim_hidden)
      sent2sent_hidden = self.sent2sent_encoder(sents, lens)

    # (b, num_sentence, num_kernel*dim_kernel)
    if self._config.tied_sentence_encoder:
      sent2img_embed = self.sent_img_embed(hidden)
    else:
      sent2img_embed = self.sent_img_embed(sent2img_hidden)
    sent2img_embed = F.dropout(sent2img_embed, p=self._config.noise).view(b, num_sentence, -1)
    if self._config.tied_sentence_encoder:
      sent2sent_embed = self.sent_sent_embed(hidden)
    else:
      sent2sent_embed = self.sent_sent_embed(sent2sent_hidden)
    sent2sent_embed = F.dropout(sent2sent_embed, p=self._config.noise).view(b, num_sentence, -1)

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

    out_img = self.ff_img(dist2img)
    out_sent = self.ff_sent(dist2sent)
    out = out_img + out_sent

    return out, out_img, out_sent, dist2img, dist2sent

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

    pos_captionids = data['pos_captionids']
    pos_captionids = [
      torch.LongTensor(d) for d in pos_captionids
    ]
    pos_captionids = pad_sequence(pos_captionids).cuda()
    pos_caption_lens = torch.LongTensor(data['pos_caption_lens']).cuda()

    b = pos_captionids.size(1)
    y = torch.ones(b, dtype=torch.long).cuda()
    pos_loss, _,  __ = self.model('trn', fts, pos_captionids, pos_caption_lens, y=y)

    neg_captionids = data['neg_captionids']
    neg_captionids = [
      torch.LongTensor(d) for d in neg_captionids
    ]
    neg_captionids = pad_sequence(neg_captionids).cuda()
    neg_caption_lens = torch.LongTensor(data['neg_caption_lens']).cuda()

    b = neg_captionids.size(1)
    y = torch.zeros(b, dtype=torch.long).cuda()
    neg_loss, _, __ = self.model('trn', fts, neg_captionids, neg_caption_lens, y=y)

    loss = pos_loss + neg_loss
    # loss = pos_img_loss + neg_img_loss
    loss = loss.mean()
    loss.backward()

    return loss

  def validation(self):
    predicts = []
    img_predicts = []
    sent_predicts = []
    gts = []
    img_gts = []
    sent_gts = []
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda()
      
      pos_captionids = data['pos_captionids']
      pos_captionids = [
        torch.LongTensor(d) for d in pos_captionids
      ]
      pos_captionids = pad_sequence(pos_captionids).cuda()
      pos_caption_lens = torch.LongTensor(data['pos_caption_lens']).cuda()

      # print pos_captionids.size()
      b = pos_captionids.size(1)
      y = torch.ones(b, dtype=torch.long).cuda()
      pos_predict, pos_img_predict, pos_sent_predict = self.model('val', fts, pos_captionids, pos_caption_lens, y=y)
      pos_predict = pos_predict[:, 1].data.cpu().tolist()
      predicts += pos_predict
      gts += [1 for _ in pos_predict]
      pos_img_predict = pos_img_predict[:, 1].data.cpu().tolist()
      img_predicts += pos_img_predict
      pos_sent_predict = pos_sent_predict[:, 1].data.cpu().tolist()
      sent_predicts += pos_sent_predict

      neg_captionids = data['neg_captionids']
      neg_captionids = [
        torch.LongTensor(d) for d in neg_captionids
      ]
      neg_captionids = pad_sequence(neg_captionids).cuda()
      neg_caption_lens = torch.LongTensor(data['neg_caption_lens']).cuda()

      b = neg_captionids.size(1)
      y = torch.zeros(b, dtype=torch.long).cuda()
      neg_predict, neg_img_predict, neg_sent_predict = self.model('val', fts, neg_captionids, neg_caption_lens, y=y)
      neg_predict = neg_predict[:, 1].data.cpu().tolist()
      predicts += neg_predict
      gts += [0 for _ in neg_predict]
      neg_img_predict = neg_img_predict[:, 1].data.cpu().tolist()
      img_predicts += neg_img_predict
      neg_sent_predict = neg_sent_predict[:, 1].data.cpu().tolist()
      sent_predicts += neg_sent_predict

    ap = sklearn.metrics.average_precision_score(gts, predicts)
    ap_img = sklearn.metrics.average_precision_score(gts, img_predicts)
    ap_sent = sklearn.metrics.average_precision_score(gts, sent_predicts)
    return {
      'ap': float(ap),
      'ap_img': float(ap_img),
      'ap_sent': float(ap_sent),
    }
 

def pad_sequence(seq):
  b = len(seq)
  max_len = max([len(d) for d in seq])
  y = torch.zeros(max_len, b, dtype=torch.long)
  for i, s in enumerate(seq):
    for j, d in enumerate(s):
      y[j, i] = d
  return y

