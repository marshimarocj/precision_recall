import sys
import os
import json
sys.path.append('../')

import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import numpy as np
from bleu import bleu
from cider import cider

from base import framework
import encoder.pca
import decoder.rnn
import model.data
import model.util

ENC = 'encoder'
DEC = 'decoder'

class ModelConfig(framework.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[ENC] = encoder.pca.Config()
    self.subcfgs[DEC] = decoder.rnn.Config()

    self.cell = 'gru'
    self.strategy = 'beam'
    self.metric = 'cider'
    self.min_lr = 1e-4

    self.pool_size = 5
    self.sample_topk = 5
    self.threshold_p = .8


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.trn_batch_size = 64
  cfg.tst_batch_size = 64
  cfg.num_epoch = kwargs['num_epoch']
  cfg.val_iter = -1
  cfg.monitor_iter = 100
  cfg.base_lr = kwargs['lr']
  cfg.min_lr = kwargs['min_lr']

  cfg.cell = kwargs['cell']

  enc_cfg = cfg.subcfgs[ENC]
  enc_cfg.dim_ft = kwargs['dim_ft']
  enc_cfg.dim_output = kwargs['dim_hidden']

  dec_cfg = cfg.subcfgs[DEC]
  dec_cfg.cell = cfg.cell
  dec_cfg.dim_embed = kwargs['dim_embed']
  dec_cfg.dim_hidden = kwargs['dim_hidden']
  dec_cfg.dropin = kwargs['dropin']
  dec_cfg.dropout = kwargs['dropout']
  dec_cfg.max_step = kwargs['max_step']
  dec_cfg.tied = kwargs['tied']
  dec_cfg.beam_width = kwargs['beam_width']
  dec_cfg.init_fg = kwargs['init_fg']

  return cfg


class Model(nn.Module):
  def __init__(self, config):
    super(Model, self).__init__()

    self._config = config
    enc_cfg = self._config.subcfgs[ENC]
    dec_cfg = self._config.subcfgs[DEC]

    self.encoder = encoder.pca.Encoder(enc_cfg)
    self.decoder = decoder.rnn.Decoder(dec_cfg)

    self.op2monitor = {}

  def forward(self, mode, ft, **kwargs):
    embed = self.encoder(ft)
    if self._config.subcfgs[DEC].cell == 'lstm':
      init_state = (embed, embed)
    elif self._config.subcfgs[DEC].cell == 'gru':
      init_state = embed

    if mode == 'trn':
      log_prob, out_mask = self.decoder(mode, init_state, 
        y=kwargs['captionids'], lens=kwargs['caption_lens'])
      loss = -log_prob * out_mask
      loss = torch.sum(loss) / torch.sum(out_mask)

      return loss
    elif mode == 'val':
      return self.decoder(mode, init_state, 
        y=kwargs['captionids'], lens=kwargs['caption_lens'])
    elif mode == 'tst':
      if kwargs['strategy'] in 'beam|sample':
        return self.decoder(mode, init_state, strategy=kwargs['strategy'])
      elif kwargs['strategy'] == 'sample_topk':
        return self.decoder(mode, init_state, strategy=kwargs['strategy'], topk=self._config.sample_topk)
      elif kwargs['strategy'] == 'nucleus_sample':
        return self.decoder(mode, init_state, strategy=kwargs['strategy'], threshold_p=self._config.threshold_p)

  def trainable_params(self):
    params = []
    for name, param in self.named_parameters():
      if param.requires_grad:
        print(name)
        params.append(param)
    return params


PathCfg = model.util.PathCfg


class TrnTst(framework.TrnTst):
  def __init__(self, model_cfg, path_cfg, gpuids):
    super(TrnTst, self).__init__(model_cfg, path_cfg, gpuids)

    self.int2str = model.util.CaptionInt2str(path_cfg.word_file)

  def feed_data_forward_backward(self, data):
    fts = torch.Tensor(data['fts']).cuda()

    captionids = data['captionids']
    captionids = [
      torch.LongTensor(captionid).cuda()
      for captionid in captionids
    ]
    captionids = pad_sequence(captionids)
    captionids = torch.transpose(captionids, 0, 1) # (b, t, captionids)

    caption_lens = torch.LongTensor(data['caption_lens']).cuda()

    loss = self.model('trn', fts, captionids=captionids, caption_lens=caption_lens)
    loss = loss.mean()
    loss.backward()

    return loss

  def validation(self):
    vid2predicts = {}
    base = 0
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda()
      # print fts.size()

      captionids = data['captionids']
      captionids = [
        torch.LongTensor(captionid).cuda()
        for captionid in captionids
      ]
      captionids = pad_sequence(captionids)
      captionids = torch.transpose(captionids, 0, 1).cuda() # (b, t, captionids)
      caption_lens = torch.LongTensor(data['caption_lens']).cuda()

      with torch.no_grad():
        out_wids = self.model('val', fts, captionids=captionids, caption_lens=caption_lens)
      # print out_wids

      out_wids = out_wids.data.cpu().numpy()
      for i, sent in enumerate(out_wids):
        vid = data['vids'][i]
        vid2predicts[vid] = self.int2str(np.expand_dims(sent, 0))
      base += self.model_cfg.tst_batch_size

    metrics = {}

    bleu_scorer = bleu.Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(self.tst_reader.videoid2captions, vid2predicts)
    for i in range(4):
      metrics['bleu%d'%(i+1)] = bleu_score[i]

    cider_scorer = cider.Cider()
    cider_score, _ = cider_scorer.compute_score(self.tst_reader.videoid2captions, vid2predicts)
    metrics['cider'] = cider_score

    return metrics

  def predict_in_tst(self):
    vid2predict = {}
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda()

      if self.model_cfg.strategy == 'beam':
        with torch.no_grad():
          beam_cum_log_probs, beam_pres, beam_ends, out_wids = self.model(
            'tst', fts, strategy='beam')
        beam_cum_log_probs = beam_cum_log_probs.data.cpu().numpy()
        beam_pres = beam_pres.data.cpu().numpy()
        beam_ends = beam_ends.data.cpu().numpy()
        out_wids = out_wids.data.cpu().numpy()

        candidates = model.util.beamsearch_recover_captions(out_wids, beam_cum_log_probs, beam_pres, beam_ends)

      for i, candidate in enumerate(candidates):
        vid = data['vids'][i]
        sent = np.array(candidate, dtype=np.int)
        predict = self.int2str(np.expand_dims(sent, 0))[0]
        vid2predict[str(vid)] = predict

    with open(self.path_cfg.predict_file, 'w') as fout:
      json.dump(vid2predict, fout)

  def change_lr(self, history, metrics):
    if len(history) < 10: # warmup stage:
      num = len(history)
      delta = (self.model_cfg.base_lr - self.model_cfg.min_lr) / 10
      lr = self.model_cfg.min_lr + num * delta
      model = self.model
      if isinstance(self.model, nn.DataParallel):
        model = model.module
      self.optimizer = torch.optim.Adam(model.trainable_params(), lr=lr)
    else:
      patience = 5
      max_metric = 0.
      max_epoch = -1
      for i, metric in enumerate(history):
        if metric[self.model_cfg.metric] > max_metric:
          max_metric = metric[self.model_cfg.metric]
          max_epoch = i

      if metrics[self.model_cfg.metric] < max_metric and len(history) - max_epoch >= patience:
        self.model_cfg.base_lr *= .5
        if self.model_cfg.base_lr < self.model_cfg.min_lr:
          self.model_cfg.base_lr = self.model_cfg.min_lr
        model = self.model
        if isinstance(self.model, nn.DataParallel):
          model = model.module
        self.optimizer = torch.optim.Adam(model.trainable_params(), lr=self.model_cfg.base_lr)

    history.append(metrics)

    return history


class TrnTstFixLr(TrnTst):
  def change_lr(self, history, metrics):
    history.append(metrics)
    return history


class TrnTstDecode(TrnTst):
  def predict_in_tst(self):
    vid2predict = {}
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda()

      if self.model_cfg.strategy == 'beam':
        with torch.no_grad():
          beam_cum_log_probs, beam_pres, beam_ends, out_wids = self.model(
            'tst', fts, strategy='beam')
        beam_cum_log_probs = beam_cum_log_probs.data.cpu().numpy()
        beam_pres = beam_pres.data.cpu().numpy()
        beam_ends = beam_ends.data.cpu().numpy()
        out_wids = out_wids.data.cpu().numpy()

        candidate_scores = model.util.beamsearch_recover_multiple_captions(
          out_wids, beam_cum_log_probs, beam_pres, beam_ends, self.model_cfg.pool_size)
      else:
        with torch.no_grad():
          out_wids, log_probs = self.model('tst', fts, strategy=self.model_cfg.strategy)
        out_wids = out_wids.data.cpu().numpy()
        log_probs = log_probs.data.cpu().numpy()

        candidate_scores = model.util.recover_sample_caption_scores(
          out_wids, log_probs, self.model_cfg.subcfgs[DEC].num_sample)
        
      for i, candidate_score in enumerate(candidate_scores):
        vid = data['vids'][i]
        out = []
        for d in candidate_score:
          sent = np.array(d['sent'])
          predict = self.int2str(np.expand_dims(sent, 0))[0]
          score = float(d['score'])
          out.append({
            'sent': predict,
            'score': score,
          }) 
        vid2predict[vid] = out

    with open(self.path_cfg.predict_file, 'w') as fout:
      json.dump(vid2predict, fout, indent=2)
