import sys
import os
import json
import pickle
import pprint
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
from bleu import bleu
from cider import cider

from base import framework
import decoder.att_rnn_rl
import d.simple
import model.util

DEC = 'decoder'
DIS = 'discriminator'

class ModelConfig(framework.GanModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[DEC] = decoder.att_rnn_rl.Config()
    self.subcfgs[DIS] = d.simple.Config()

    self.strategy = 'beam'


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.trn_batch_size = 64
  cfg.tst_batch_size = 64
  cfg.val_iter = -1
  cfg.monitor_iter = 100

  cfg.g_num_epoch = kwargs['g_num_epoch']
  cfg.g_base_lr = kwargs['g_lr']
  cfg.g_freeze = kwargs['g_freeze']
  cfg.g_freeze_epoch = kwargs['g_freeze_epoch']

  cfg.d_num_epoch = kwargs['d_num_epoch']
  cfg.d_base_lr = kwargs['d_lr']
  cfg.d_iter = kwargs['d_iter']
  cfg.d_val_acc = kwargs['d_val_acc']
  cfg.d_buffer_size = kwargs['d_buffer_size']

  dec_cfg = cfg.subcfgs[DEC]
  dec_cfg.dim_embed = kwargs['dim_embed']
  dec_cfg.dim_att_ft = kwargs['dim_att_ft']
  dec_cfg.dropin = kwargs['g_dropin']
  dec_cfg.dropout = kwargs['g_dropout']
  dec_cfg.max_step = kwargs['max_step']
  dec_cfg.beam_width = kwargs['beam_width']
  dec_cfg.tied_key_val = kwargs['tied_key_val']
  dec_cfg.val_proj = kwargs['val_proj']

  cell_cfg = dec_cfg.subcfgs[decoder.att_rnn_rl.CELL]
  cell_cfg.dim_embed = kwargs['dim_embed']
  cell_cfg.dim_hidden = kwargs['dim_hidden']
  cell_cfg.dim_key = kwargs['dim_key']
  cell_cfg.dim_val = kwargs['dim_val']
  cell_cfg.num_att_ft = kwargs['num_att_ft']
  cell_cfg.dim_boom = kwargs['dim_boom']

  att_cfg = cell_cfg.subcfgs[decoder.att_rnn_rl.ATT]
  att_cfg.dim_hidden = kwargs['dim_hidden']
  att_cfg.dim_key = kwargs['dim_key']
  att_cfg.dim_val = kwargs['dim_val']
  att_cfg.num_att_ft = kwargs['num_att_ft']
  att_cfg.sim = kwargs['sim']

  dis_cfg = cfg.subcfgs[DIS]
  dis_cfg.dim_kernel = kwargs['dim_kernel']
  dis_cfg.num_kernel = kwargs['num_kernel']
  dis_cfg.noise = kwargs['d_noise']
  dis_cfg.dim_ft = kwargs['dim_ft']

  sent_enc_cfg = dis_cfg.subcfgs[d.simple.SE]
  sent_enc_cfg.cell = kwargs['cell']
  sent_enc_cfg.dim_embed = kwargs['dim_embed']
  sent_enc_cfg.dim_hidden = kwargs['dim_hidden']
  sent_enc_cfg.dropin = kwargs['d_noise']

  return cfg


class Model(nn.Module):
  def __init__(self, config):
    super(Model, self).__init__()

    self._config = config
    dec_cfg = self._config.subcfgs[DEC]
    dis_cfg = self._config.subcfgs[DIS]

    self.decoder = decoder.att_rnn_rl.Decoder(dec_cfg)
    self.discriminator = d.simple.Discriminator(dis_cfg)

    self.op2monitor = {}

  def forward(self, mode, **kwargs):
    if mode == 'g_trn':
      log_probs = kwargs['log_probs']
      log_masks = kwargs['log_masks']
      rewards = kwargs['rewards']

      b = rewards.size(0)
      loss = -rewards.view(b, 1) * log_probs
      loss = torch.sum(loss * log_masks) / torch.sum(log_masks)

      self.op2monitor['reward'] = rewards.mean()
      self.op2monitor['loss'] = loss

      return loss
    elif mode == 'd_trn':
      fts = kwargs['fts']
      sents = kwargs['sents']
      lens = kwargs['lens']
      y = kwargs['y']

      loss, dist2img = self.discriminator('trn', fts, sents, lens, y=y)
      return loss
    elif mode == 'g_sample':
      att_fts = kwargs['att_fts']
      att_masks = kwargs['att_masks']

      cell_config = self._config.subcfgs[DEC].subcfgs[decoder.att_rnn_rl.CELL]
      b = att_fts.size(0)
      init_state = torch.zeros(b, cell_config.dim_hidden).cuda()
      init_state = (init_state, init_state)

      return self.decoder('sample', init_state, att_fts, att_masks)
    elif mode == 'd_eval':
      fts = kwargs['fts']
      sents = kwargs['sents']
      lens = kwargs['lens']

      return self.discriminator('eval', fts, sents, lens)
    elif mode == 'd_val':
      fts = kwargs['fts']
      sents = kwargs['sents']
      lens = kwargs['lens']
      
      return self.discriminator('val', fts, sents, lens)
    elif mode == 'g_val':
      att_fts = kwargs['att_fts']
      att_masks = kwargs['att_masks']

      cell_config = self._config.subcfgs[DEC].subcfgs[decoder.att_rnn_rl.CELL]
      b = att_fts.size(0)
      init_state = torch.zeros(b, cell_config.dim_hidden).cuda()
      init_state = (init_state, init_state)

      return self.decoder('val', init_state, att_fts, att_masks)
    elif mode == 'g_tst':
      att_fts = kwargs['att_fts']
      att_masks = kwargs['att_masks']

      cell_config = self._config.subcfgs[DEC].subcfgs[decoder.att_rnn_rl.CELL]
      b = att_fts.size(0)
      init_state = torch.zeros(b, cell_config.dim_hidden).cuda()
      init_state = (init_state, init_state)

      return self.decoder('tst', init_state, att_fts, att_masks, strategy=kwargs['strategy'])

  def g_trainable_params(self):
    params = []
    for name, param in self.named_parameters():
      if name.startswith('encoder') or name.startswith('decoder'):
        print(name)
        params.append(param)
    return params

  def d_trainable_params(self):
    params = []
    for name, param in self.named_parameters():
      if name.startswith('discriminator'):
        print(name)
        params.append(param)
    return params


PathCfg = model.util.AttPathCfg


class TrnTst(framework.GanTrnTst):
  def __init__(self, model_cfg, path_cfg, gpuids):
    super(TrnTst, self).__init__(model_cfg, path_cfg, gpuids)

    self.int2str = model.util.CaptionInt2str(path_cfg.word_file)

  def g_feed_data_forward_backward(self, data):
    # 1. sample from g
    att_fts = torch.Tensor(data['att_fts']).cuda()
    att_masks = torch.ones(att_fts.size()[:-1]).cuda()
    sample_out_wids, log_probs, greedy_out_wids = self.model('g_sample', att_fts=att_fts, att_masks=att_masks)

    # 2. eval d to get reward
    EOS = 1
    b, num_sample, _ = sample_out_wids.size()
    sample_out_wids = sample_out_wids.view(b*num_sample, -1)
    lens = []
    for sample in sample_out_wids:
      for k, wid in enumerate(sample):
        if wid == EOS:
          lens.append(k) # exclude EOS
          break
      else:
        lens.append(k+1)
    lens = torch.LongTensor(lens).cuda()

    fts = torch.Tensor(data['fts']).cuda()
    b, f = fts.size()
    expand_fts = fts.unsqueeze(1).expand(b, num_sample, f).view(b*num_sample, f)
    log_p_sample = self.model('d_eval', fts=expand_fts, sents=sample_out_wids.transpose(0, 1), lens=lens)
    log_p_sample = log_p_sample.view(b, num_sample)

    data['samples'] = sample_out_wids.data.cpu().numpy()
    data['sample_lens'] = lens.data.cpu().tolist()

    lens = []
    for greedy in greedy_out_wids:
      for k, wid in enumerate(greedy):
        if wid == EOS:
          lens.append(k) # exclude EOS
          break
      else:
        lens.append(k+1)
    lens = torch.LongTensor(lens).cuda()
    log_p_greedy = self.model('d_eval', fts=fts, sents=greedy_out_wids.transpose(0, 1), lens=lens)

    rewards = log_p_sample - log_p_greedy.unsqueeze(1) # (b, num_sample)
    rewards = rewards.view(b*num_sample)

    # 3. train by surrogate loss
    log_probs = log_probs.view(b*num_sample, -1)
    log_masks = torch.zeros_like(log_probs)
    for i, sample in enumerate(sample_out_wids):
      for j, wid in enumerate(sample):
        if wid == EOS:
          log_masks[i, :j+1] = 1.
    log_masks = log_masks.cuda()

    loss = self.model(
      'g_trn', log_probs=log_probs, log_masks=log_masks, rewards=rewards)
    loss.backward()

  def g_feed_data_forward(self, data):
    att_fts = torch.Tensor(data['att_fts']).cuda()
    att_masks = torch.ones(att_fts.size()[:-1]).cuda()
    sample_out_wids, log_probs, greedy_out_wids = self.model('g_sample', att_fts=att_fts, att_masks=att_masks)

    EOS = 1
    b, num_sample, _ = sample_out_wids.size()
    sample_out_wids = sample_out_wids.view(b*num_sample, -1)
    lens = []
    for sample in sample_out_wids:
      for k, wid in enumerate(sample):
        if wid == EOS:
          lens.append(k) # exclude EOS
          break
      else:
        lens.append(k+1)
    lens = torch.LongTensor(lens).cuda()
    data['samples'] = sample_out_wids.data.cpu().numpy()
    data['sample_lens'] = lens.data.cpu().tolist()

  def d_feed_data_forward_backward(self, data):
    fts = torch.Tensor(data['fts']).cuda()
    captionids = model.util.pad_sequence(data['pos_captionids'])
    captionids = torch.LongTensor(captionids).cuda()
    lens = torch.LongTensor(data['pos_lens']).cuda()
    b = fts.size(0)
    y = torch.ones(b, dtype=torch.long).cuda()
    loss = self.model('d_trn', fts=fts, sents=captionids, lens=lens, y=y)

    captionids = model.util.pad_sequence(data['neg_captionids'])
    captionids = torch.LongTensor(captionids).cuda()
    lens = torch.LongTensor(data['neg_lens']).cuda()
    y = torch.zeros(b, dtype=torch.long).cuda()
    loss += self.model('d_trn', fts=fts, sents=captionids, lens=lens, y=y)

    b, f = fts.size()
    num_sample = self.model_cfg.subcfgs[DEC].num_sample
    fts = fts.unsqueeze(1).expand(b, num_sample, f).view(-1, f)
    captionids = torch.LongTensor(data['samples']).transpose(0, 1).cuda()
    lens = torch.LongTensor(data['sample_lens']).cuda()
    y = torch.zeros(b, dtype=torch.long).cuda()
    loss += self.model('d_trn', fts=fts, sents=captionids, lens=lens, y=y)

    loss.backward()
    return loss

  # return acc
  def d_validation(self, buffer):
    hit = 0.
    cnt = 0
    for data in buffer:
      fts = torch.Tensor(data['fts']).cuda()
      captionids = model.util.pad_sequence(data['pos_captionids'])
      captionids = torch.LongTensor(captionids).cuda()
      lens = torch.LongTensor(data['pos_lens']).cuda()
      predicts = self.model('d_val', fts=fts, sents=captionids, lens=lens)
      hit += torch.sum(predicts[:, 1] > predicts[:, 0]).item()
      cnt += lens.size(0)

      b, f = fts.size()
      num_sample = self.model_cfg.subcfgs[DEC].num_sample
      fts = fts.unsqueeze(1).expand(b, num_sample, f).view(-1, f)
      captionids = torch.LongTensor(data['samples']).transpose(0, 1).cuda()
      lens = torch.LongTensor(data['sample_lens']).cuda()
      b = captionids.size(0)
      predicts = self.model('d_val', fts=fts, sents=captionids, lens=lens)
      hit += torch.sum(predicts[:, 0] > predicts[:, 1]).item()
      cnt += lens.size(0)
    return hit / cnt

  # return metric dictionary
  def g_validation(self):
    vid2predicts = {}
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      att_fts = torch.Tensor(data['att_fts']).cuda()
      att_masks = torch.ones(att_fts.size()[:-1]).cuda()

      with torch.no_grad():
        out_wids = self.model('g_val', att_fts=att_fts, att_masks=att_masks)

      out_wids = out_wids.data.cpu().numpy()
      for i, sent in enumerate(out_wids):
        vid = data['vids'][i]
        vid2predicts[vid] = self.int2str(np.expand_dims(sent, 0))

    metrics = {}

    bleu_scorer = bleu.Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(self.tst_reader.vid2captions, vid2predicts)
    for i in range(4):
      metrics['bleu%d'%(i+1)] = bleu_score[i]

    cider_scorer = cider.Cider()
    cider_score, _ = cider_scorer.compute_score(self.tst_reader.vid2captions, vid2predicts)
    metrics['cider'] = cider_score

    return metrics

  def g_predict_in_tst(self):
    vid2predict = {}
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      att_fts = torch.Tensor(data['att_fts']).cuda()
      att_masks = torch.ones(att_fts.size()[:-1]).cuda()

      if self.model_cfg.strategy == 'beam':
        with torch.no_grad():
          beam_cum_log_probs, beam_pres, beam_ends, out_wids = self.model(
            'g_tst', att_fts=att_fts, att_masks=att_masks, strategy='beam')
        beam_cum_log_probs = beam_cum_log_probs.data.cpu().numpy()
        beam_pres = beam_pres.data.cpu().numpy()
        beam_ends = beam_ends.data.cpu().numpy()
        out_wids = out_wids.data.cpu().numpy()

        candidates = util.beamsearch_recover_captions(out_wids, beam_cum_log_probs, beam_pres, beam_ends)

        for i, candidate in enumerate(candidates):
          vid = data['vids'][i]
          sent = np.array(candidate, dtype=np.int)
          predict = self.int2str(np.expand_dims(sent, 0))[0]
          vid2predict[str(vid)] = predict

    with open(self.path_cfg.predict_file, 'w') as fout:
      json.dump(vid2predict, fout)
