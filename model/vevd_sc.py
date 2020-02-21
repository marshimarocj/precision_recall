import sys
import os
import json
import pickle
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
from bleu import bleu
from cider import cider

from base import framework
import encoder.pca
import decoder.rnn_rl
import model.util
import model.vevd_ml
import model.cached_cider

ENC = 'encoder'
DEC = 'decoder'

class ModelConfig(framework.ModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[ENC] = encoder.pca.Config()
    self.subcfgs[DEC] = decoder.rnn_rl.Config()

    self.cell = 'gru'
    self.strategy = 'beam'
    self.metric = 'cider'


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.trn_batch_size = 64
  cfg.tst_batch_size = 64
  cfg.num_epoch = kwargs['num_epoch']
  cfg.val_iter = -1
  cfg.monitor_iter = 100
  cfg.base_lr = kwargs['lr']

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
  dec_cfg.num_sample = kwargs['num_sample']

  return cfg


class Model(nn.Module):
  def __init__(self, config):
    super(Model, self).__init__()

    self._config = config
    enc_cfg = self._config.subcfgs[ENC]
    dec_cfg = self._config.subcfgs[DEC]

    self.encoder = encoder.pca.Encoder(enc_cfg)
    self.decoder = decoder.rnn_rl.Decoder(dec_cfg)

    self.op2monitor = {}

  def forward(self, mode, **kwargs):
    if mode == 'trn':
      log_probs = kwargs['log_probs'] # (b, t)
      log_masks = kwargs['log_masks'] # (b, t)
      rewards = kwargs['rewards'] # (b,)

      b = rewards.size(0)
      loss = -rewards.view(b, 1) * log_probs
      loss = torch.sum(loss * log_masks) / torch.sum(log_masks)

      return loss
    else:
      ft = kwargs['ft']
      embed = self.encoder(ft)
      if self._config.subcfgs[DEC].cell == 'lstm':
        init_state = (embed, embed)
      elif self._config.subcfgs[DEC].cell == 'gru':
        init_state = embed

      if mode == 'sample':
        return self.decoder(mode, init_state)
      elif mode == 'val':
        return self.decoder(mode, init_state, 
          y=kwargs['captionids'], lens=kwargs['caption_lens'])
      elif mode == 'tst':
        return self.decoder(mode, init_state, strategy=kwargs['strategy'])

  def trainable_params(self):
    params = []
    for name, param in self.named_parameters():
      if param.requires_grad:
        print(name)
        params.append(param)
    return params


PathCfg = model.util.RLPathCfg


class TrnTst(model.vevd_ml.TrnTstFixLr):
  def __init__(self, model_cfg, path_cfg, gpuids):
    super(TrnTst, self).__init__(model_cfg, path_cfg, gpuids)

    self.int2str = model.util.CaptionInt2str(path_cfg.word_file)

    gt_file = path_cfg.groundtruth_file
    df_file = path_cfg.df_file
    self.cider = model.cached_cider.CiderScorer()
    self.cider.load(df_file, gt_file)

  def feed_data_forward_backward(self, data):
    # 1. sample
    fts = torch.Tensor(data['fts']).cuda()

    sample_out_wids, log_probs, greedy_out_wids = self.model('sample', ft=fts)
    sample_out_wids = sample_out_wids.data.cpu().numpy() # (b, num_sample, max_step)
    greedy_out_wids = greedy_out_wids.data.cpu().numpy() # (b, max_step)
    b, num_sample, step = sample_out_wids.shape

    # 2. calculate reward
    vids = data['vids']
    captionids = data['captionids']

    sample_scores = model.util.eval_cider_in_rollout(
      sample_out_wids, vids, self.int2str, self.cider) # (b, num_sample)
    greedy_scores = model.util.eval_cider_in_rollout(
      np.expand_dims(greedy_out_wids, 1), vids, self.int2str, self.cider) # (b, 1)
    rewards = sample_scores - greedy_scores
    rewards = rewards.reshape((b*num_sample,))
    rewards = torch.Tensor(rewards).cuda() # (b*num_sample,)

    sample_out_wids = sample_out_wids.reshape((b*num_sample,-1))
    sample_out_wids = model.util.transform_predict_captionid_array_to_list(sample_out_wids)

    log_probs = log_probs.view(b*num_sample, step)
    log_masks = torch.zeros_like(log_probs)
    for i, predict in enumerate(sample_out_wids):
      l = len(predict)-1 # exclude BOS
      log_masks[i, :l] = 1.
    log_masks = log_masks.cuda()

    # 3. train by surrogate loss
    loss = self.model(
      'trn', log_probs=log_probs, log_masks=log_masks, rewards=rewards)
    loss.backward()

    return loss

  def validation(self):
    vid2predicts = {}
    base = 0
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda(0)
      # print fts.size()

      captionids = data['captionids']
      captionids = [
        torch.LongTensor(captionid).cuda(0)
        for captionid in captionids
      ]
      caption_lens = torch.LongTensor(data['caption_lens']).cuda(0)

      with torch.no_grad():
        out_wids = self.model('val', ft=fts, captionids=captionids, caption_lens=caption_lens)

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
    base = 0
    for data in self.tst_reader.yield_batch(self.model_cfg.tst_batch_size):
      fts = torch.Tensor(data['fts']).cuda(0)

      if self.model_cfg.strategy == 'beam':
        with torch.no_grad():
          beam_cum_log_probs, beam_pres, beam_ends, out_wids = self.model(
            'tst', ft=fts, strategy='beam')
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
      base += self.model_cfg.tst_batch_size

    with open(self.path_cfg.predict_file, 'w') as fout:
      json.dump(vid2predict, fout)
