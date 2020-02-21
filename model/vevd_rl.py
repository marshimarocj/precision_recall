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
import model.vevd_sc
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

    self.use_greedy = True


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.trn_batch_size = 64
  cfg.tst_batch_size = 64
  cfg.num_epoch = kwargs['num_epoch']
  cfg.val_iter = -1
  cfg.monitor_iter = 100
  cfg.base_lr = kwargs['lr']

  cfg.cell = kwargs['cell']
  cfg.use_greedy = kwargs['use_greedy']

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

Model = model.vevd_sc.Model

PathCfg = model.vevd_sc.PathCfg


class TrnTst(model.vevd_sc.TrnTst):
  def feed_data_forward_backward(self, data):
    # 1. sample
    fts = torch.Tensor(data['fts']).cuda()

    sample_out_wids, log_probs, greedy_out_wids = self.model('sample', ft=fts)
    sample_out_wids = sample_out_wids.data.cpu().numpy() # (b, num_sample, max_step)
    greedy_out_wids = greedy_out_wids.data.cpu().numpy() # (b, max_step)
    b, num_sample, step = sample_out_wids.shape

    # 2. calculate reward
    vids = data['vids']

    sample_scores = model.util.eval_cider_in_rollout(
      sample_out_wids, vids, self.int2str, self.cider) # (b, num_sample)
    greedy_scores = model.util.eval_cider_in_rollout(
      np.expand_dims(greedy_out_wids, 1), vids, self.int2str, self.cider) # (b, 1)
    if self.model_cfg.use_greedy:
      baseline_scores = np.concatenate([sample_scores[:, 1:], greedy_scores], 1)
    else:
      baseline_scores = sample_scores[:, 1:]
    baseline_scores = np.mean(baseline_scores, 1)
    rewards = sample_scores[:, 0] - baseline_scores
    rewards = torch.Tensor(rewards).cuda() # (b,)

    sample_out_wids = sample_out_wids[:, 0] # (b, max_step)
    sample_out_wids = model.util.transform_predict_captionid_array_to_list(sample_out_wids)

    log_probs = log_probs[:, 0]
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
