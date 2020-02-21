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
import encoder.pca
import decoder.rnn_rl
import d.simple
import model.util
import model.gan_simple_sc
import model.cached_cider

ENC = 'encoder'
DEC = 'decoder'
DIS = 'discriminator'

class ModelConfig(framework.GanModelConfig):
  def __init__(self):
    super(ModelConfig, self).__init__()

    self.subcfgs[ENC] = encoder.pca.Config()
    self.subcfgs[DEC] = decoder.rnn_rl.Config()
    self.subcfgs[DIS] = d.simple.Config()

    self.strategy = 'beam'

    self.reward_alpha = 1.


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
  cfg.reward_alpha = kwargs['reward_alpha']

  enc_cfg = cfg.subcfgs[ENC]
  enc_cfg.dim_ft = kwargs['dim_ft']
  enc_cfg.dim_output = kwargs['dim_hidden']

  dec_cfg = cfg.subcfgs[DEC]
  dec_cfg.cell = kwargs['cell']
  dec_cfg.dim_embed = kwargs['dim_embed']
  dec_cfg.dim_hidden = kwargs['dim_hidden']
  dec_cfg.dropin = kwargs['g_dropin']
  dec_cfg.dropout = kwargs['g_dropout']
  dec_cfg.max_step = kwargs['max_step']
  dec_cfg.tied = kwargs['tied']
  dec_cfg.beam_width = kwargs['beam_width']
  dec_cfg.init_fg = kwargs['init_fg']
  dec_cfg.num_sample = kwargs['num_sample']

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


Model = model.gan_simple_sc.Model
PathCfg = model.util.RLPathCfg


class TrnTst(model.gan_simple_sc.TrnTst):
  def __init__(self, model_cfg, path_cfg, gpuids):
    super(TrnTst, self).__init__(model_cfg, path_cfg, gpuids)

    gt_file = path_cfg.groundtruth_file
    df_file = path_cfg.df_file
    self.cider = model.cached_cider.CiderScorer()
    self.cider.load(df_file, gt_file)

  def g_feed_data_forward_backward(self, data):
    # 1. sample from g
    fts = torch.Tensor(data['fts']).cuda()
    sample_out_wids, log_probs, greedy_out_wids = self.model('g_sample', fts=fts)

    # 2. eval cider reward
    sample_out_wids_data = sample_out_wids.data.cpu().numpy()
    greedy_out_wids_data = greedy_out_wids.data.cpu().numpy()
    vids = data['vids']
    sample_scores = model.util.eval_cider_in_rollout(
      sample_out_wids_data, vids, self.int2str, self.cider) # (b, num_sample)
    greedy_scores = model.util.eval_cider_in_rollout(
      np.expand_dims(greedy_out_wids_data, 1), vids, self.int2str, self.cider) # (b, 1)
    cider_rewards = sample_scores - greedy_scores
    cider_rewards = cider_rewards.reshape((-1,))
    cider_rewards = torch.Tensor(cider_rewards).cuda()

    # 3. eval d reward
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

    d_rewards = log_p_sample - log_p_greedy.unsqueeze(1) # (b, num_sample)
    d_rewards = d_rewards.view(b*num_sample)

    rewards = (d_rewards + self.model_cfg.reward_alpha * cider_rewards) / (1. + self.model_cfg.reward_alpha)
    rewards = rewards.detach()

    self.op2monitor['d_rewards'] = d_rewards.mean()
    self.op2monitor['cider_rewards'] = cider_rewards.mean()

    # print d_rewards.mean().item(), cider_rewards.mean().item()

    # 4. train by surrogate loss
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
    fts = torch.Tensor(data['fts']).cuda()
    sample_out_wids, log_probs, greedy_out_wids = self.model('g_sample', fts=fts)

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
