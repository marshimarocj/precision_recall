import sys
import math
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base import framework

CELL = 'cell'
ATT = 'attention'

class Config(framework.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.num_word = 10000
    self.dim_embed = 256
    self.dim_att_ft = 1024
    self.dropin = .5
    self.dropout = .5
    self.max_step = 15
    self.beam_width = 5
    self.tied_key_val = True
    self.val_proj = True

    self.subcfgs[CELL] = CellConfig()

  def _assert(self):
    assert self.dim_embed == self.subcfgs[CELL].dim_embed


class CellConfig(framework.ModuleConfig):
  def __init__(self):
    super(CellConfig, self).__init__()

    self.dim_hidden = 256
    self.dim_key = 256
    self.dim_val = 256
    self.num_att_ft = 36
    self.dim_boom = 2048

    self.subcfgs[ATT] = AttConfig()

  def _assert(self):
    assert self.dim_hidden == self.subcfgs[ATT].dim_hidden
    assert self.dim_key == self.subcfgs[ATT].dim_key
    assert self.dim_val == self.subcfgs[ATT].dim_val
    assert self.num_att_ft == self.subcfgs[ATT].num_att_ft
    assert self.dim_boom % self.dim_hidden == 0


class AttConfig(framework.ModuleConfig):
  def __init__(self):
    super(AttConfig, self).__init__()

    self.dim_hidden = 256
    self.dim_key = 256
    self.dim_val = 256
    self.num_att_ft = 36
    self.sim = 'add' # add | product


class Attention(nn.Module):
  def __init__(self, config):
    super(Attention, self).__init__()

    self._config = config

    self.h2key = nn.Linear(self._config.dim_hidden, self._config.dim_key)
    if config.sim == 'add':
      self.alpha_net = nn.Linear(self._config.dim_key, 1)

  def forward(self, q, k, v, att_masks):
    """
    input:
      q: (b, dim_hidden)
      k: (b, num_att_ft, dim_key)
      v: (b, num_att_ft, dim_val)
      att_masks: (b, num_att_ft)
    output:
      att_res: (b, dim_val)
    """
    # The k and v are already projected
    att_h = self.h2key(q)  # (None, dim_key)
    att_h = att_h.unsqueeze(1) # (None, 1, dim_key)

    if self._config.sim == 'add':
      sim = k + att_h # (None, num_att_ft, dim_key)
      sim = torch.tanh(sim) # (None, num_att_ft, dim_key)
      sim = sim.view(-1, self._config.dim_key) # (None*num_att_ft, dim_key)
      sim = self.alpha_net(sim) # (None*num_att_ft, 1)
      sim = sim.view(-1, self._config.num_att_ft) # (None, num_att_ft)
    elif self._config.sim == 'product':
      sim = torch.sum(k * att_h, 2) # (None, num_att_ft)
      sim = sim / math.sqrt(self._config.dim_key)
    
    weight = F.softmax(sim, dim=1) # (None, num_att_ft)
    weight = weight * att_masks.float()
    weight = weight / weight.sum(1, keepdim=True) # normalize to 1
    att_res = torch.bmm(weight.unsqueeze(1), v).squeeze(1) # (None, dim_val)

    return att_res


class Att2InBoom(nn.Module):
  def __init__(self, config):
    super(Att2InBoom, self).__init__()

    self._config = config

    # Build a LSTM
    self.a2b = nn.Linear(self._config.dim_val, self._config.dim_boom)
    self.i2h = nn.Linear(self._config.dim_embed, 4 * self._config.dim_hidden) # i, f, o, c
    self.h2h = nn.Linear(self._config.dim_hidden, 4 * self._config.dim_hidden) # i, f, o, c

    self.attention = Attention(self._config.subcfgs[ATT])

  def forward(self, input, state, k, v, masks):
    def gelu(x):
      return x * torch.sigmoid(1.702 * x)

    h, c = state
    att_res = self.attention(h, k, v, masks)
    att_res = self.a2b(att_res)
    att_res = gelu(att_res)
    num_group = self._config.dim_boom / self._config.dim_hidden
    att_res = att_res.view(-1, num_group, self._config.dim_hidden)
    att_res = torch.sum(att_res, 1) # (b, dim_hidden)

    all_input_sums = self.i2h(input) + self.h2h(h)
    sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self._config.dim_hidden)
    sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
    in_gate = sigmoid_chunk.narrow(1, 0, self._config.dim_hidden)
    forget_gate = sigmoid_chunk.narrow(1, self._config.dim_hidden, self._config.dim_hidden)
    out_gate = sigmoid_chunk.narrow(1, self._config.dim_hidden * 2, self._config.dim_hidden)

    in_transform = all_input_sums.narrow(1, 3 * self._config.dim_hidden, self._config.dim_hidden) + att_res
    next_c = forget_gate * c + in_gate * in_transform
    next_h = out_gate * torch.tanh(next_c)

    return (next_h, next_c)


class Decoder(nn.Module):
  def __init__(self, config):
    super(Decoder, self).__init__()

    self._config = config
    self.embed = nn.Embedding(self._config.num_word, self._config.dim_embed)
    torch.nn.init.xavier_uniform_(self.embed.weight)

    self.word_dist = nn.Linear(self._config.subcfgs[CELL].dim_hidden, self._config.num_word)
    torch.nn.init.xavier_uniform_(self.word_dist.weight)
    self.word_dist.bias.data.fill_(0.)

    cell_config = self._config.subcfgs[CELL]
    self.cell = Att2InBoom(cell_config)

    self.key_proj = nn.Conv1d(self._config.dim_att_ft, cell_config.dim_key, 1, bias=False)
    if not self._config.val_proj:
      self.val_proj = lambda x: x
    elif self._config.tied_key_val:
      self.val_proj = self.key_proj
    else:
      self.val_proj = nn.Conv1d(self._config.dim_att_ft, cell_config.dim_val, 1, bias=False)

    self.dropin = nn.Dropout(p=self._config.dropin)
    self.dropout = nn.Dropout(p=self._config.dropout)

  def forward(self, mode, init_state, att_fts, att_masks, **kwargs):
    if mode == 'trn':
      y = kwargs['y']
      lens = kwargs['lens']
      return self.log_prob(init_state, att_fts, att_masks, y, lens)
    elif mode == 'val':
      return self.greedy_decode(init_state, att_fts, att_masks)
    elif mode == 'tst':
      if kwargs['strategy'] == 'beam':
        return self.beam_decode(init_state, att_fts, att_masks)

  def log_prob(self, init_state, att_fts, att_masks, y, lens):
    """
    input: 
      init_state: (b, dim_hidden) or pair of this shape for LSTM cell
      att_fts: (b, num_att_ft, dim_att_ft)
      att_masks: (b, num_att_ft)
      lens: (b,)
    output:
      logits: (b, t-1, num_word)
      logit_mask: (b, t-1)
      labels: (b, t-1)
    """
    y = torch.transpose(y, 0, 1) # (t, b)
    y_embed = self.embed(y) # (t, b, dim_embed)
    y_embed = self.dropin(y_embed)

    t_steps = y.size(0)
    b = y.size(1)
    # print y_embed.size()
    out_mask = torch.arange(t_steps).long().view(t_steps, 1).expand(t_steps, b).cuda()
    out_mask = out_mask < lens.view(1, b)
    out_mask = out_mask[1:] # one shift as <BOS> doesn't appear in output
    out_mask = out_mask.transpose(0, 1).float() # (b, t-1)

    labels = y[1:].transpose(0, 1) # (b, t-1)

    att_fts = torch.transpose(att_fts, 1, 2) # (b, dim_att_ft, num_att_ft)
    keys = self.key_proj(att_fts)
    keys = torch.transpose(keys, 1, 2).contiguous()
    vals = self.val_proj(att_fts)
    vals = torch.transpose(vals, 1, 2).contiguous()

    state = init_state
    logits = []
    for t in range(t_steps-1):
      # print t
      input = y_embed[t]
      state = self.cell(input, state, keys, vals, att_masks)
      output = state[0]
      output = self.dropout(output)

      logit = self.word_dist(output)
      logits.append(logit)
    logits = torch.stack(logits).transpose(0, 1) # (b, t-1, num_word)

    logits = logits.contiguous().view(b*(t_steps-1), -1)
    labels = labels.contiguous().view(b*(t_steps-1))
    log_prob = -F.cross_entropy(logits, labels, reduction='none')
    log_prob = log_prob.view(b, t_steps-1)

    return log_prob, out_mask

  def greedy_decode(self, init_state, att_fts, att_masks):
    """
    input: 
      init_state: (b, dim_hidden) or pair of this shape for LSTM cell 
      att_fts: (b, num_att_ft, dim_att_ft)
      att_masks: (b, num_att_ft)
    output:
      out_wids: (b, t)
    """
    b = init_state[0].size(0)
    wordids = torch.zeros(b, dtype=torch.long).cuda()

    att_fts = torch.transpose(att_fts, 1, 2) # (b, dim_att_ft, num_att_ft)
    keys = self.key_proj(att_fts)
    keys = torch.transpose(keys, 1, 2).contiguous()
    vals = self.val_proj(att_fts)
    vals = torch.transpose(vals, 1, 2).contiguous()

    state = init_state
    out_wids = []
    for t in range(self._config.max_step):
      embed = self.embed(wordids)
      
      input = embed
      state = self.cell(input, state, keys, vals, att_masks)
      output = state[0]

      logit = self.word_dist(output)
      wordids = torch.argmax(logit, 1)
      out_wids.append(wordids)
    out_wids = torch.stack(out_wids, 1) # (b, t)
    return out_wids

  def sample_decode(self, init_state, att_fts, att_masks):
    max_step = self._config.max_step
    num_sample = self._config.num_sample
    num_word = self._config.num_word

    cell_config = self._config.subcfgs[CELL]
    dim_hidden = cell_config.dim_hidden
    dim_embed = cell_config.dim_embed
    dim_key = cell_config.dim_key
    dim_val = cell_config.dim_val
    num_att_ft = cell_config.num_att_ft

    b = init_state[0].size(0)
    state = [
      s.unsqueeze(1).expand(b, num_sample, dim_hidden).contiguous().view(b*num_sample, dim_hidden)
      for s in init_state
    ]
    wordids = torch.zeros(b*num_sample, dtype=torch.long).cuda()

    att_fts = torch.transpose(att_fts, 1, 2) # (b, dim_att_ft, num_att_ft)
    keys = self.key_proj(att_fts)
    keys = torch.transpose(keys, 1, 2).contiguous()
    vals = self.val_proj(att_fts)
    vals = torch.transpose(vals, 1, 2).contiguous()
    keys = keys.view(b, 1, num_att_ft, dim_key).expand(b, num_sample, num_att_ft, dim_key)
    keys = keys.contiguous().view(b*num_sample, num_att_ft, dim_key)
    vals = vals.view(b, 1, num_att_ft, dim_val).expand(b, num_sample, num_att_ft, dim_val)
    vals = vals.contiguous().view(b*num_sample, num_att_ft, dim_val)
    att_masks = att_masks.view(b, 1, num_att_ft).expand(b, num_sample, num_att_ft)
    att_masks = att_masks.contiguous().view(b*num_sample, num_att_ft)

    out_wids = []
    log_probs = []
    for t in range(max_step):
      embed = self.embed(wordids) # (b*num_sample, dim_embed)
      embed = self.dropin(embed)
      state = self.cell(embed, state, keys, vals, att_masks)
      output = state[0]
      output = self.dropout(output)
      logit = self.word_dist(output) # (b*num_sample, num_word)

      categorical = torch.distributions.categorical.Categorical(logits=logit)
      wordids = categorical.sample() # (b*num_sample, )
      log_prob = -F.cross_entropy(logit, wordids, reduction='none')

      out_wids.append(wordids)
      log_probs.append(log_prob)
    out_wids = torch.stack(out_wids, 1) # (b*num_sample, t)
    log_probs = torch.stack(log_probs, 1) # (b*num_sample, t)
    out_wids = out_wids.contiguous().view(b, num_sample, max_step)
    log_probs = log_probs.contiguous().view(b, num_sample, max_step)

    return out_wids, log_probs

  def beam_decode(self, init_state, att_fts, att_masks):
    EOS = 1

    max_step = self._config.max_step
    topk = self._config.beam_width
    num_word = self._config.num_word
    cell_config = self._config.subcfgs[CELL]
    dim_hidden = cell_config.dim_hidden
    dim_embed = cell_config.dim_embed
    dim_key = cell_config.dim_key
    dim_val = cell_config.dim_val
    num_att_ft = cell_config.num_att_ft

    b = init_state[0].size(0)
    wordids = torch.zeros(b, dtype=torch.long).cuda()

    att_fts = torch.transpose(att_fts, 1, 2) # (b, dim_att_ft, num_att_ft)
    keys = self.key_proj(att_fts)
    keys = torch.transpose(keys, 1, 2).contiguous() # (b, num_att_ft, dim_key)
    vals = self.val_proj(att_fts)
    vals = torch.transpose(vals, 1, 2).contiguous() # (b, num_att_ft, dim_val)

    state = init_state
    beam_cum_log_probs = [] # (max_step, b, topk)
    beam_pres = [] # (max_step, b, topk)
    beam_ends = [] # (max_step, b, topk)
    out_wids = [] # (max_step, b, topk)
    log_prob_topk = None
    for t in range(max_step):
      if t == 0:
        embed = self.embed(wordids) # (b, dim_embed)

        input = embed
        state = self.cell(input, state, keys, vals, att_masks)
        output = state[0]
        logit = self.word_dist(output) # (b, num_word)

        log_softmax = F.log_softmax(logit, dim=1)
        val, idx = torch.topk(log_softmax, topk, 1) # (b, topk)

        out_wids.append(idx)
        pre = torch.arange(b).long().view(b, 1).expand(b, topk).cuda()
        beam_pres.append(pre)
        beam_ends.append(idx == EOS)
        beam_cum_log_probs.append(val)

        log_prob_topk = val.clone()
        log_prob_topk[idx==EOS] = -1e10
        log_prob_topk = log_prob_topk.view(b*topk, 1)

        # expand 
        state = [
          s.view(b, 1, dim_hidden).expand(b, topk, dim_hidden).contiguous().view(b*topk, dim_hidden) 
          for s in state
        ]
        keys = keys.view(b, 1, num_att_ft, dim_key).expand(b, topk, num_att_ft, dim_key)
        keys = keys.contiguous().view(b*topk, num_att_ft, dim_key)
        vals = vals.view(b, 1, num_att_ft, dim_val).expand(b, topk, num_att_ft, dim_val)
        vals = vals.contiguous().view(b*topk, num_att_ft, dim_val)
        att_masks = att_masks.view(b, 1, num_att_ft).expand(b, topk, num_att_ft)
        att_masks = att_masks.contiguous().view(b*topk, num_att_ft)

        wordids = idx.view(b*topk)
      else:
        embed = self.embed(wordids) # (b*topk, dim_embed)

        input = embed
        state = self.cell(input, state, keys, vals, att_masks)
        output = state[0]
        logit = self.word_dist(output) # (b*topk, num_word)
        log_softmax = F.log_softmax(logit, dim=1)

        log_prob = log_prob_topk + log_softmax # (b*topk, num_word)
        log_prob = log_prob.view(b, topk*num_word)
        val, idx_1d = torch.topk(log_prob, topk, 1) # (b, topk)
        pre = idx_1d / num_word
        idx = idx_1d % num_word

        out_wids.append(idx)
        beam_pres.append(pre)
        beam_ends.append(idx == EOS)
        beam_cum_log_probs.append(val)

        log_prob_topk = val.clone()
        log_prob_topk[idx==EOS] = -1e10
        log_prob_topk = log_prob_topk.view(b*topk, 1)
        wordids = idx.view(b*topk)

        # rearrange states
        pre_1d = pre + torch.arange(b).view(b, 1).long().cuda() * topk
        pre_1d = pre_1d.contiguous().view(b*topk)
        state = [torch.index_select(d, 0, pre_1d) for d in state]

    beam_cum_log_probs = torch.stack(beam_cum_log_probs, dim=1) # (b, max_step, topk)
    beam_pres = torch.stack(beam_pres, dim=1) # (b, max_step, topk)
    beam_ends = torch.stack(beam_ends, dim=1) # (b, max_step, topk)
    out_wids = torch.stack(out_wids, dim=1) # (b, max_step, topk)

    return beam_cum_log_probs, beam_pres, beam_ends, out_wids
