import sys
import math
import bisect
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import *
import numpy as np

from base import framework

class Config(framework.ModuleConfig):
  def __init__(self):
    super(Config, self).__init__()

    self.cell = 'lstm'
    self.dim_embed = 256
    self.dim_hidden = 256
    self.num_word = 10000
    self.dropin = .5
    self.dropout = .5
    self.max_step = 15
    self.beam_width = 5
    self.num_sample = 5

    self.tied = False
    self.init_fg = False


class Decoder(nn.Module):
  def __init__(self, config):
    super(Decoder, self).__init__()

    self._config = config

    self.embed = nn.Embedding(self._config.num_word, self._config.dim_embed)
    torch.nn.init.xavier_uniform_(self.embed.weight)

    if self._config.tied:
      self.out = nn.Linear(self._config.dim_hidden, self._config.dim_embed)
      self.word_dist = nn.Linear(self._config.dim_embed, self._config.num_word)
      self.word_dist.weight = self.embed.weight # tied weight
      self.word_dist.bias.data.fill_(0.)
    else:
      self.word_dist = nn.Linear(self._config.dim_hidden, self._config.num_word)
      torch.nn.init.xavier_uniform_(self.word_dist.weight)
      self.word_dist.bias.data.fill_(0.)

    if self._config.cell == 'gru':
      self.cell = nn.GRUCell(self._config.dim_embed, self._config.dim_hidden)
    elif self._config.cell == 'lstm':
      self.cell = nn.LSTMCell(self._config.dim_embed, self._config.dim_hidden)
    stddev = 1 / math.sqrt(self._config.dim_hidden + self._config.dim_embed)
    torch.nn.init.normal_(self.cell.weight_hh, std=stddev)
    torch.nn.init.normal_(self.cell.weight_ih, std=stddev)
    self.cell.bias_hh.data.fill_(0.)
    self.cell.bias_ih.data.fill_(0.)
    if self._config.init_fg:
      self.cell.bias_hh.data[self._config.dim_hidden:2*self._config.dim_hidden].fill_(1.)
      self.cell.bias_ih.data[self._config.dim_hidden:2*self._config.dim_hidden].fill_(1.)

    self.dropin = nn.Dropout(p=self._config.dropin)
    self.dropout = nn.Dropout(p=self._config.dropout)

  def forward(self, mode, init_state, **kwargs):
    if mode == 'trn':
      y = kwargs['y']
      lens = kwargs['lens']
      if 'drop' in kwargs:
        return self.log_prob(init_state, y, lens, drop=kwargs['drop'])
      else:
        return self.log_prob(init_state, y, lens)
    elif mode == 'trn.mixup':
      y = kwargs['y']
      lens = kwargs['lens']
      return self.log_prob_mixup(init_state, y, lens)
    elif mode == 'val':
      return self.greedy_decode(init_state)
    elif mode == 'tst':
      if kwargs['strategy'] == 'beam':
        return self.beam_decode(init_state)
      elif kwargs['strategy'] == 'greedy':
        return self.greedy_decode(init_state)
      elif kwargs['strategy'] == 'sample':
        return self.sample_decode(init_state)
      elif kwargs['strategy'] == 'sample_topk':
        return self.sample_topk_decode(init_state, kwargs['topk'])
      elif kwargs['strategy'] == 'nucleus_sample':
        return self.nucleus_sample_decode(init_state, kwargs['threshold_p'])

  def log_prob(self, init_state, y, lens, drop=True):
    # type: (Any, List[torch.LongTensor], torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]
    """
    input:
      init_state: (b, dim_hidden) or pair of this shape for LSTM cell
      lens: (b,)
    output:
      logits: (b, t-1, num_word)
      logit_mask: (b, t-1)
      labels: (b, t-1)
    """
    y = torch.transpose(y, 0, 1) # (t, b)
    y_embed = self.embed(y) # (t, b, dim_embed)
    if drop:
      y_embed = self.dropin(y_embed)

    t_steps = y.size(0)
    b = y.size(1)
    # print y_embed.size()
    out_mask = torch.arange(t_steps).long().view(t_steps, 1).expand(t_steps, b).cuda()
    out_mask = out_mask < lens.view(1, b)
    out_mask = out_mask[1:] # one shift as <BOS> doesn't appear in output
    out_mask = out_mask.transpose(0, 1).float() # (b, t-1)

    labels = y[1:].transpose(0, 1) # (b, t-1)

    state = init_state
    logits = []
    for t in range(t_steps-1):
      # print t
      input = y_embed[t]
      state = self.cell(input, state)
      if self._config.cell == 'gru':
        output = state
      elif self._config.cell == 'lstm':
        output = state[0]
      if drop:
        output = self.dropout(output)

      if self._config.tied:
        out = self.out(output)
        logit = self.word_dist(out)
      else:
        logit = self.word_dist(output)
      logits.append(logit)
    logits = torch.stack(logits).transpose(0, 1) # (b, t-1, num_word)

    logits = logits.contiguous().view(b*(t_steps-1), -1)
    labels = labels.contiguous().view(b*(t_steps-1))
    log_prob = -F.cross_entropy(logits, labels, reduction='none')
    log_prob = log_prob.view(b, t_steps-1)

    return log_prob, out_mask

  def log_prob_mixup(self, init_state, y, lens):
    # type: (Any, List[torch.LongTensor], torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]
    """
    input:
      init_state: (b, dim_hidden) or pair of this shape for LSTM cell
      lens: (b,)
    output:
      logits: (b, t-1, num_word)
      logit_mask: (b, t-1)
      labels: (b, t-1)
    """
    b, t_steps, num_word = y.size()
    y_embed = torch.matmul(y.view(-1, num_word), self.embed.weight) # (b*t, dim_embed)
    y_embed = self.dropin(y_embed)
    y_embed = y_embed.view(b, t_steps, -1) # (b, t, dim_embed)
    y_embed = torch.transpose(y_embed, 0, 1) # (t, b, dim_embed)

    out_mask = torch.arange(t_steps).long().unsqueeze(1).expand(t_steps, b).cuda()
    out_mask = out_mask < lens.unsqueeze(0)
    out_mask = out_mask[1:] # one shift as <BOS> doesn't appear in output
    out_mask = out_mask.transpose(0, 1).float() # (b, t-1)

    labels = y.transpose(0, 1)
    labels = labels[1:].transpose(0, 1) # (b, t-1, num_word)

    state = init_state
    logits = []
    for t in range(t_steps-1):
      # print t
      input = y_embed[t]
      state = self.cell(input, state)
      if self._config.cell == 'gru':
        output = state
      elif self._config.cell == 'lstm':
        output = state[0]
      output = self.dropout(output)

      if self._config.tied:
        out = self.out(output)
        logit = self.word_dist(out)
      else:
        logit = self.word_dist(output)
      logits.append(logit)
    logits = torch.stack(logits).transpose(0, 1) # (b, t-1, num_word)
    log_softmaxs = F.log_softmax(logits, 2)

    loss = torch.sum(log_softmaxs * labels, 2)

    return loss, out_mask

  def greedy_decode(self, init_state):
    # type: (Any) -> torch.LongTensor
    """
    input:
      init_state: (b, dim_hidden) or pair of this shape for LSTM cell 
      lens: (b,)
    output:
      out_wids: (b, t)
    """
    if self._config.cell == 'gru':
      b = init_state.size(0)
    elif self._config.cell == 'lstm':
      b = init_state[0].size(0)
    wordids = torch.zeros(b, dtype=torch.long).cuda()

    state = init_state
    out_wids = []
    for t in range(self._config.max_step):
      embed = self.embed(wordids)
      
      input = embed
      state = self.cell(input, state)
      if self._config.cell == 'gru':
        output = state
      elif self._config.cell == 'lstm':
        output = state[0]

      if self._config.tied:
        out = self.out(output)
        logit = self.word_dist(out)
      else:
        logit = self.word_dist(output)
      wordids = torch.argmax(logit, 1)
      out_wids.append(wordids)
    out_wids = torch.stack(out_wids, 1) # (b, t)
    return out_wids

  def beam_decode(self, init_state):
    EOS = 1

    max_step = self._config.max_step
    topk = self._config.beam_width
    dim_hidden = self._config.dim_hidden
    dim_embed = self._config.dim_embed
    num_word = self._config.num_word

    if self._config.cell == 'gru':
      b = init_state.size(0)
    elif self._config.cell == 'lstm':
      b = init_state[0].size(0)
    wordids = torch.zeros(b, dtype=torch.long).cuda()

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
        state = self.cell(input, state)
        if self._config.cell == 'gru':
          output = state
        elif self._config.cell == 'lstm':
          output = state[0]

        if self._config.tied:
          out = self.out(output)
          logit = self.word_dist(out)
        else:
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

        # expand state
        if self._config.cell == 'gru':
          state = state.view(b, 1, dim_hidden).expand(b, topk, dim_hidden).contiguous().view(b*topk, dim_hidden)
        elif self._config.cell == 'lstm':
          state = [
            s.view(b, 1, dim_hidden).expand(b, topk, dim_hidden).contiguous().view(b*topk, dim_hidden) 
            for s in state
          ]
        wordids = idx.view(b*topk)
      else:
        embed = self.embed(wordids) # (b*topk, dim_embed)

        input = embed
        state = self.cell(input, state)
        if self._config.cell == 'gru':
          output = state
        elif self._config.cell == 'lstm':
          output = state[0]

        if self._config.tied:
          out = self.out(output)
          logit = self.word_dist(out)
        else:
          logit = self.word_dist(output) # (b, num_word)
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
        if self._config.cell == 'gru':
          state = torch.index_select(state, 0, pre_1d)
        elif self._config.cell == 'lstm':
          state = [torch.index_select(d, 0, pre_1d) for d in state]

    beam_cum_log_probs = torch.stack(beam_cum_log_probs, dim=1) # (b, max_step, topk)
    beam_pres = torch.stack(beam_pres, dim=1) # (b, max_step, topk)
    beam_ends = torch.stack(beam_ends, dim=1) # (b, max_step, topk)
    out_wids = torch.stack(out_wids, dim=1) # (b, max_step, topk)

    return beam_cum_log_probs, beam_pres, beam_ends, out_wids

  def sample_decode(self, init_state):
    max_step = self._config.max_step
    dim_hidden = self._config.dim_hidden
    num_word = self._config.num_word
    num_sample = self._config.num_sample

    if self._config.cell == 'gru':
      b = init_state.size(0)
      state = init_state.unsqueeze(1).expand(b, num_sample, dim_hidden)
    elif self._config.cell == 'lstm':
      b = init_state[0].size(0)
      state = [
        s.unsqueeze(1).expand(b, num_sample, dim_hidden).contiguous().view(b*num_sample, dim_hidden)
        for s in init_state
      ]
    wordids = torch.zeros(b*num_sample, dtype=torch.long).cuda(0)

    out_wids = []
    log_probs = []
    for t in range(max_step):
      embed = self.embed(wordids) # (b*num_sample, dim_embed)
      embed = self.dropin(embed)
      state = self.cell(embed, state)
      if self._config.cell == 'gru':
        output = state
      elif self._config.cell == 'lstm':
        output = state[0]
      output = self.dropout(output)

      if self._config.tied:
        out = self.out(output)
        logit = self.word_dist(out)
      else:
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

  def sample_topk_decode(self, init_state, topk):
    max_step = self._config.max_step
    dim_hidden = self._config.dim_hidden
    num_word = self._config.num_word
    num_sample = self._config.num_sample

    if self._config.cell == 'gru':
      b = init_state.size(0)
      state = init_state.unsqueeze(1).expand(b, num_sample, dim_hidden)
    elif self._config.cell == 'lstm':
      b = init_state[0].size(0)
      state = [
        s.unsqueeze(1).expand(b, num_sample, dim_hidden).contiguous().view(b*num_sample, dim_hidden)
        for s in init_state
      ]
    wordids = torch.zeros(b*num_sample, dtype=torch.long).cuda(0)

    out_wids = []
    log_probs = []
    for t in range(max_step):
      embed = self.embed(wordids) # (b*num_sample, dim_embed)
      embed = self.dropin(embed)
      state = self.cell(embed, state)
      if self._config.cell == 'gru':
        output = state
      elif self._config.cell == 'lstm':
        output = state[0]
      output = self.dropout(output)

      if self._config.tied:
        out = self.out(output)
        logit = self.word_dist(out)
      else:
        logit = self.word_dist(output) # (b*num_sample, num_word)

      topk_logit, topk_idxs = torch.topk(logit, topk, 1) # (b*num_sample, topk)
      categorical = torch.distributions.categorical.Categorical(logits=topk_logit)
      samples = categorical.sample()
      wordids = []
      for i in range(b*num_sample):
        wordids.append(topk_idxs[i, samples[i]])
      wordids = torch.stack(wordids)
      log_prob = -F.cross_entropy(logit, wordids, reduction='none')

      out_wids.append(wordids)
      log_probs.append(log_prob)
    out_wids = torch.stack(out_wids, 1) # (b*num_sample, t)
    log_probs = torch.stack(log_probs, 1) # (b*num_sample, t)
    out_wids = out_wids.contiguous().view(b, num_sample, max_step)
    log_probs = log_probs.contiguous().view(b, num_sample, max_step)

    # print out_wids.size()

    return out_wids, log_probs

  def nucleus_sample_decode(self, init_state, threshold_p):
    max_step = self._config.max_step
    dim_hidden = self._config.dim_hidden
    num_word = self._config.num_word
    num_sample = self._config.num_sample

    if self._config.cell == 'gru':
      b = init_state.size(0)
      state = init_state.unsqueeze(1).expand(b, num_sample, dim_hidden).contiguous().view(b*num_sample, dim_hidden)
    elif self._config.cell == 'lstm':
      b = init_state[0].size(0)
      state = [
        s.unsqueeze(1).expand(b, num_sample, dim_hidden).contiguous().view(b*num_sample, dim_hidden)
        for s in init_state
      ]
    wordids = torch.zeros(b*num_sample, dtype=torch.long).cuda(0)

    out_wids = []
    log_probs = []
    for t in range(max_step):
      embed = self.embed(wordids)
      state = self.cell(embed, state)
      if self._config.cell == 'gru':
        output = state
      elif self._config.cell == 'lstm':
        output = state[0]
      output = self.dropout(output)

      if self._config.tied:
        out = self.out(output)
        logit = self.word_dist(out)
      else:
        logit = self.word_dist(output) # (b*num_sample, num_word)
      p = F.softmax(logit, dim=1)
      p, idxs = torch.sort(p, dim=1, descending=True) # (b*num_sample, num_word)
      pcum = torch.cumsum(p, dim=1)
      mask = (pcum < threshold_p).float()
      sentinel = torch.zeros(mask.size()).cuda()
      sentinel[:, 0] = 1.
      mask = torch.max(mask, sentinel)
      p = p * mask

      categorical = torch.distributions.categorical.Categorical(p)
      samples = categorical.sample()
      wordids = []
      for i in range(b*num_sample): # map back to wid
        wordids.append(idxs[i, samples[i]])
      wordids = torch.stack(wordids)
      log_prob = -F.cross_entropy(logit, wordids, reduction='none')

      out_wids.append(wordids)
      log_probs.append(log_prob)
    out_wids = torch.stack(out_wids, 1) # (b*num_sample, t)
    log_probs = torch.stack(log_probs, 1) # (b*num_sample, t)
    out_wids = out_wids.contiguous().view(b, num_sample, max_step)
    log_probs = log_probs.contiguous().view(b, num_sample, max_step)

    return out_wids, log_probs
