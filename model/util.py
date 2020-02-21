import os
# import cPickle
import pickle
import sys
sys.path.append('../')

import numpy as np

from base import framework

EOS = 1

class CaptionInt2str(object):
  def __init__(self, int2word_file):
    self.int2word = []

    with open(int2word_file, 'rb') as f:
      self.int2word = pickle.load(f)

  # captionInt should be a batch of captionInts
  def __call__(self, captionid):
    batch_size = captionid.shape[0]

    captionStr = []
    for i in range(batch_size):
      sent = []
      for t in captionid[i]:
        if t == EOS:
          break
        else:
          sent.append(self.int2word[t])
      captionStr.append(' '.join(sent))

    return captionStr


def beamsearch_recover_captions(wordids, cum_log_probs, pres, ends):
  """
  wordids, cum_log_probs, pres, ends: (b, t, topk)
  """
  b = wordids.shape[0]
  t = wordids.shape[1]

  candidates = [[] for _ in range(b)]
  for step in range(t):
    end = ends[:, step] # (b, topk)
    for b, b_end in enumerate(end):
      for i, d in enumerate(b_end):
        if d == 0:
          continue

        score = cum_log_probs[b, step, i] / (step+1)

        sent = []
        pre = i
        for step1 in range(step, -1, -1):
          sent.append(wordids[b, step1, pre])
          pre = pres[b, step1, pre]

        candidates[b].append({
          'sent': sent[::-1],
          'score': score,
        })
      if step == t-1 and len(candidates[b]) == 0: # in case no EOS appear in beam search results
        for i, d in enumerate(b_end):
          score = cum_log_probs[b, step, i] / (step+1)

          sent = [EOS] # manually append EOS
          pre = i
          for step1 in range(step, -1, -1):
            sent.append(wordids[b, step1, pre])
            pre = pres[b, step1, pre]

          candidates[b].append({
            'sent': sent[::-1],
            'score': score,
          })

  best_candidates = []
  for candidate in candidates:
    candidate = sorted(candidate, key=lambda x: x['score'], reverse=True)
    # print len(candidate)
    # print candidate
    best_candidates.append(candidate[0]['sent'])

  return best_candidates


def beamsearch_recover_multiple_captions(wordids, cum_log_probs, pres, ends, pool_size):
  """
  wordids, cum_log_probs, pres, ends: (b, t, topk)
  """
  b = wordids.shape[0]
  t = wordids.shape[1]

  candidates = [[] for _ in range(b)]
  for step in range(t):
    end = ends[:, step] # (b, topk)
    for b, b_end in enumerate(end):
      for i, d in enumerate(b_end):
        if d == 0:
          continue

        score = cum_log_probs[b, step, i] / (step+1)

        sent = []
        pre = i
        for step1 in range(step, -1, -1):
          sent.append(wordids[b, step1, pre])
          pre = pres[b, step1, pre]

        candidates[b].append({
          'sent': sent[::-1],
          'score': score,
        })
      if step == t-1 and len(candidates[b]) == 0: # in case no EOS appear in beam search results
        for i, d in enumerate(b_end):
          score = cum_log_probs[b, step, i] / (step+1)

          sent = [EOS] # manually append EOS
          pre = i
          for step1 in range(step, -1, -1):
            sent.append(wordids[b, step1, pre])
            pre = pres[b, step1, pre]

          candidates[b].append({
            'sent': sent[::-1],
            'score': score,
          })

  best_candidates = []
  for candidate in candidates:
    candidate = sorted(candidate, key=lambda x: x['score'], reverse=True)
    candidate = candidate[:pool_size]
    best_candidates.append([d for d in candidate])

  return best_candidates


def recover_sample_caption_scores(wordids, log_probs, num_sample):
  EOS = 1
  b, num_sample, t = wordids.shape

  candidates = [[] for _ in range(b)]
  for i in range(b):
    for j in range(num_sample):
      cnt = 0
      sent = []
      for wid in wordids[i, j]:
        cnt += 1
        sent.append(wid)

        if wid == EOS:
          break
      score = np.mean(log_probs[i, j, :cnt])
      candidates[i].append({
        'sent': sent,
        'score': score,
      })

  out = []
  for candidate in candidates:
    candidate = sorted(candidate, key=lambda x: x['score'], reverse=True)
    out.append(candidate)
  return out


class PathCfg(framework.PathCfg):
  def __init__(self):
    super(PathCfg, self).__init__()

    # manually provided in the cfg file
    self.annotation_dir = ''
    self.output_dir = ''
    self.trn_ftfile = ''
    self.val_ftfile = ''
    self.tst_ftfile = ''
    self.trn_annotation_file = ''
    self.val_annotation_file = ''
    self.word_file = ''
    self.groundtruth_file = ''

    # automatically generated paths
    self.trn_videoid_file = ''
    self.val_videoid_file = ''
    self.tst_videoid_file = ''


class RLPathCfg(framework.PathCfg):
  def __init__(self):
    super(RLPathCfg, self).__init__()

    # manually provided in the cfg file
    self.annotation_dir = ''
    self.output_dir = ''
    self.trn_ftfile = ''
    self.val_ftfile = ''
    self.tst_ftfile = ''
    self.trn_annotation_file = ''
    self.val_annotation_file = ''
    self.word_file = ''
    self.groundtruth_file = ''
    self.df_file = ''

    # automatically generated paths
    self.trn_videoid_file = ''
    self.val_videoid_file = ''
    self.tst_videoid_file = ''


def transform_predict_captionid_array_to_list(captionids): # (None, step)
  outs = []
  for captionid in captionids:
    out = [0]
    for wid in captionid:
      out.append(wid)
      if wid == 1:
        break
    outs.append(out)

  return outs


def eval_cider_in_rollout(out_wids, vids, int2str, cider):
  batch_size, num_sample, num_step = out_wids.shape
  out_scores = []
  for i in range(batch_size):
    pred_captions = []
    for j in range(num_sample):
      pred_caption = int2str(np.expand_dims(out_wids[i, j], 0))[0]
      pred_captions.append(pred_caption)
    _vids = [vids[i]]*num_sample
    score, scores = cider.compute_cider(pred_captions, _vids)
    out_scores.append(scores)

  out_scores = np.array(out_scores, dtype=np.float32)
  return out_scores


# pad to #(t, b, max_len)
def pad_sequence(seq):
  b = len(seq)
  max_len = max([len(d) for d in seq])
  y = np.zeros((max_len, b), dtype=np.long)
  for i, s in enumerate(seq):
    for j, d in enumerate(s):
      y[j, i] = d
  return y
