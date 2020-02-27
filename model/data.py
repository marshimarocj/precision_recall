import sys
import os
import pickle
import random
sys.path.append('../')

import numpy as np
import torch
import torchvision
import cv2

from base import framework


class TrnValReader(framework.Reader):
  def __init__(self, ft_file, videoid_file, 
      shuffle=True, annotation_file=None, captionstr_file=None):
    self.fts = np.empty(0)
    self.ft_idxs = np.empty(0) # (num_caption,)
    self.captionids = []
    self.videoids = []
    self.shuffle = shuffle
    self.videoid2captions = {}

    self.shuffled_idxs = [] # (num_caption,)
    self.num_caption = 0 # used in trn and val
    self.num_ft = 0

    self.fts = np.load(ft_file)
    if len(self.fts.shape) > 2:
      self.fts = self.fts.squeeze(3).squeeze(2)
    self.num_ft = self.fts.shape[0]

    self.videoids = np.load(videoid_file)

    if annotation_file is not None:
      with open(annotation_file, 'rb') as f:
        self.ft_idxs, self.captionids = pickle.load(f)
      self.num_caption = len(self.ft_idxs)
    if captionstr_file is not None:
      with open(captionstr_file, 'rb') as f:
        videoid2captions = pickle.load(f)
      for videoid in self.videoids:
        self.videoid2captions[videoid] = videoid2captions[videoid]

    self.shuffled_idxs = range(self.num_caption)

  def reset(self):
    if self.shuffle:
      random.shuffle(self.shuffled_idxs)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      start = i
      end = i + batch_size
      idxs = self.shuffled_idxs[start:end]

      captionids = [self.captionids[idx] for idx in idxs]
      caption_lens = [len(captionid) for captionid in captionids]

      ft_idxs = [self.ft_idxs[idx] for idx in idxs]
      # print 'numpy', self.fts[ft_idxs].shape
      yield {
        'fts': self.fts[ft_idxs],
        'captionids': captionids,
        'caption_lens': caption_lens,
        'vids': self.videoids[ft_idxs],
      }


class TstReader(framework.Reader):
  def __init__(self, ft_file, videoid_file):
    self.fts = np.empty(0)
    self.videoids = []

    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]
    if len(self.fts.shape) > 2:
      self.fts = self.fts.squeeze(3).squeeze(2)

    self.videoids = np.load(videoid_file)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size

      yield {
        'fts': self.fts[start:end],
        'vids': self.videoids[start:end],
      }


class AttTstReader(framework.Reader):
  def __init__(self, ft_file, att_ft_file, videoid_file):
    self.fts = np.empty(0)
    self.att_fts = np.empty(0)
    self.videoids = []

    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]
    self.att_fts = np.load(att_ft_file)

    self.videoids = np.load(videoid_file)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size

      yield {
        'fts': self.fts[start:end],
        'att_fts': self.att_fts[start:end],
        'vids': self.videoids[start:end],
      }


class TrnDiscriminatorReader(framework.Reader):
  def __init__(self, ft_file, annotation_file):
    self.fts = np.empty(0)
    self.ft_idxs = []
    self.captionids = []
    self.ft_idx2captionids = {}

    self.num_ft = 0
    self.num_caption = 0
    self.shuffled_ft_idxs = []

    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]

    with open(annotation_file, 'rb') as f:
      self.ft_idxs, self.captionids = pickle.load(f)
    self.num_caption = len(self.ft_idxs)
    for ft_idx, captionid in zip(self.ft_idxs, self.captionids):
      if ft_idx not in self.ft_idx2captionids:
        self.ft_idx2captionids[ft_idx] = []
      self.ft_idx2captionids[ft_idx].append(captionid[1:-1]) # remove BOS and EOS

    self.shuffled_ft_idxs = self.ft_idx2captionids.keys()

  def reset(self):
    random.shuffle(self.shuffled_ft_idxs)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size
      ft_idxs = self.shuffled_ft_idxs[start:end]

      clean_ft_idxs = []
      pos_captionids = []
      neg_captionids = []
      for ft_idx in ft_idxs:
        captionids = self.ft_idx2captionids[ft_idx]
        num = len(captionids)
        if num < 5:
          continue
        pos_captionids += captionids[:5]

        cnt = 0
        while True and cnt < 5:
          r = random.randint(0, self.num_caption-1)
          if self.ft_idxs[r] != ft_idx:
            neg_captionids.append(self.captionids[r][1:-1])
            cnt += 1

        clean_ft_idxs.append(ft_idx)
      pos_lens = [len(d) for d in pos_captionids]
      neg_lens = [len(d) for d in neg_captionids]
      fts = self.fts[clean_ft_idxs]

      yield {
        'fts': fts,
        'pos_captionids': pos_captionids,
        'pos_caption_lens': pos_lens,
        'neg_captionids': neg_captionids,
        'neg_caption_lens': neg_lens,
      }


class TrnImgReader(framework.Reader):
  def __init__(self, img_dir, lst_file, annotation_file):
    self.ft_idxs = []
    self.captionids = []
    self.ft_idx2captionids = {}
    self.img_names = []
    self.img_dir = img_dir
    self.num_ft = 0

    self.transform = torchvision.transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    self.img_names = np.load(lst_file)
    self.num_ft = len(self.img_names)

    with open(annotation_file, 'rb') as f:
      self.ft_idxs, self.captionids = pickle.load(f)
    self.num_caption = len(self.ft_idxs)
    for ft_idx, captionid in zip(self.ft_idxs, self.captionids):
      if ft_idx not in self.ft_idx2captionids:
        self.ft_idx2captionids[ft_idx] = []
      self.ft_idx2captionids[ft_idx].append(captionid[1:-1]) # remove BOS and EOS

    self.shuffled_ft_idxs = self.ft_idx2captionids.keys()

  def reset(self):
    random.shuffle(self.shuffled_ft_idxs)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size
      ft_idxs = self.shuffled_ft_idxs[start:end]

      clean_ft_idxs = []
      pos_captionids = []
      neg_captionids = []
      for ft_idx in ft_idxs:
        captionids = self.ft_idx2captionids[ft_idx]
        num = len(captionids)
        if num < 5:
          continue
        pos_captionids += captionids[:5]

        cnt = 0
        while True and cnt < 5:
          r = random.randint(0, self.num_caption-1)
          if self.ft_idxs[r] != ft_idx:
            neg_captionids.append(self.captionids[r][1:-1])
            cnt += 1

        clean_ft_idxs.append(ft_idx)
      pos_lens = [len(d) for d in pos_captionids]
      neg_lens = [len(d) for d in neg_captionids]

      imgs = torch.zeros((len(clean_ft_idxs), 3, 450, 450))
      masks = torch.zeros((len(clean_ft_idxs), 15, 15))
      for j, ft_idx in enumerate(clean_ft_idxs):
        img_file = os.path.join(self.img_dir, self.img_names[ft_idx])
        img, h, w = load_and_norm_img(img_file, self.transform)
        imgs[j, :, :h, :w] = img
        if h > w:
          rh = 15
          rw = w * rh / h
        else:
          rw = 15
          rh = h * rw / w
        masks[j, :rh, :rw] = 1.
      
      yield {
        'imgs': imgs,
        'masks': masks,
        'pos_captionids': pos_captionids,
        'pos_caption_lens': pos_lens,
        'neg_captionids': neg_captionids,
        'neg_caption_lens': neg_lens,
      }


class ValDiscriminatorReader(framework.Reader):
  def __init__(self, ft_file, annotation_file):
    self.fts = np.empty(0)
    self.ft_idx2pos_captionids = {}
    self.ft_idx2neg_captionids = {}

    self.num_ft = 0
    
    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]

    with open(annotation_file, 'rb') as f:
      ft_idxs, captionids = pickle.load(f)
    for ft_idx, captionid in zip(ft_idxs, captionids):
      if ft_idx not in self.ft_idx2pos_captionids:
        self.ft_idx2pos_captionids[ft_idx] = []
      self.ft_idx2pos_captionids[ft_idx].append(captionid[1:-1])

    num_caption = len(captionids)
    for ft_idx in self.ft_idx2pos_captionids:
      pos_captionids = self.ft_idx2pos_captionids[ft_idx]
      num = len(pos_captionids)
      neg_captionids = []
      cnt = 0
      while True and cnt < num:
        r = random.randint(0, num_caption-1)
        if ft_idxs[r] != ft_idx:
          neg_captionids.append(captionids[r][1:-1])
          cnt += 1
      self.ft_idx2neg_captionids[ft_idx] = neg_captionids
  
  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = min(i + batch_size, self.num_ft)

      clean_ft_idxs = []
      pos_captionids = []
      neg_captionids = []
      for ft_idx in range(start, end):
        captionids = self.ft_idx2pos_captionids[ft_idx]
        if len(captionids) < 5:
          continue
        pos_captionids += captionids[:5]

        captionids = self.ft_idx2neg_captionids[ft_idx]
        neg_captionids += captionids[:5]

        clean_ft_idxs.append(ft_idx)
      pos_lens = [len(d) for d in pos_captionids]
      neg_lens = [len(d) for d in neg_captionids]
      fts = self.fts[clean_ft_idxs]

      yield {
        'fts': fts,
        'pos_captionids': pos_captionids,
        'pos_caption_lens': pos_lens,
        'neg_captionids': neg_captionids,
        'neg_caption_lens': neg_lens,
      }


class ValImgReader(framework.Reader):
  def __init__(self, img_dir, lst_file, annotation_file):
    self.ft_idx2pos_captionids = {}
    self.ft_idx2neg_captionids = {}
    self.img_dir = img_dir
    self.img_names = []
    self.num_ft = 0

    self.transform = torchvision.transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    self.img_names = np.load(lst_file)
    self.num_ft = len(self.img_names)
    # print lst_file, self.num_ft

    with open(annotation_file, 'rb') as f:
      ft_idxs, captionids = pickle.load(f)
    for ft_idx, captionid in zip(ft_idxs, captionids):
      if ft_idx not in self.ft_idx2pos_captionids:
        self.ft_idx2pos_captionids[ft_idx] = []
      self.ft_idx2pos_captionids[ft_idx].append(captionid[1:-1])

    num_caption = len(captionids)
    for ft_idx in self.ft_idx2pos_captionids:
      pos_captionids = self.ft_idx2pos_captionids[ft_idx]
      num = len(pos_captionids)
      neg_captionids = []
      cnt = 0
      while True and cnt < num:
        r = random.randint(0, num_caption-1)
        if ft_idxs[r] != ft_idx:
          neg_captionids.append(captionids[r][1:-1])
          cnt += 1
      self.ft_idx2neg_captionids[ft_idx] = neg_captionids

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = min(i + batch_size, self.num_ft)

      clean_ft_idxs = []
      pos_captionids = []
      neg_captionids = []
      for ft_idx in range(start, end):
        captionids = self.ft_idx2pos_captionids[ft_idx]
        if len(captionids) < 5:
          continue
        pos_captionids += captionids[:5]

        captionids = self.ft_idx2neg_captionids[ft_idx]
        neg_captionids += captionids[:5]

        clean_ft_idxs.append(ft_idx)
      pos_lens = [len(d) for d in pos_captionids]
      neg_lens = [len(d) for d in neg_captionids]

      imgs = torch.zeros((len(clean_ft_idxs), 3, 450, 450))
      masks = torch.zeros((len(clean_ft_idxs), 15, 15))
      for j, ft_idx in enumerate(clean_ft_idxs):
        img_file = os.path.join(self.img_dir, self.img_names[ft_idx])
        img, h, w = load_and_norm_img(img_file, self.transform)
        imgs[j, :, :h, :w] = img
        if h > w:
          rh = 15
          rw = w * rh / h
        else:
          rw = 15
          rh = h * rw / w
        masks[j, :rh, :rw] = 1.
      
      yield {
        'imgs': imgs,
        'masks': masks,
        'pos_captionids': pos_captionids,
        'pos_caption_lens': pos_lens,
        'neg_captionids': neg_captionids,
        'neg_caption_lens': neg_lens,
      }


class TrnGanReader(framework.Reader):
  def __init__(self, ft_file, videoid_file, annotation_file):
    self.fts = np.empty(0)
    self.vids = []
    self.ft_idxs = []
    self.captionids = []
    self.ft_idx2captionids = {}

    self.num_ft = 0
    self.num_caption = 0
    self.shuffled_ft_idxs = []

    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]

    self.vids = np.load(videoid_file)

    with open(annotation_file, 'rb') as f:
      self.ft_idxs, self.captionids = pickle.load(f)
    self.num_caption = len(self.ft_idxs)
    for ft_idx, captionid in zip(self.ft_idxs, self.captionids):
      if ft_idx not in self.ft_idx2captionids:
        self.ft_idx2captionids[ft_idx] = []
      self.ft_idx2captionids[ft_idx].append(captionid[1:-1]) # remove BOS and EOS

    self.shuffled_ft_idxs = self.ft_idx2captionids.keys()

  def reset(self):
    random.shuffle(self.shuffled_ft_idxs)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size
      ft_idxs = self.shuffled_ft_idxs[start:end]

      clean_ft_idxs = []
      pos_captionids = []
      neg_captionids = []
      for ft_idx in ft_idxs:
        captionids = self.ft_idx2captionids[ft_idx]
        num = len(captionids)
        if num < 5:
          continue
        pos_captionids += captionids[:5]

        cnt = 0
        while True and cnt < 5:
          r = random.randint(0, self.num_caption-1)
          if self.ft_idxs[r] != ft_idx:
            neg_captionids.append(self.captionids[r][1:-1])
            cnt += 1

        clean_ft_idxs.append(ft_idx)
      pos_lens = [len(d) for d in pos_captionids] # b*5
      neg_lens = [len(d) for d in neg_captionids] # b*5
      fts = self.fts[clean_ft_idxs] # b
      vids = self.vids[clean_ft_idxs]

      yield {
        'vids': vids,
        'fts': fts,
        'pos_captionids': pos_captionids,
        'pos_lens': pos_lens,
        'neg_captionids': neg_captionids, 
        'neg_lens': neg_lens,
      }


class AttTrnGanReader(framework.Reader):
  def __init__(self, ft_file, att_ft_file, videoid_file, annotation_file):
    self.fts = np.empty(0)
    self.att_fts = np.empty(0)
    self.vids = []
    self.ft_idxs = []
    self.captionids = []
    self.ft_idx2captionids = {}

    self.num_ft = 0
    self.num_caption = 0
    self.shuffled_ft_idxs = []

    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]
    self.att_fts = np.load(att_ft_file)

    self.vids = np.load(videoid_file)

    with open(annotation_file, 'rb') as f:
      self.ft_idxs, self.captionids = pickle.load(f)
    self.num_caption = len(self.ft_idxs)
    for ft_idx, captionid in zip(self.ft_idxs, self.captionids):
      if ft_idx not in self.ft_idx2captionids:
        self.ft_idx2captionids[ft_idx] = []
      self.ft_idx2captionids[ft_idx].append(captionid[1:-1]) # remove BOS and EOS

    self.shuffled_ft_idxs = self.ft_idx2captionids.keys()

  def reset(self):
    random.shuffle(self.shuffled_ft_idxs)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size
      ft_idxs = self.shuffled_ft_idxs[start:end]

      clean_ft_idxs = []
      pos_captionids = []
      neg_captionids = []
      for ft_idx in ft_idxs:
        captionids = self.ft_idx2captionids[ft_idx]
        num = len(captionids)
        if num < 5:
          continue
        pos_captionids += captionids[:5]

        cnt = 0
        while True and cnt < 5:
          r = random.randint(0, self.num_caption-1)
          if self.ft_idxs[r] != ft_idx:
            neg_captionids.append(self.captionids[r][1:-1])
            cnt += 1

        clean_ft_idxs.append(ft_idx)
      pos_lens = [len(d) for d in pos_captionids] # b*5
      neg_lens = [len(d) for d in neg_captionids] # b*5
      fts = self.fts[clean_ft_idxs] # b
      att_fts = self.att_fts[clean_ft_idxs]
      vids = self.vids[clean_ft_idxs]

      yield {
        'vids': vids,
        'fts': fts,
        'att_fts': att_fts,
        'pos_captionids': pos_captionids,
        'pos_lens': pos_lens,
        'neg_captionids': neg_captionids, 
        'neg_lens': neg_lens,
      }


class ValGanReader(framework.Reader):
  def __init__(self, ft_file, videoid_file, captionstr_file):
    self.fts = np.empty(0)
    self.vids = []
    self.vid2captions = {}

    self.num_ft = 0

    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]

    self.vids = np.load(videoid_file)

    with open(captionstr_file) as f:
      vid2captions = pickle.load(f)
    for vid in self.vids:
      self.vid2captions[vid] = vid2captions[vid]

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size
      
      yield {
        'fts': self.fts[start:end],
        'vids': self.vids[start:end],
      }


class AttValGanReader(framework.Reader):
  def __init__(self, ft_file, att_ft_file, videoid_file, captionstr_file):
    self.fts = np.empty(0)
    self.att_fts = np.empty(0)
    self.vids = []
    self.vid2captions = {}

    self.num_ft = 0

    self.fts = np.load(ft_file)
    self.num_ft = self.fts.shape[0]
    self.att_fts = np.load(att_ft_file)

    self.vids = np.load(videoid_file)

    with open(captionstr_file) as f:
      vid2captions = pickle.load(f)
    for vid in self.vids:
      self.vid2captions[vid] = vid2captions[vid]

  def yield_batch(self, batch_size):
    for i in range(0, self.num_ft, batch_size):
      start = i
      end = i + batch_size
      
      yield {
        'fts': self.fts[start:end],
        'att_fts': self.att_fts[start:end],
        'vids': self.vids[start:end],
      }


class TrnSimpleGanReader(framework.Reader):
  def __init__(self, ft_file, videoid_file, annotation_file):
    self.fts = np.empty(0)
    self.vids = []
    self.ft_idxs = []
    self.captionids = []

    self.num_caption = 0
    self.shuffled_idxs = []

    self.fts = np.load(ft_file)

    self.vids = np.load(videoid_file)
    
    with open(annotation_file, 'rb') as f:
      self.ft_idxs, self.captionids = pickle.load(f)
    self.num_caption = len(self.ft_idxs)

    self.shuffled_idxs = range(self.num_caption)

  def reset(self):
    random.shuffle(self.shuffled_idxs)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      start = i
      end = i + batch_size
      idxs = self.shuffled_idxs[start:end]

      pos_captionids = []
      neg_captionids = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        pos_captionids.append(self.captionids[idx][1:-1]) # exclude BOS and EOS

        cnt = 0
        while True and cnt < 1:
          r = random.randint(0, self.num_caption-1)
          if self.ft_idxs[r] != ft_idx:
            neg_captionids.append(self.captionids[r][1:-1]) # exclude BOS and EOS
            cnt += 1
      pos_lens = [len(d) for d in pos_captionids]
      neg_lens = [len(d) for d in neg_captionids]
      ft_idxs = [self.ft_idxs[d] for d in idxs]
      fts = self.fts[ft_idxs]
      vids = self.vids[ft_idxs]

      yield {
        'vids': vids,
        'fts': fts,
        'pos_captionids': pos_captionids,
        'pos_lens': pos_lens,
        'neg_captionids': neg_captionids,
        'neg_lens': neg_lens,
      }


class AttTrnSimpleGanReader(framework.Reader):
  def __init__(self, ft_file, att_ft_file, videoid_file, annotation_file):
    self.fts = np.empty(0)
    self.att_fts = np.empty(0)
    self.vids = []
    self.ft_idxs = []
    self.captionids = []

    self.num_caption = 0
    self.shuffled_idxs = []

    self.fts = np.load(ft_file)
    self.att_fts = np.load(att_ft_file)

    self.vids = np.load(videoid_file)
    
    with open(annotation_file, 'rb') as f:
      self.ft_idxs, self.captionids = pickle.load(f)
    self.num_caption = len(self.ft_idxs)

    self.shuffled_idxs = range(self.num_caption)

  def reset(self):
    random.shuffle(self.shuffled_idxs)

  def yield_batch(self, batch_size):
    for i in range(0, self.num_caption, batch_size):
      start = i
      end = i + batch_size
      idxs = self.shuffled_idxs[start:end]

      pos_captionids = []
      neg_captionids = []
      for idx in idxs:
        ft_idx = self.ft_idxs[idx]
        pos_captionids.append(self.captionids[idx][1:-1]) # exclude BOS and EOS

        cnt = 0
        while True and cnt < 1:
          r = random.randint(0, self.num_caption-1)
          if self.ft_idxs[r] != ft_idx:
            neg_captionids.append(self.captionids[r][1:-1]) # exclude BOS and EOS
            cnt += 1
      pos_lens = [len(d) for d in pos_captionids]
      neg_lens = [len(d) for d in neg_captionids]
      ft_idxs = [self.ft_idxs[d] for d in idxs]
      fts = self.fts[ft_idxs]
      att_fts = self.att_fts[ft_idxs]
      vids = self.vids[ft_idxs]

      yield {
        'vids': vids,
        'fts': fts,
        'att_fts': att_fts,
        'pos_captionids': pos_captionids,
        'pos_lens': pos_lens,
        'neg_captionids': neg_captionids,
        'neg_lens': neg_lens,
      }


def load_and_norm_img(img_file, transform):
  img = cv2.imread(img_file)
  h, w, _ = img.shape
  if max(h, w) > 450:
    img = norm_img(img)
  img = img[:, :, ::-1] # change to RGB
  img = img / 255. # scale to [0, 1]
  h, w, _ = img.shape
  img = np.moveaxis(img, [0, 1, 2], [1, 2, 0]) # c, h, w
  img = torch.from_numpy(img)
  img = transform(img)

  return img, h, w
