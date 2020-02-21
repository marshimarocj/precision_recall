import argparse
import sys
import os
import cPickle
sys.path.append('../')

import numpy as np

import model.gan_sc
import model.data
import common

def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  # only in tst
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=True)
  parser.add_argument('--strategy', dest='strategy', default='beam')
  parser.add_argument('--beam_width', dest='beam_width', type=int, default=5)

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = model.gan_sc.PathCfg()
  return common.gen_caption_dir_struct_info(path_cfg, path_cfg_file)


def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = model.gan_sc.ModelConfig()
  model_cfg.load(model_cfg_file)

  # auto fill params
  words = cPickle.load(open(path_cfg.word_file))
  model_cfg.subcfgs[model.gan_sc.DEC].num_word = len(words)

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  m = model.gan_sc.Model(model_cfg)

  if opts.is_train:
    trntst = model.gan_sc.PretrainDTrnTst(model_cfg, path_cfg, [0])
    trn_reader = model.data.TrnGanReader(path_cfg.trn_ftfile, path_cfg.trn_annotation_file)
    tst_reader = model.data.ValGanReader(path_cfg.val_ftfile, path_cfg.val_videoid_file, path_cfg.groundtruth_file)
    trntst.train(m, trn_reader, tst_reader)
