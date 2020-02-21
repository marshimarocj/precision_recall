import os
import json
import sys
import argparse
import cPickle
sys.path.append('../')

import model.discriminator
import model.data
import common

def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--gpuids', dest='gpuids', default='0')

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = model.discriminator.PathCfg()
  return common.gen_caption_dir_struct_info(path_cfg, path_cfg_file)


def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = model.discriminator.ModelConfig()
  model_cfg.load(model_cfg_file)

  # auto fill params
  words = cPickle.load(open(path_cfg.word_file))
  model_cfg.subcfgs[model.discriminator.SE].num_word = len(words)

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  gpuids = opts.gpuids.split(',')
  gpuids = [int(d) for d in gpuids]

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  m = model.discriminator.Model(model_cfg)

  if opts.is_train:
    trntst = model.discriminator.PretrainTrnTst(model_cfg, path_cfg, gpuids)
    trn_reader = model.data.TrnDiscriminatorReader(path_cfg.trn_ftfile, path_cfg.trn_annotation_file)
    val_reader = model.data.ValDiscriminatorReader(path_cfg.val_ftfile, path_cfg.val_annotation_file)
    trntst.train(m, trn_reader, val_reader)
