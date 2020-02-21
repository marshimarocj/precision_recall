import argparse
import sys
import os
import cPickle
sys.path.append('../')

import numpy as np

import model.vevd_ml
import model.data
import common


def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file')
  parser.add_argument('path_cfg_file')
  parser.add_argument('--is_train', dest='is_train', type=int, default=True)
  parser.add_argument('--gpuids', dest='gpuids', default='0')
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=True)
  parser.add_argument('--strategy', dest='strategy', default='beam')
  parser.add_argument('--beam_width', dest='beam_width', type=int, default=5)
  parser.add_argument('--pool_size', dest='pool_size', type=int, default=5)
  parser.add_argument('--num_sample', dest='num_sample', type=int, default=5)
  parser.add_argument('--sample_topk', dest='sample_topk', type=int, default=5)
  parser.add_argument('--threshold_p', dest='threshold_p', type=float, default=.8)

  return parser


def gen_dir_struct_info(path_cfg_file):
  path_cfg = model.vevd_ml.PathCfg()
  return common.gen_caption_dir_struct_info(path_cfg, path_cfg_file)


def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = model.vevd_ml.ModelConfig()
  model_cfg.load(model_cfg_file)

  # auto fill params
  words = cPickle.load(open(path_cfg.word_file))
  model_cfg.subcfgs[model.vevd_ml.DEC].num_word = len(words)

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  gpuids = opts.gpuids.split(',')
  gpuids = [int(d) for d in gpuids]

  path_cfg = gen_dir_struct_info(opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  m = model.vevd_ml.Model(model_cfg)

  path_cfg.model_file = os.path.join(path_cfg.model_dir, 'epoch-%d.pth'%opts.best_epoch)
  model_cfg.subcfgs[model.vevd_ml.DEC].beam_width = opts.beam_width
  model_cfg.strategy = opts.strategy
  if model_cfg.strategy == 'beam':
    model_cfg.pool_size = opts.pool_size
    model_cfg.subcfgs[model.vevd_ml.DEC].beam_width = opts.beam_width
    path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred', 
      '%d-%s-%d-%d.json'%(opts.best_epoch, model_cfg.strategy, opts.beam_width, opts.pool_size))
  elif model_cfg.strategy == 'sample_topk':
    model_cfg.subcfgs[model.vevd_ml.DEC].num_sample = opts.num_sample
    model_cfg.sample_topk = opts.sample_topk
    path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred',
      '%d-%s-%d-%d.json'%(opts.best_epoch, model_cfg.strategy, opts.sample_topk, opts.num_sample))
  elif model_cfg.strategy == 'nucleus_sample':
    model_cfg.subcfgs[model.vevd_ml.DEC].num_sample = opts.num_sample
    model_cfg.threshold_p = opts.threshold_p
    path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred',
      '%d-%s-%.2f-%d.json'%(opts.best_epoch, model_cfg.strategy, opts.threshold_p, opts.num_sample))
  path_cfg.log_file = ''

  trntst = model.vevd_ml.TrnTstDecode(model_cfg, path_cfg, gpuids)
  tst_reader = model.data.TstReader(path_cfg.tst_ftfile, path_cfg.tst_videoid_file)
  trntst.test(m, tst_reader)
