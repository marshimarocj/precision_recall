from __future__ import division

import json
import logging
import os
import datetime
import pprint
import collections
import random

import numpy as np

import torch
import torch.nn as nn


class ModuleConfig(object):
  """
  config of a module
  in addition to the customized parameters in the config, it contains three special attributes:
  [subcfgs] a dictionary of configs belong to the submodules in this module
  """
  def __init__(self):
    self.subcfgs = {}

  def load(self, cfg_dict):
    for key in cfg_dict:
      if key == 'subcfgs': # recursion
        data = cfg_dict[key]
        for key in data:
          self.subcfgs[key].load(data[key])
      elif key in self.__dict__:
        setattr(self, key, cfg_dict[key])

    self._assert()

  def save(self):
    out = {}
    for attr in self.__dict__:
      if attr == 'subcfgs':
        cfgs = self.__dict__[attr]
        out['subcfgs'] = {}
        for key in cfgs:
          out['subcfgs'][key] = cfgs[key].save()
      else:
        val = self.__dict__[attr]
        if type(val) is not np.ndarray: # ignore nparray fields, which are used to initialize weights
          out[attr] = self.__dict__[attr]
    return out

  def _assert(self):
    """
    check compatibility between configs
    """
    pass


class ModelConfig(ModuleConfig):
  def __init__(self):
    ModuleConfig.__init__(self)

    self.trn_batch_size = 256
    self.tst_batch_size = 128
    self.num_epoch = 100
    self.val_iter = 100
    self.val_loss = True
    self.monitor_iter = -1
    self.base_lr = 1e-4

  def load(self, file):
    with open(file) as f:
      data = json.load(f)
      ModuleConfig.load(self, data)

  def save(self, out_file):
    out = ModuleConfig.save(self)
    with open(out_file, 'w') as fout:
      json.dump(out, fout, indent=2)


class GanModelConfig(ModuleConfig):
  def __init__(self):
    super(GanModelConfig, self).__init__()

    self.trn_batch_size = 256
    self.tst_batch_size = 128
    self.val_iter = 100
    self.val_loss = True
    self.monitor_iter = -1

    self.g_base_lr = 1e-4
    self.g_num_epoch = 50
    self.g_freeze = False
    self.g_freeze_epoch = 1

    self.d_base_lr = 1e-4
    self.d_num_epoch = 5
    self.d_iter = -1 # -1 means not update
    self.d_val_acc = 0. # the target validation performance when training discriminator
    self.d_exit_acc = .65
    self.d_buffer_size = 5

  def load(self, file):
    with open(file) as f:
      data = json.load(f)
      ModuleConfig.load(self, data)

  def save(self, out_file):
    out = ModuleConfig.save(self)
    with open(out_file, 'w') as fout:
      json.dump(out, fout, indent=2)


class PathCfg(object):
  def __init__(self):
    self.output_dir = ''
    self.log_dir = ''
    self.model_dir = ''

    self.log_file = ''
    self.model_file = ''
    self.predict_file = ''

  def load(self, file):
    data = json.load(open(file))
    for key in data:
      setattr(self, key, data[key])


class TrnTst(object):
  def __init__(self, model_cfg, path_cfg, gpuids):
    self.model_cfg = model_cfg
    self.path_cfg = path_cfg
    self.gpuids = gpuids

    # trn & tst
    self.tst_reader = None
    self.model = None

    # trn only
    self.scheduler = None
    self.trn_reader = None
    self.loss = None
    self.logger = None

  # return loss
  def feed_data_forward_backward(self, data):
    raise NotImplementedError

  # return metrics, which is a dictionary
  def validation(self):
    raise NotImplementedError

  def predict_in_tst(self):
    raise NotImplementedError

  def change_lr(self, history, metrics):
    pass

  def train(self, m, trn_reader, tst_reader, scheduler=None):
    self.model = m
    self.trn_reader = trn_reader
    self.tst_reader = tst_reader
    self.scheduler = scheduler

    self.logger = set_logger('Trn', log_path=self.path_cfg.log_file)

    base_epoch = self._load_and_place_device()

    self.model.eval()
    metrics = self.validation()
    self.logger.info('epoch %d', base_epoch)
    for key in metrics:
      self.logger.info('%s:%f', key, metrics[key])

    step = 0
    history = []
    for epoch in range(base_epoch, self.model_cfg.num_epoch):
      history = self.change_lr(history, metrics)

      step, avg_loss = self._iterate_epoch(step, epoch)

      self.model.eval()
      metrics = self.validation()
      metrics['train_loss'] = float(avg_loss)
      metrics['epoch'] = epoch
      self.logger.info('epoch %d', epoch)
      for key in metrics:
        self.logger.info('%s:%f', key, metrics[key])
      val_file = os.path.join(self.path_cfg.log_dir, 'val_metrics.%d.json'%epoch)
      with open(val_file, 'w') as fout:
        json.dump(metrics, fout, indent=2)

  def test(self, m, tst_reader):
    self.model = m
    self.tst_reader = tst_reader

    self._load_and_place_device()

    self.model.eval()
    self.predict_in_tst()

  def _load_and_place_device(self):
    if self.path_cfg.model_file != '':
      data = torch.load(self.path_cfg.model_file)
      self.model.load_state_dict(data['state_dict'], strict=False)
      model = self.model
      if len(self.gpuids) > 1:
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpuids)
        self.model.cuda()
      else:
        self.model.cuda(self.gpuids[0])
      self.optimizer = torch.optim.Adam(model.trainable_params(), lr=self.model_cfg.base_lr)
      if data['optimizer'] is not None:
        self.optimizer.load_state_dict(data['optimizer'])
        for state in self.optimizer.state.values():
          for k, v in state.items():
            if torch.is_tensor(v):
              state[k] = v.cuda(self.gpuids[0])
      if data['epoch'] is not None:
        base_epoch = data['epoch'] + 1
      else:
        base_epoch = 0
    else:
      model = self.model
      if len(self.gpuids) > 1:
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpuids)
        self.model.cuda()
      else:
        self.model.cuda(self.gpuids[0])
      self.optimizer = torch.optim.Adam(model.trainable_params(), lr=self.model_cfg.base_lr)
      base_epoch = 0

    return base_epoch

  def _iterate_epoch(self, step, epoch):
    avg_loss = 0.
    cnt = 0
    self.trn_reader.reset()
    for data in self.trn_reader.yield_batch(self.model_cfg.trn_batch_size):
      self.model.train()
      self.optimizer.zero_grad()
      loss = self.feed_data_forward_backward(data)
      self.optimizer.step()

      step += 1
      avg_loss += loss.data.cpu().numpy()
      cnt += 1

      if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
        self.logger.info('(step %d) loss: %f', step, loss.data.cpu().numpy())
        model = self.model
        if isinstance(model, nn.DataParallel):
          model = model.module
        for name in model.op2monitor:
          val = model.op2monitor[name].data.cpu().numpy()
          self.logger.info('(step %d) monitor "%s":%s', step, name, pprint.pformat(val))

      if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
        self.model.eval()
        metrics = self.validation()
        self.logger.info('step %d', step)
        for key in metrics:
          self.logger.info('%s:%f', key, metrics[key])

    model_file = os.path.join(self.path_cfg.model_dir, 'epoch-%d.pth'%epoch)
    save_checkpoint(self.model, self.optimizer, epoch, model_file)

    avg_loss /= cnt

    return step, avg_loss


class GanTrnTst(object):
  def __init__(self, model_cfg, path_cfg, gpuids):
    self.model_cfg = model_cfg
    self.path_cfg = path_cfg
    self.gpuids = gpuids

    # trn & tst
    self.tst_reader = None
    self.model = None

    # trn only
    self.scheduler = None
    self.trn_reader = None
    self.loss = None
    self.logger = None

    self.op2monitor = {}

  def g_feed_data_forward_backward(self, data):
    raise NotImplementedError

  def g_feed_data_forward(self, data):
    pass

  # return loss
  def d_feed_data_forward_backward(self, data):
    raise NotImplementedError

  # return acc
  def d_validation(self, buffer):
    raise NotImplementedError

  # return metric dictionary
  def g_validation(self):
    raise NotImplementedError

  def g_predict_in_tst(self):
    raise NotImplementedError

  def d_predict_in_tst(self):
    pass

  def change_lr(self, history, metrics):
    pass

  def train(self, m, trn_reader, tst_reader, scheduler=None):
    self.model = m
    self.trn_reader = trn_reader
    self.tst_reader = tst_reader
    self.scheduler = scheduler

    self.logger = set_logger('Trn', log_path=self.path_cfg.log_file)

    base_epoch = self._load_and_place_device()

    self.model.eval()
    metrics = self.g_validation()
    self.logger.info('epoch %d', base_epoch)
    for key in metrics:
      self.logger.info('%s:%f', key, metrics[key])

    step = 0
    history = []
    for epoch in range(base_epoch, self.model_cfg.g_num_epoch):
      history = self.change_lr(history, metrics)

      step, acc = self._iterate_epoch(step, epoch)

      self.model.eval()
      metrics = self.g_validation()
      metrics['epoch'] = epoch
      self.logger.info('epoch %d', epoch)
      for key in metrics:
        self.logger.info('%s:%f', key, metrics[key])
      val_file = os.path.join(self.path_cfg.log_dir, 'val_metrics.%d.json'%epoch)
      with open(val_file, 'w') as fout:
        json.dump(metrics, fout, indent=2)
      
      if acc < self.model_cfg.d_exit_acc:
        break

  def test(self, m, tst_reader):
    self.model = m
    self.tst_reader = tst_reader

    self._load_and_place_device()

    self.model.eval()
    self.g_predict_in_tst()

  def _iterate_epoch(self, step, epoch):
    self.trn_reader.reset()
    buffer = collections.deque()
    acc = 0.
    for data in self.trn_reader.yield_batch(self.model_cfg.trn_batch_size):
      # generator phase
      if self.model_cfg.g_freeze and epoch < self.model_cfg.g_freeze_epoch:
        self.model.eval()
        self.g_feed_data_forward(data)
      else:
        self.model.train()
        self.g_optimizer.zero_grad()
        self.g_feed_data_forward_backward(data)
        self.g_optimizer.step()

      if self.model_cfg.d_iter != -1:
        if len(buffer) == self.model_cfg.d_buffer_size:
          buffer.popleft()
        buffer.append(data)

      step += 1

      # discriminator phase
      if self.model_cfg.d_iter > 0 and step % self.model_cfg.d_iter == 0:
        d_num_epoch = self.model_cfg.d_num_epoch
        for _ in range(d_num_epoch):
          idxs = range(len(buffer))
          random.shuffle(idxs)
          for idx in idxs:
            data = buffer[idx]
            self.model.eval()
            acc = self.d_validation([data])
            if acc >= self.model_cfg.d_val_acc:
              break

            self.model.train()
            self.d_optimizer.zero_grad()
            loss = self.d_feed_data_forward_backward(data)
            self.d_optimizer.step()
          if acc >= self.model_cfg.d_val_acc:
            break
        print acc, self.model_cfg.d_exit_acc

        if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
          self.logger.info('(step %d) discrimintor acc: %f', step, acc)

        self.model.eval()
        acc = self.d_validation(buffer)
        if acc < self.model_cfg.d_exit_acc and epoch < 1: # end training, reach equilibrium
          break

        # buffer = []

      if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
        model = self.model
        if isinstance(model, nn.DataParallel):
          model = model.module
        for name in model.op2monitor:
          val = model.op2monitor[name].data.cpu().numpy()
          self.logger.info('(step %d) monitor "%s":%s', step, name, pprint.pformat(val))
        for name in self.op2monitor:
          val = self.op2monitor[name].data.cpu().numpy()
          self.logger.info('(step %d) monitor "%s":%s', step, name, pprint.pformat(val))

      if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
        self.model.eval()
        metrics = self.g_validation()
        self.logger.info('step %d', step)
        for key in metrics:
          self.logger.info('%s:%f', key, metrics[key])

    model_file = os.path.join(self.path_cfg.model_dir, 'epoch-%d.pth'%epoch)
    save_gan_checkpoint(self.model, self.g_optimizer, self.d_optimizer, epoch, model_file)

    return step, acc

  def _load_and_place_device(self):
    if self.path_cfg.model_file != '':
      data = torch.load(self.path_cfg.model_file)
      self.model.load_state_dict(data['state_dict'], strict=False)
      model = self.model
      if len(self.gpuids) > 1:
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpuids)
        self.model.cuda()
      else:
        self.model.cuda(self.gpuids[0])
      self.g_optimizer = torch.optim.Adam(model.g_trainable_params(), lr=self.model_cfg.g_base_lr)
      self.d_optimizer = torch.optim.Adam(model.d_trainable_params(), lr=self.model_cfg.d_base_lr)
      if data['g_optimizer'] is not None:
        self.g_optimizer.load_state_dict(data['g_optimizer'])
        for state in self.g_optimizer.state.values():
          for k, v in state.items():
            if torch.is_tensor(v):
              state[k] = v.cuda(self.gpuids[0])
      if data['d_optimizer'] is not None:
        self.d_optimizer.load_state_dict(data['d_optimizer'])
        for state in self.d_optimizer.state.values():
          for k, v in state.items():
            if torch.is_tensor(v):
              state[k] = v.cuda(self.gpuids[0])
      if data['epoch'] is not None:
        base_epoch = data['epoch'] + 1
      else:
        base_epoch = 0
    else:
      model = self.model
      if len(self.gpuids) > 1:
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpuids)
        self.model.cuda()
      else:
        self.model.cuda(self.gpuids[0])
      self.g_optimizer = torch.optim.Adam(model.g_trainable_params(), lr=self.model_cfg.g_base_lr)
      self.d_optimizer = torch.optim.Adam(model.d_trainable_params(), lr=self.model_cfg.d_base_lr)
      base_epoch = 0

    return base_epoch


# class PretrainDTrnTst(GanTrnTst):
#   def _iterate_epoch(self, step, epoch):
#     self.trn_reader.reset()
#     buffer = []
#     for data in self.trn_reader.yield_batch(self.model_cfg.trn_batch_size):
#       # generator phase
#       self.model.eval()
#       self.g_feed_data_forward_backward(data)

#       if self.model_cfg.d_iter != -1:
#         buffer.append(data)

#       step += 1

#       # discriminator phase
#       if self.model_cfg.d_iter > 0 and step % self.model_cfg.d_iter == 0:
#         for _ in range(self.model_cfg.d_num_epoch):
#           self.model.train()
#           for data in buffer:
#             self.d_optimizer.zero_grad()
#             self.d_feed_data_forward_backward(data)
#             self.d_optimizer.step()

#           self.model.eval()
#           acc = self.d_validation(buffer)
#           if acc >= self.model_cfg.d_val_acc:
#             break
#         # print acc
#         if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
#           self.logger.info('(step %d) discrimintor acc: %f', step, acc)

#         buffer = []

#       if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
#         model = self.model
#         if isinstance(model, nn.DataParallel):
#           model = model.module
#         for name in model.op2monitor:
#           val = model.op2monitor[name].data.cpu().numpy()
#           self.logger.info('(step %d) monitor "%s":%s', step, name, pprint.pformat(val))

#       if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
#         self.model.eval()
#         metrics = self.g_validation()
#         self.logger.info('step %d', step)
#         for key in metrics:
#           self.logger.info('%s:%f', key, metrics[key])

#     # discriminator step
#     if self.model_cfg.d_iter == 0:
#       for n in range(self.model_cfg.d.num_epoch):
#         self.model.train()
#         for data in buffer:
#           self.d_optimizer.zero_grad()
#           self.d_feed_data_forward_backward(data)
#           self.d_optimizer.step()
        
#         self.model.eval()
#         acc = self.d_validation(buffer)
#         if acc >= self.model_cfg.d_val_acc:
#           break
#         # print acc
#       self.logger.info('discrimintor acc: %f', acc)

#       buffer = []

#     model_file = os.path.join(self.path_cfg.model_dir, 'epoch-%d.pth'%epoch)
#     save_gan_checkpoint(self.model, self.g_optimizer, self.d_optimizer, epoch, model_file)

#     return step


class Reader(object):
  def yield_batch(self, batch_size, **kwargs):
    raise NotImplementedError

  def reset(self):
    pass


def set_logger(name, log_path=None):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)

  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  logger.addHandler(console)

  if log_path is not None:
    if os.path.exists(log_path):
      os.remove(log_path)

    logfile = logging.FileHandler(log_path)
    logfile.setLevel(logging.INFO)
    logfile.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(logfile)

  return logger


def gen_dir_struct_info(path_cfg, path_cfg_file):
  path_cfg.load(path_cfg_file)

  output_dir = path_cfg.output_dir

  log_dir = os.path.join(output_dir, 'log')
  if not os.path.exists(log_dir): 
    os.makedirs(log_dir)
  model_dir = os.path.join(output_dir, 'model')
  if not os.path.exists(model_dir): 
    os.makedirs(model_dir)
  predict_dir = os.path.join(output_dir, 'pred')
  if not os.path.exists(predict_dir): 
    os.makedirs(predict_dir)

  path_cfg.log_dir = log_dir
  path_cfg.model_dir = model_dir

  timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  path_cfg.log_file = os.path.join(log_dir, 'log-' + timestamp)

  return path_cfg


def save_checkpoint(module, optimizer, epoch, model_file):
  if isinstance(module, nn.DataParallel):
    module = module.module
  state_dict = module.state_dict()
  state_dict_cpu = {}
  for key in state_dict:
    state_dict_cpu[key] = state_dict[key].cpu()
  torch.save({
    'state_dict': state_dict_cpu,
    'optimizer': optimizer.state_dict() if optimizer is not None else None,
    'epoch': epoch,
    }, model_file)


def save_gan_checkpoint(module, g_optimizer, d_optimizer, epoch, model_file):
  if isinstance(module, nn.DataParallel):
    module = module.module
  state_dict = module.state_dict()
  state_dict_cpu = {}
  for key in state_dict:
    state_dict_cpu[key] = state_dict[key].cpu()
  torch.save({
    'state_dict': state_dict_cpu,
    'g_optimizer': g_optimizer.state_dict() if g_optimizer is not None else None,
    'd_optimizer': d_optimizer.state_dict() if d_optimizer is not None else None,
    'epoch': epoch,
    }, model_file)
