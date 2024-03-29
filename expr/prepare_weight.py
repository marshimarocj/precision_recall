import os
import sys
sys.path.append('../')

import torch

import model.gan_simple_sc
import driver.gan_simple_sc
import model.gan_sc
import driver.gan_sc
import model.vead_gan_simple_sc
import driver.vead_gan_simple_sc
import model.vead_gan_cider_sc
import driver.vead_gan_cider_sc
import model.deep_discriminator
import driver.deep_discriminator


def prepare_rl_from_ml():
  root_dir = '/data1/jiac/mscoco/pytorch' # mercurial
  ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  rl_model_file = os.path.join(root_dir, 'vevd_sc_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'pretrain.pth')

  data = torch.load(ml_model_file)
  out = {
    'state_dict': data['state_dict'],
    'optimizer': None,
    'epoch': None,
  }
  torch.save(out, rl_model_file)


def prepare_for_gan_simple():
  # root_dir = '/data1/jiac/mscoco/pytorch' # mercurial
  # root_dir = '/data1/jiac/MSCOCO/pytorch' # uranus
  root_dir = '/hdd/mscoco/pytorch' # aws

  # ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.5.50.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.-1.0.80')

  # ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.50.5.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.50.5.5.0.80.5.0')

  # ml_model_file = os.path.join(root_dir, 'vevd_sc_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-48.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.5.50.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.vevd_sc')

  # ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.512.1.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.512.1.5.0.80')

  # ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.512.1.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.512.1.5.0.80.5.0')

  # ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.64.8.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.64.8.5.0.80')

  # ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.50.5.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.50.5.5.0.80')

  ml_model_file = os.path.join(root_dir, 'pure_vead_ml_expr', 'bottomup.512.512.512.512.2048.1.0.att2in_boom.add', 'model', 'epoch-23.pth')
  discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.5.50.512.lstm', 'model', 'epoch-20.path')
  expr_name = os.path.join(root_dir, 'vead_gan_simple_sc_expr', 'bottomup.512.512.512.512.2048.add.5.50.5.0.80')

  model_cfg_file = expr_name + '.model.json'
  path_cfg_file = expr_name + '.path.json'
  out_file = os.path.join(expr_name, 'model', 'pretrain.pth')

  path_cfg = driver.gan_simple_sc.gen_dir_struct_info(path_cfg_file)
  model_cfg = driver.gan_simple_sc.load_and_fill_model_cfg(model_cfg_file, path_cfg)

  m = model.gan_simple_sc.Model(model_cfg)

  data = torch.load(ml_model_file)
  m.load_state_dict(data['state_dict'], strict=False)

  data = torch.load(discriminator_model_file)
  m.discriminator.load_state_dict(data['state_dict'])

  torch.save({
    'state_dict': m.state_dict(),
    'g_optimizer': None,
    'd_optimizer': None,
    'epoch': None
  }, out_file)


def prepare_for_gan():
  # root_dir = '/data1/jiac/mscoco/pytorch' # mercurial
  root_dir = '/hdd/mscoco/pytorch' # aws

  # ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  # discriminator_model_file = os.path.join(root_dir, 'discriminator', 'tf_resnet152_450.5.50.512.1.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.0')

  # ml_model_file = os.path.join(root_dir, 'vevd_sc_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-48.pth')
  # discriminator_model_file = os.path.join(root_dir, 'discriminator', 'tf_resnet152_450.5.50.512.1.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vevd_gan_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.vevd_sc')

  ml_model_file = os.path.join(root_dir, 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'model', 'epoch-38.pth')
  discriminator_model_file = os.path.join(root_dir, 'discriminator', 'tf_resnet152_450.5.50.512.1.lstm', 'model', 'epoch-20.pth')
  expr_name = os.path.join(root_dir, 'vevd_gan_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.5.0')

  model_cfg_file = expr_name + '.model.json'
  path_cfg_file = expr_name + '.path.json'
  out_file = os.path.join(expr_name, 'model', 'pretrain.pth')

  path_cfg = driver.gan_sc.gen_dir_struct_info(path_cfg_file)
  model_cfg = driver.gan_sc.load_and_fill_model_cfg(model_cfg_file, path_cfg)

  m = model.gan_sc.Model(model_cfg)

  data = torch.load(ml_model_file)
  m.load_state_dict(data['state_dict'], strict=False)

  data = torch.load(discriminator_model_file)
  m.discriminator.load_state_dict(data['state_dict'])

  torch.save({
    'state_dict': m.state_dict(),
    'g_optimizer': None,
    'd_optimizer': None,
    'epoch': None
  }, out_file)


def prepare_for_vead_gan_simple():
  root_dir = '/hdd/mscoco/pytorch' # aws

  ml_model_file = os.path.join(root_dir, 'pure_vead_ml_expr', 'bottomup.512.512.512.512.2048.1.0.att2in_boom.add', 'model', 'epoch-23.pth')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.5.50.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vead_gan_simple_sc_expr', 'bottomup.512.512.512.512.2048.add.5.50.5.0.80')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.50.5.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vead_gan_simple_sc_expr', 'bottomup.512.512.512.512.2048.add.50.5.5.0.80')
  # discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.8.64.512.lstm', 'model', 'epoch-20.pth')
  # expr_name = os.path.join(root_dir, 'vead_gan_simple_sc_expr', 'bottomup.512.512.512.512.2048.add.8.64.5.0.80')
  discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'tf_resnet152_450.64.64.512.lstm', 'model', 'epoch-20.pth')
  expr_name = os.path.join(root_dir, 'vead_gan_simple_sc_expr', 'bottomup.512.512.512.512.2048.add.64.64.5.0.80')

  model_cfg_file = expr_name + '.model.json'
  path_cfg_file = expr_name + '.path.json'
  out_file = os.path.join(expr_name, 'model', 'pretrain.pth')

  path_cfg = driver.vead_gan_simple_sc.gen_dir_struct_info(path_cfg_file)
  model_cfg = driver.vead_gan_simple_sc.load_and_fill_model_cfg(model_cfg_file, path_cfg)

  m = model.vead_gan_simple_sc.Model(model_cfg)

  data = torch.load(ml_model_file)
  m.load_state_dict(data['state_dict'], strict=False)

  data = torch.load(discriminator_model_file)
  m.discriminator.load_state_dict(data['state_dict'])

  torch.save({
    'state_dict': m.state_dict(),
    'g_optimizer': None,
    'd_optimizer': None,
    'epoch': None
  }, out_file)


def prepare_for_vead_gan():
  root_dir = '/hdd/mscoco/pytorch' # aws

  ml_model_file = os.path.join(root_dir, 'pure_vead_ml_expr', 'bottomup.512.512.512.512.2048.1.0.att2in_boom.add', 'model', 'epoch-23.pth')
  discriminator_model_file = os.path.join(root_dir, 'discriminator', 'tf_resnet152_450.64.8.512.1.lstm', 'model', 'epoch-20.pth')
  expr_name = os.path.join(root_dir, 'vead_gan_cider_sc_expr', 'bottomup.512.512.512.512.2048.add.mean.64.8.5.0.80.1.0.8.5.0')

  model_cfg_file = expr_name + '.model.json'
  path_cfg_file = expr_name + '.path.json'
  out_file = os.path.join(expr_name, 'model', 'pretrain.pth')

  path_cfg = driver.vead_gan_cider_sc.gen_dir_struct_info(path_cfg_file)
  model_cfg = driver.vead_gan_cider_sc.load_and_fill_model_cfg(model_cfg_file, path_cfg)

  m = model.vead_gan_cider_sc.Model(model_cfg)

  data = torch.load(ml_model_file)
  m.load_state_dict(data['state_dict'], strict=False)

  data = torch.load(discriminator_model_file)
  m.discriminator.load_state_dict(data['state_dict'])

  torch.save({
    'state_dict': m.state_dict(),
    'g_optimizer': None,
    'd_optimizer': None,
    'epoch': None
  }, out_file)


def prepare_for_deep_discriminator():
  root_dir = '/mnt/data1/jiac/mscoco/pytorch' # neptune

  discriminator_model_file = os.path.join(root_dir, 'simple_discriminator', 'resnet152_450.512.1.512.lstm', 'model', 'epoch-20.pth')
  cnn_model_file = '/home/jiac/models/pytorch/resnet/resnet152-b121ed2d.pth'
  expr_name = os.path.join(root_dir, 'deep_discriminator', 'resnet152.512.1.512.lstm')

  model_cfg_file = expr_name + '.model.json'
  path_cfg_file = expr_name + '.path.json'
  out_file = os.path.join(expr_name, 'model', 'pretrain.pth')

  path_cfg = driver.deep_discriminator.gen_dir_struct_info(path_cfg_file)
  model_cfg = driver.deep_discriminator.load_and_fill_model_cfg(model_cfg_file, path_cfg)

  m = model.deep_discriminator.Model(model_cfg)

  print 'model'

  data = torch.load(discriminator_model_file)
  m.load_state_dict(data['state_dict'], strict=False)

  print 'load cnn'
  data = torch.load(cnn_model_file)
  m.cnn.load_state_dict(data, strict=True)

  torch.save({
    'state_dict': m.state_dict(),
    'optimizer': None,
    'epoch': None,
  }, out_file)



if __name__ == '__main__':
  # prepare_rl_from_ml()
  # prepare_for_gan_simple()
  # prepare_for_gan()
  # prepare_for_vead_gan_simple()
  # prepare_for_vead_gan()
  prepare_for_deep_discriminator()
