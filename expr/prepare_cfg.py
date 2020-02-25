import os
import json
import sys
sys.path.append('../')

import model.discriminator
import model.simple_discriminator
import model.vevd_ml
import model.vevd_sc
import model.vevd_rl
import model.gan_simple_sc
import model.gan_sc
import model.gan_simple_cider_sc
import model.gan_cider_sc
import model.vead_gan_simple_sc
import model.vead_gan_simple_cider_sc
import model.vead_gan_cider_sc


'''func
'''


'''expr
'''
def gen_discriminator_cfg():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  root_dir = '/data1/jiac/MSCOCO' # uranus
  # root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'discriminator')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'num_epoch': 21,
    'lr': 1e-3,
    'dim_kernel': 64,
    'num_kernel': 8,
    'discriminator_noise': .5,
    'dim_ft': 2048,
    'num_sentence': 5,
    'tied_sentence_encoder': True,

    'cell': 'lstm',
    'dim_input': 512,
    'dim_hidden': 512,
    'dropin': .5,
  }

  model_cfg = model.discriminator.gen_cfg(**params)
  outprefix = '%s/%s.%d.%d.%d.%d.%s'%(
    out_dir, ft_name, 
    params['dim_kernel'], params['num_kernel'], 
    params['dim_hidden'], params['tied_sentence_encoder'], params['cell']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_simple_discriminator_cfg():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/data1/jiac/MSCOCO' # uranus
  # root_dir = '/mnt/data1/jiac/mscoco' # neptune
  root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'simple_discriminator')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'num_epoch': 21,
    'lr': 1e-3,
    'dim_kernel': 64,
    'num_kernel': 64,
    'discriminator_noise': .5,
    'dim_ft': 2048,

    'cell': 'lstm',
    'dim_input': 512,
    'dim_hidden': 512,
    'dropin': .5,
  }

  model_cfg = model.simple_discriminator.gen_cfg(**params)
  outprefix = '%s/%s.%d.%d.%d.%s'%(
    out_dir, ft_name, 
    params['dim_kernel'], params['num_kernel'], 
    params['dim_hidden'], params['cell']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_simple_birnn_discriminator_cfg():
  root_dir = '/data1/jiac/mscoco' # mercurial
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'simple_discriminator')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'num_epoch': 21,
    'lr': 1e-3,
    'dim_kernel': 64,
    'num_kernel': 64,
    'discriminator_noise': .5,
    'dim_ft': 2048,
    'bidirectional': True,

    'cell': 'lstm',
    'dim_input': 512,
    'dim_hidden': 512,
    'dropin': .5,
  }

  model_cfg = model.simple_discriminator.gen_cfg(**params)
  outprefix = '%s/%s.%d.%d.%d.%s.birnn'%(
    out_dir, ft_name, 
    params['dim_kernel'], params['num_kernel'], 
    params['dim_hidden'], params['cell']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_vevd_ml_cfg():
  root_dir = '/data1/jiac/mscoco' # mercurial
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_ml_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'num_epoch': 50,
    'dropin': 0.5,
    'dropout': 0.5,
    'tied': False,
    'dim_ft': 2048,
    'beam_width': 5,
    'cell': 'lstm',
    'lr': 1e-4,
    'min_lr': 1e-4,
    'init_fg': False,
  }

  model_cfg = model.vevd_ml.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%s'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'],
    params['cell'])
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_vevd_sc_cfg():
  root_dir = '/data1/jiac/mscoco' # mercurial
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'num_epoch': 50,
    'dropin': 0.0,
    'dropout': 0.0,
    'tied': False,
    'dim_ft': 2048,
    'beam_width': 5,

    'cell': 'lstm',
    'lr': 1e-5,
    'min_lr': 1e-5,
    'init_fg': False,
    'num_sample': 1,
  }

  model_cfg = model.vevd_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%s'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'], 
    params['cell'])
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'df_file': os.path.join(annotation_dir, 'document_frequency.pkl'),
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_vevd_rl_cfg():
  root_dir = '/data1/jiac/MSCOCO' # uranus
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_rl_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'num_epoch': 50,
    'dropin': 0.0,
    'dropout': 0.0,
    'tied': False,
    'dim_ft': 2048,
    'beam_width': 5,

    'cell': 'lstm',
    'lr': 1e-5,
    'min_lr': 1e-5,
    'init_fg': False,
    'num_sample': 5,
    # 'use_greedy': True,
    'use_greedy': False,
  }

  model_cfg = model.vevd_rl.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%d.%s'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'], 
    params['use_greedy'],
    params['cell'])
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'df_file': os.path.join(annotation_dir, 'document_frequency.pkl'),
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_gan_simple_sc_cfg():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  root_dir = '/data1/jiac/MSCOCO' # uranus
  # root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'tied': False,
    'beam_width': 5,
    'init_fg': False,
    'num_sample': 1,
    'g_num_epoch': 50,
    'g_lr': 1e-5,
    'g_freeze': False,
    'g_freeze_epoch': -1,

    'd_noise': .5,
    'dim_kernel': 50,
    'num_kernel': 5,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
  }

  model_cfg = model.gan_simple_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%s.%d.%d.%d.%.2f'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'], params['cell'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_vead_gan_simple_sc_cfg():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_simple_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'beam_width': 5,
    'num_sample': 1,
    'g_num_epoch': 50,
    'g_lr': 1e-5,
    'g_freeze': True,
    'g_freeze_epoch': 1,

    'dim_att_ft': 2048,
    'num_att_ft': 36,
    'dim_key': 512,
    'dim_val': 512,
    'tied_key_val': False,
    'val_proj': True,
    'dim_boom': 2048,
    'sim': 'add',

    'd_noise': .5,
    'dim_kernel': 64,
    'num_kernel': 64,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
    'd_buffer_size': 5,
  }

  model_cfg = model.vead_gan_simple_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%d.%d.%s.%d.%d.%d.%.2f'%(
    os.path.join(out_dir, 'bottomup'),
    params['dim_hidden'], params['dim_embed'], params['dim_key'], params['dim_val'], params['dim_boom'], params['sim'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'trn_ft.npy'),
    'val_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'val_ft.npy'),
    'tst_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_gan_simple_cider_sc_cfg():
  root_dir = '/data1/jiac/MSCOCO' # uranus
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/mnt/data1/jiac/mscoco' # neptune
  # root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
    'reward_alpha': 2.5,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'tied': False,
    'beam_width': 5,
    'init_fg': False,
    'num_sample': 1,
    'g_num_epoch': 50,
    'g_lr': 1e-5,
    'g_freeze': False,
    'g_freeze_epoch': 1,

    'd_noise': .5,
    'dim_kernel': 5,
    'num_kernel': 50,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
  }

  model_cfg = model.gan_simple_cider_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%s.%d.%d.%d.%.2f.%.1f'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'], params['cell'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc'],
    params['reward_alpha']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'df_file': os.path.join(annotation_dir, 'document_frequency.pkl'),
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_gan_sc_pretrain_cfg():
  root_dir = '/data1/jiac/mscoco' # mercurial
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'tied': False,
    'beam_width': 5,
    'init_fg': False,
    'num_sample': 5,
    'g_num_epoch': 5,
    'g_lr': 1e-5,

    'd_noise': .5,
    'dim_kernel': 5,
    'num_kernel': 50,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
    'd_late_fusion': False,
  }

  model_cfg = model.gan_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%s.%d.%d.%d.%.2f.%d.pretrain_d'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'], params['cell'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc'], params['d_late_fusion']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_gan_sc_cfg():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'tied': False,
    'beam_width': 5,
    'init_fg': False,
    'num_sample': 5,
    'g_num_epoch': 50,
    'g_lr': 1e-5,
    'g_freeze': True,
    'g_freeze_epoch': 1,
    'g_baseline': 'mean',

    'd_noise': .5,
    'dim_kernel': 50,
    'num_kernel': 5,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
    'd_late_fusion': True,
    'd_quality_alpha': .8,
  }

  model_cfg = model.gan_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%s.%s.%d.%d.%d.%.2f.%d.%.1f'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'], params['cell'], params['g_baseline'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc'], params['d_late_fusion'], params['d_quality_alpha']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_gan_cider_sc_cfg():
  root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'tied': False,
    'beam_width': 5,
    'init_fg': False,
    'num_sample': 5,
    'g_num_epoch': 50,
    'g_lr': 1e-5,
    'g_freeze': True,
    'g_freeze_epoch': 1,
    'g_baseline': 'mean',

    'd_noise': .5,
    'dim_kernel': 5,
    'num_kernel': 50,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
    'd_late_fusion': True,
    'd_quality_alpha': .8,
    'd_cider_alpha': 2.5,
  }

  model_cfg = model.gan_cider_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%s.%s.%d.%d.%d.%.2f.%d.%.1f.%.1f'%(
    os.path.join(out_dir, ft_name),
    params['dim_hidden'], params['dim_embed'], params['tied'], params['cell'], params['g_baseline'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc'], 
    params['d_late_fusion'], params['d_quality_alpha'], params['d_cider_alpha']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
    'df_file': os.path.join(annotation_dir, 'document_frequency.pkl'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_vead_gan_cider_sc_cfg():
  root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_cider_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'tied': False,
    'beam_width': 5,
    'init_fg': False,
    'num_sample': 5,
    'g_num_epoch': 50,
    'g_lr': 1e-5,
    'g_freeze': True,
    'g_freeze_epoch': 1,
    'g_baseline': 'mean',

    'dim_att_ft': 2048,
    'num_att_ft': 36,
    'dim_key': 512,
    'dim_val': 512,
    'tied_key_val': False,
    'val_proj': True,
    'dim_boom': 2048,
    'sim': 'add',

    'd_noise': .5,
    'dim_kernel': 64,
    'num_kernel': 8,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
    'd_late_fusion': True,
    'd_quality_alpha': .8,
    'd_cider_alpha': 5.,
    'd_buffer_size': 5,
  }

  model_cfg = model.vead_gan_cider_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%d.%d.%s.%s.%d.%d.%d.%.2f.%d.%.1f.%.1f'%(
    os.path.join(out_dir, 'bottomup'),
    params['dim_hidden'], params['dim_embed'], params['dim_key'], params['dim_val'], params['dim_boom'], params['sim'], params['g_baseline'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc'], 
    params['d_late_fusion'], params['d_quality_alpha'], params['d_cider_alpha']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'trn_ft.npy'),
    'val_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'val_ft.npy'),
    'tst_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
    'df_file': os.path.join(annotation_dir, 'document_frequency.pkl'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


def gen_vead_gan_simple_cider_sc_cfg():
  root_dir = '/hdd/mscoco' # aws
  split_dir = os.path.join(root_dir, 'pytorch', 'split')
  annotation_dir = os.path.join(root_dir, 'aux')
  out_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_simple_cider_sc_expr')

  ft_name = 'tf_resnet152_450'

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  params = {
    'max_step': 20,
    'dim_embed': 512,
    'dim_hidden': 512,
    'cell': 'lstm',
    'dim_ft': 2048,
  
    'g_dropin': 0.0,
    'g_dropout': 0.0,
    'beam_width': 5,
    'num_sample': 1,
    'g_num_epoch': 50,
    'g_lr': 1e-5,
    'g_freeze': False,
    'g_freeze_epoch': 1,

    'dim_att_ft': 2048,
    'num_att_ft': 36,
    'dim_key': 512,
    'dim_val': 512,
    'tied_key_val': False,
    'val_proj': True,
    'dim_boom': 2048,
    'sim': 'add',

    'd_noise': .5,
    'dim_kernel': 8,
    'num_kernel': 64,
    'd_num_epoch': 5,
    'd_lr': 1e-3,
    'd_iter': 5,
    'd_val_acc': .8,
    'reward_alpha': 5.,
    'd_buffer_size': 5,
  }

  model_cfg = model.vead_gan_simple_cider_sc.gen_cfg(**params)
  outprefix = '%s.%d.%d.%d.%d.%d.%s.%d.%d.%d.%.2f.%.1f'%(
    os.path.join(out_dir, 'bottomup'),
    params['dim_hidden'], params['dim_embed'], params['dim_key'], params['dim_val'], params['dim_boom'], params['sim'],
    params['dim_kernel'], params['num_kernel'], params['d_iter'], params['d_val_acc'],
    params['reward_alpha']
  )
  model_cfg_file = '%s.model.json'%outprefix
  model_cfg.save(model_cfg_file)

  output_dir = outprefix
  path_cfg = {
    'trn_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'trn_ft.npy'),
    'val_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'val_ft.npy'),
    'tst_ftfile': os.path.join(root_dir, 'mp_feature', ft_name, 'tst_ft.npy'),
    'trn_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'trn_ft.npy'),
    'val_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'val_ft.npy'),
    'tst_att_ftfile': os.path.join(root_dir, 'bottom_up_feature', 'tst_ft.npy'),
    'trn_annotation_file': os.path.join(split_dir, 'trn_id_caption_mask.pkl'),
    'val_annotation_file': os.path.join(split_dir, 'val_id_caption_mask.pkl'),
    'split_dir': split_dir,
    'annotation_dir': annotation_dir,
    'groundtruth_file': os.path.join(annotation_dir, 'human_caption_dict.pkl'),
    'word_file': os.path.join(annotation_dir, 'int2word.pkl'),
    'output_dir': output_dir,
    'df_file': os.path.join(annotation_dir, 'document_frequency.pkl'),
    'model_file': os.path.join(output_dir, 'model', 'pretrain.pth'),
  }
  path_cfg_file = '%s.path.json'%outprefix

  if not os.path.exists(path_cfg['output_dir']):
    os.mkdir(path_cfg['output_dir'])

  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)


if __name__ == '__main__':
  # gen_discriminator_cfg()
  # gen_simple_discriminator_cfg()
  # gen_simple_birnn_discriminator_cfg()
  # gen_margin_discriminator_cfg()
  # gen_vevd_ml_cfg()
  # gen_vevd_sc_cfg()
  # gen_vevd_rl_cfg()

  # gen_gan_simple_sc_cfg()
  # gen_gan_simple_cider_sc_cfg()
  # gen_gan_sc_pretrain_cfg()
  # gen_gan_sc_cfg()
  # gen_gan_cider_sc_cfg()

  # gen_vead_gan_simple_sc_cfg()
  # gen_vead_gan_simple_cider_sc_cfg()
  gen_vead_gan_cider_sc_cfg()
