import os
import json
import pickle
import subprocess
import sys
import md5
sys.path.append('../')

import numpy as np

from bleu.bleu import Bleu
from cider.cider import Cider
from rouge.rouge import Rouge
from meteor.meteor import Meteor
from spice.spice import Spice


'''func
'''
def get_res_gts_dict(res_file, gts_file):
  human_caption = pickle.load(file(gts_file))
  data = json.load(file(res_file))

  res, gts = {}, {}
  for key, value in data.iteritems():
    gts[key] = human_caption[int(key)]
    res[key] = [value]

  return res, gts


def eval(predict_file, groundtruth_file):
  res, gts = get_res_gts_dict(predict_file, groundtruth_file)

  bleu_scorer = Bleu(4)
  meteor_scorer = Meteor()
  rouge_scorer = Rouge()
  cider_scorer = Cider()
  spice_scorer = Spice()
  # closest score
  res_bleu, _ = bleu_scorer.compute_score(gts, res)
  # metero handles the multi references (don't know the details yet)
  res_meteor, _ = meteor_scorer.compute_score(gts, res)
  meteor_scorer.meteor_p.kill()
  # average
  res_rouge, _ = rouge_scorer.compute_score(gts, res)
  # average
  res_cider, _ = cider_scorer.compute_score(gts, res)
  res_spice, _ = spice_scorer.compute_score(gts, res)

  out = {
    'bleu': res_bleu, 
    'meteor': res_meteor,
    'rouge': res_rouge,
    'cider': res_cider,
    'spice': res_spice,
  }

  return out


def eval_precision(vid2sent_scores, vid2gt, num):
  vids = vid2sent_scores.keys()

  gts = {}
  for vid in vids:
    gts[vid] = vid2gt[int(vid)]

  cum_bleu4, cum_meteor, cum_rouge, cum_cider, cum_spice = 0., 0., 0., 0., 0.
  precisions = {
    'bleu4': [],
    'meteor': [],
    'rouge': [],
    'cider': [],
    'spice': [],
  }
  for i in range(num):
    if (i+1) % 10 == 0:
      print(i+1)
    bleu_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider() # need to accelerate
    spice_scorer = Spice()

    predicts = {}
    for vid in vids:
      sent_scores = vid2sent_scores[vid]
      if len(sent_scores) <= i:
        predicts[vid] = ['<EOS>']
      else:
        predicts[vid] = [sent_scores[i]['sent']]
    
    res_bleu, _ = bleu_scorer.compute_score(gts, predicts)
    res_meteor, _ = meteor_scorer.compute_score(gts, predicts)
    meteor_scorer.meteor_p.kill()
    res_rouge, _ = rouge_scorer.compute_score(gts, predicts)
    res_cider, _ = cider_scorer.compute_score(gts, predicts)
    res_spice, _ = spice_scorer.compute_score(gts, predicts)

    cum_bleu4 += res_bleu[-1]
    cum_meteor += res_meteor
    cum_rouge += res_rouge
    cum_cider += res_cider
    cum_spice += res_spice

    # precision = (cum_bleu4 + cum_meteor + cum_rouge + cum_cider) / (i+1)
    # precisions.append(precision)
    precisions['bleu4'].append(cum_bleu4 / (i+1))
    precisions['meteor'].append(cum_meteor / (i+1))
    precisions['rouge'].append(cum_rouge / (i+1))
    precisions['cider'].append(cum_cider / (i+1))
    precisions['spice'].append(cum_spice / (i+1))

  return precisions


def eval_precision_detail(vid2sent_scores, vid2gt, num):
  vids = vid2sent_scores.keys()

  gts = {}
  for vid in vids:
    gts[vid] = vid2gt[int(vid)]

  vid2out = {}
  for vid in vids:
    vid2out[vid] = []

  for k in range(num):
    if (k+1) % 10 == 0:
      print k+1

    predicts = {}
    for vid in vids:
      sent_scores = vid2sent_scores[vid]
      if len(sent_scores) <= k:
        predicts[vid] = ['<EOS>']
      else:
        predicts[vid] = [sent_scores[k]['sent']]

    bleu_scorer = Bleu(4)
    meteor_scorer = Meteor()
    rouge_scorer = Rouge()
    cider_scorer = Cider() # need to accelerate
    spice_scorer = Spice()

    _, res_bleus = bleu_scorer.compute_score(gts, predicts)
    _, res_meteors = meteor_scorer.compute_score(gts, predicts)
    meteor_scorer.meteor_p.kill()
    _, res_rouges = rouge_scorer.compute_score(gts, predicts)
    _, res_ciders = cider_scorer.compute_score(gts, predicts)
    _, res_spices = spice_scorer.compute_score(gts, predicts)

    for vid, b, m, r, c in zip(vids, res_bleus[-1], res_meteors, res_rouges, res_ciders):
      sent_scores = vid2sent_scores[vid]
      if len(sent_scores) <= k:
        continue
      
      sent_score = sent_scores[k]
      sent_score['bleu4'] = b
      sent_score['meteor'] = m
      sent_score['rouge'] = r
      sent_score['cider'] = c
      vid2out[vid].append(sent_score)

    for vid, s in zip(sorted(vids), res_spices):
      sent_scores = vid2sent_scores[vid]
      if len(sent_scores) <= k:
        continue
      
      vid2out[vid][-1]['spice'] = s['All']['f']

  return vid2out


def eval_spice(vid2sent_scores, vid2gt, num):
  from spice.spice import Spice

  vids = vid2sent_scores.keys()

  gts = {}
  for vid in vids:
    gts[vid] = vid2gt[int(vid)]

  cum_spice = 0.
  precisions = {
    'spice': [],
  }
  for i in range(num):
    if (i+1) % 10 == 0:
      print(i+1)
    spice = Spice()

    predicts = {}
    for vid in vids:
      sent_scores = vid2sent_scores[vid]
      if len(sent_scores) <= i:
        predicts[vid] = ['<EOS>']
      else:
        predicts[vid] = [sent_scores[i]['sent']]
    
    res_spice, _ = spice.compute_score(gts, predicts)

    cum_spice += res_spice

    precisions['spice'].append(cum_spice / (i+1))

  return precisions


def eval_recall(vid2sent_scores, num):
  vids = vid2sent_scores.keys()

  divs = {}
  for vid in vids:
    divs[vid] = [set() for _ in range(4)]

  recalls = {
    'div1': [],
    'div2': [],
    'div3': [],
    'div4': [],
  }
  for i in range(num):
    recall = [0.]*4
    for vid in vids:
      sent_scores = vid2sent_scores[vid]
      if i >= len(sent_scores):
        continue
      sent = sent_scores[i]['sent']
      words = sent.split(' ')
      n = len(words)
      for j in range(n):
        for k in range(4):
          if j < n-k:
            gram = ' '.join(words[j:j+k+1])
            divs[vid][k].add(gram)
      # recall += sum([len(divs[vid][j]) for j in range(4)]) / 4.
    # recall /= len(vids)
    # recalls.append(recall)
      for k in range(4):
        recall[k] += len(divs[vid][k])
    recalls['div1'].append(recall[0] / len(vids))
    recalls['div2'].append(recall[1] / len(vids))
    recalls['div3'].append(recall[2] / len(vids))
    recalls['div4'].append(recall[3] / len(vids))
  return recalls


def eval_corpus_recall(vid2sent_scores, num):
  vids = vid2sent_scores.keys()

  divs = [set() for _ in range(4)]

  recalls = {
    'div1': [],
    'div2': [],
    'div3': [],
    'div4': []
  }
  for i in range(num):
    for vid in vids:
      sent_scores = vid2sent_scores[vid]
      if i >= len(sent_scores):
        continue
      sent = sent_scores[i]['sent']
      words = sent.split(' ')
      n = len(words)
      for j in range(n):
        for k in range(4):
          if j < n-k:
            gram = ' '.join(words[j:j+k+1])
            divs[k].add(gram)
    recalls['div1'].append(len(divs[0]))
    recalls['div2'].append(len(divs[1]))
    recalls['div3'].append(len(divs[2]))
    recalls['div4'].append(len(divs[3]))
  return recalls


def auto_select(logdir, lower=-1, upper=-1):
  names = os.listdir(logdir)
  bleu4s = []
  ciders = []
  epochs = []
  for name in names:
    if 'val_metrics' not in name:
      continue
    val_file = os.path.join(logdir, name)
    with open(val_file) as f:
      data = json.load(f)
    epoch = data['epoch']
    if (lower != -1 and epoch < lower) or (upper != -1 and epoch >= upper):
      continue
    bleu4s.append(data['bleu4'])
    ciders.append(data['cider'])
    epochs.append(data['epoch'])

  best_epochs = set()
  
  idx = np.argmax(bleu4s)
  epoch = epochs[int(idx)]
  best_epochs.add(epoch)

  idx = np.argmax(ciders)
  epoch = epochs[int(idx)]
  best_epochs.add(epoch)

  return best_epochs


def auto_select_discriminator(logdir):
  names = os.listdir(logdir)
  best_epoch = -1
  min_loss = 1e10
  for name in names:
    if 'val_metrics' not in name:
      continue
    val_file = os.path.join(logdir, name)
    with open(val_file) as f:
      data = json.load(f)
    epoch = data['epoch']
    loss = data['loss']
    if loss < min_loss:
      best_epoch = epoch
      min_loss = loss
  return best_epoch


def predict(python_file, model_cfg_file, path_cfg_file, best_epochs, gpuid, **kwargs):
  with open('eval.sh', 'w') as fout:
    fout.write('#!/bin/sh\n')
    fout.write('export CUDA_VISIBLE_DEVICES=%d\n'%gpuid)
    fout.write('cd ../driver\n')
    for best_epoch in best_epochs:
      cmd = [
        'python', python_file,
        model_cfg_file, path_cfg_file,
        '--best_epoch', str(best_epoch),
        '--is_train', '0',
      ]
      for key in kwargs:
        cmd += ['--' + key, str(kwargs[key])]
      fout.write(' '.join(cmd) + '\n')

  os.system('chmod 777 eval.sh')
  subprocess.call(['./eval.sh'])


'''expr
'''
def predict_eval():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/mnt/data1/jiac/mscoco' # neptune
  # root_dir = '/data1/jiac/MSCOCO' # uranus
  root_dir = '/hdd/mscoco' # aws
  gt_file = os.path.join(root_dir, 'aux', 'human_caption_dict.pkl')

  # model_name = 'pytorch/vevd_ml_expr/tf_resnet152_450.512.512.0.lstm'
  # python_file = 'vevd_ml.py'

  # model_name = 'pytorch/vevd_sc_expr/tf_resnet152_450.512.512.0.lstm'
  # model_name = 'pytorch/vevd_sc_expr/tf_resnet152_450.512.512.0.lstm.1'
  # python_file = 'vevd_sc.py'

  # model_name = 'pytorch/vevd_rl_expr/tf_resnet152_450.512.512.0.0.lstm'
  # model_name = 'pytorch/vevd_rl_expr/tf_resnet152_450.512.512.0.1.lstm'
  # python_file = 'vevd_rl.py'

  # model_name = 'pytorch/vevd_gan_simple_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80'
  # python_file = 'gan_simple_sc.py'

  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.50.5.5.0.80.5.0'
  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.5.0'
  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.1.0'
  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.2.5'
  # python_file = 'gan_simple_cider_sc.py'

  # model_name = 'pytorch/vevd_gan_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.5.0'
  # model_name = 'pytorch/vevd_gan_sc_expr/tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8'
  # model_name = 'pytorch/vevd_gan_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.1.0'
  # python_file = 'gan_sc.py'

  # model_name = 'pytorch/vead_gan_cider_sc_expr/bottomup.512.512.512.512.2048.add.mean.64.8.5.0.80.1.0.8.5.0'
  # python_file = 'vead_gan_cider_sc.py'

  model_name = 'pytorch/vead_gan_simple_cider_sc_expr/bottomup.512.512.512.512.2048.add.64.8.5.0.80.5.0'
  python_file = 'vead_gan_simple_cider_sc.py'

  logdir = os.path.join(root_dir, model_name, 'log')
  preddir = os.path.join(root_dir, model_name, 'pred')
  model_cfg_file = os.path.join(root_dir, model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, model_name + '.path.json')

  gpuid = 3

  best_epochs = auto_select(logdir, lower=1)
  print(best_epochs)

  with open('eval.%d.txt'%gpuid, 'w') as fout:
    predict(python_file, model_cfg_file, path_cfg_file, best_epochs, gpuid)

    for best_epoch in best_epochs:
      predict_file = os.path.join(preddir, '%d-5.json'%best_epoch)

      out = eval(predict_file, gt_file)
      content = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f'%(
        out['bleu'][0]*100, out['bleu'][1]*100, out['bleu'][2]*100, out['bleu'][3]*100,
        out['meteor']*100, out['rouge']*100, out['cider']*100, out['spice']*100)
      # content = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f'%(
      #   out['bleu'][0]*100, out['bleu'][1]*100, out['bleu'][2]*100, out['bleu'][3]*100,
      #   out['meteor']*100, out['rouge']*100, out['cider']*100)
      print(best_epoch)
      print(content)
      fout.write(str(best_epoch) + '\t' + content + '\n')


def predict_decode():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/data1/jiac/MSCOCO' # uranus
  root_dir = '/hdd/mscoco' # aws
  # root_dir = '/mnt/data1/jiac/mscoco' # neptune

  # model_name = 'pytorch/vevd_ml_expr/tf_resnet152_450.512.512.0.lstm'
  # model_name = 'pytorch/vevd_sc_expr/tf_resnet152_450.512.512.0.lstm'
  # model_name = 'pytorch/vevd_rl_expr/tf_resnet152_450.512.512.0.0.lstm'
  # python_file = 'vevd_ml_decode.py'

  # model_name = 'pytorch/vevd_gan_simple_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80'
  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.50.5.5.0.80.5.0'
  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.5.0'
  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.1.0'
  # model_name = 'pytorch/vevd_gan_simple_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.2.5'
  # python_file = 'gan_simple_sc_decode.py'

  # model_name = 'pytorch/vevd_gan_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.5.0'
  # model_name = 'pytorch/vevd_gan_sc_expr/tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8'
  # model_name = 'pytorch/vevd_gan_cider_sc_expr/tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.1.0'
  # python_file = 'gan_sc_decode.py'

  # model_name = 'pytorch/vead_gan_cider_sc_expr/bottomup.512.512.512.512.2048.add.mean.64.8.5.0.80.1.0.8.5.0'
  # python_file = 'vead_gan_cider_sc.py'

  model_name = 'pytorch/vead_gan_simple_cider_sc_expr/bottomup.512.512.512.512.2048.add.64.8.5.0.80.5.0'
  python_file = 'vead_gan_simple_sc.py'

  logdir = os.path.join(root_dir, model_name, 'log')
  preddir = os.path.join(root_dir, model_name, 'pred')
  model_cfg_file = os.path.join(root_dir, model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, model_name + '.path.json')

  gpuid = 3
  best_epochs = [23]

  predict(python_file, model_cfg_file, path_cfg_file, best_epochs, gpuid, 
    strategy='beam', beam_width=100, pool_size=100)
  # predict(python_file, model_cfg_file, path_cfg_file, best_epochs, gpuid, 
  #   strategy='sample_topk', sample_topk=10, num_sample=100)
  # predict(python_file, model_cfg_file, path_cfg_file, best_epochs, gpuid, 
  #   strategy='nucleus_sample', threshold_p=.9, num_sample=100)


def gather_predict_score():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/data1/jiac/MSCOCO' # uranus
  root_dir = '/hdd/mscoco' # aws
  # root_dir = '/mnt/data1/jiac/mscoco' # neptune

  topk = 100

  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm', 'pred')
  # epoch = 38
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_sc_expr', 'tf_resnet152_450.512.512.0.lstm', 'pred')
  # epoch = 48
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_rl_expr', 'tf_resnet152_450.512.512.0.0.lstm', 'pred')
  # epoch = 36
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80', 'pred')
  # epoch = 35
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.5.0', 'pred')
  # epoch = 40
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.50.5.5.0.80.5.0', 'pred')
  # epoch = 39
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.5.0', 'pred')
  # epoch = 16
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.1.0', 'pred')
  # epoch = 49
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.2.5', 'pred')
  # epoch = 18
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8', 'pred')
  # epoch = 4
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.2.5', 'pred')
  # epoch = 21
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.1.0', 'pred')
  # epoch = 44

  # pred_dir = os.path.join(root_dir, 'pytorch', 'pure_vead_ml_expr', 'bottomup.512.512.512.512.2048.1.0.att2in_boom.add', 'pred')
  # epoch = 11
  # pred_dir = os.path.join(root_dir, 'pytorch', 'pure_vead_sc_expr', 'bottomup.512.512.512.512.2048.1.0.1.att2in_boom.add', 'pred')
  # epoch = 31
  # pred_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_cider_sc_expr', 'bottomup.512.512.512.512.2048.add.mean.64.8.5.0.80.1.0.8.5.0', 'pred')
  # epoch = 30
  pred_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_simple_cider_sc_expr', 'bottomup.512.512.512.512.2048.add.64.8.5.0.80.5.0', 'pred')
  epoch = 23

  # pred_files = [
  # #   os.path.join(pred_dir, '%d-beam-50-50.json'%epoch),
  #   os.path.join(pred_dir, '%d-nucleus_sample-0.80-50.json'%epoch),
  #   os.path.join(pred_dir, '%d-sample_topk-5-50.json'%epoch),
  # ]
  # # out_file = os.path.join(pred_dir, '%d-beam-50-50-nucleus_sample-0.90-50-sample_topk-10-50.json'%epoch)
  # out_file = os.path.join(pred_dir, '%d-nucleus_sample-0.80-50-sample_topk-5-50.json'%epoch)

  # pred_files = [
  #   os.path.join(pred_dir, '%d-beam-50-50.json'%epoch),
  #   os.path.join(pred_dir, '%d-nucleus_sample-0.80-50.json'%epoch),
  #   os.path.join(pred_dir, '%d-sample_topk-5-50.json'%epoch),
  # ]
  # out_file = os.path.join(pred_dir, '%d-beam-50-50-nucleus_sample-0.80-50-sample_topk-5-50.json'%epoch)

  pred_files = [
    # os.path.join(pred_dir, '%d-nucleus_sample-0.90-100.json'%epoch),
    # os.path.join(pred_dir, '%d-nucleus_sample-0.80-100.json'%epoch),
    # os.path.join(pred_dir, '%d-sample_topk-5-100.json'%epoch),
    # os.path.join(pred_dir, '%d-sample_topk-10-100.json'%epoch),
    os.path.join(pred_dir, '%d-beam-100-100.json'%epoch),
  ]
  # out_file = os.path.join(pred_dir, '%d-nucleus_sample-0.90-100.gather.json'%epoch)
  # out_file = os.path.join(pred_dir, '%d-nucleus_sample-0.80-100.gather.json'%epoch)
  # out_file = os.path.join(pred_dir, '%d-sample_topk-5-100.gather.json'%epoch)
  # out_file = os.path.join(pred_dir, '%d-sample_topk-10-100.gather.json'%epoch)
  out_file = os.path.join(pred_dir, '%d-beam-100-100.gather.json'%epoch)

  vid2hash2sent_scores = {} 
  for pred_file in pred_files:
    with open(pred_file) as f:
      data = json.load(f)
      for vid in data:
        for d in data[vid]:
          sent = d['sent']
          hash = md5.new(sent).digest()

          if vid not in vid2hash2sent_scores:
            vid2hash2sent_scores[vid] = {}
          if hash in vid2hash2sent_scores[vid]:
            continue
          vid2hash2sent_scores[vid][hash] = d

  vid2sent_scores = {}
  for vid in vid2hash2sent_scores:
    hash2sent_scores = vid2hash2sent_scores[vid]
    sent_scores = hash2sent_scores.values()
    sent_scores = sorted(sent_scores, key=lambda x:x['score'], reverse=True)
    vid2sent_scores[vid] = sent_scores[:topk]

  with open(out_file, 'w') as fout:
    json.dump(vid2sent_scores, fout, indent=2)


def eval_precision_recall():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/data1/jiac/MSCOCO' # uranus
  root_dir = '/hdd/mscoco' # aws
  # root_dir = '/mnt/data1/jiac/mscoco' # neptune
  gt_file = os.path.join(root_dir, 'aux', 'human_caption_dict.pkl')

  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm')
  # epoch = 38
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_sc_expr', 'tf_resnet152_450.512.512.0.lstm')
  # epoch = 48
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_rl_expr', 'tf_resnet152_450.512.512.0.0.lstm')
  # epoch = 36
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80')
  # epoch = 35
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.50.5.5.0.80.5.0')
  # epoch = 39
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.5.0')
  # epoch = 40
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.5.0')
  # epoch = 16
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.1.0')
  # epoch = 49
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.2.5')
  # epoch = 18
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8')
  # epoch = 4
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.2.5')
  # epoch = 21
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.1.0')
  # epoch = 44

  # expr_dir = os.path.join(root_dir, 'pytorch', 'pure_vead_ml_expr', 'bottomup.512.512.512.512.2048.1.0.att2in_boom.add')
  # epoch = 11
  # expr_dir = os.path.join(root_dir, 'pytorch', 'pure_vead_sc_expr', 'bottomup.512.512.512.512.2048.1.0.1.att2in_boom.add')
  # epoch = 31
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_cider_sc_expr', 'bottomup.512.512.512.512.2048.add.mean.64.8.5.0.80.1.0.8.5.0')
  # epoch = 30
  expr_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_simple_cider_sc_expr', 'bottomup.512.512.512.512.2048.add.64.8.5.0.80.5.0')
  epoch = 23

  pred_files = [
    os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-nucleus_sample-0.90-100.gather.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-nucleus_sample-0.80-100.gather.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-sample_topk-5-100.gather.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-sample_topk-10-100.gather.json'%epoch),
  ]
  out_precision_files = [
    os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.precision.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-nucleus_sample-0.90-100.gather.precision.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-nucleus_sample-0.80-100.gather.precision.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-sample_topk-5-100.gather.precision.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-sample_topk-10-100.gather.precision.json'%epoch),
  ]
  out_recall_files = [
    os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.recall.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-nucleus_sample-0.90-100.gather.recall.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-nucleus_sample-0.80-100.gather.recall.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-sample_topk-5-100.gather.recall.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-sample_topk-10-100.gather.recall.json'%epoch),
  ]
  out_corpus_recall_files = [
    os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.corpus_recall.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-nucleus_sample-0.90-100.gather.corpus_recall.json'%epoch),
    # os.path.join(expr_dir, 'pred', '%d-sample_topk-10-100.gather.corpus_recall.json'%epoch),
  ]

  for pred_file, out_precision_file, out_recall_file, out_corpus_recall_file in zip(pred_files, out_precision_files, out_recall_files, out_corpus_recall_files):
    with open(pred_file) as f:
      vid2sent_scores = json.load(f)

    num = 0
    for vid in vid2sent_scores:
      num+= len(vid2sent_scores[vid])
    num /= len(vid2sent_scores)
    print(num)

    with open(gt_file) as f:
      vid2gt = pickle.load(f)

    precisions = eval_precision(vid2sent_scores, vid2gt, num)
    with open(out_precision_file, 'w') as fout:
      json.dump(precisions, fout)

    recalls = eval_recall(vid2sent_scores, num)
    with open(out_recall_file, 'w') as fout:
      json.dump(recalls, fout)

    recalls = eval_corpus_recall(vid2sent_scores, num)
    with open(out_corpus_recall_file, 'w') as fout:
      json.dump(recalls, fout)


def eval_precision_by_sent():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/data1/jiac/MSCOCO' # uranus
  root_dir = '/hdd/mscoco' # aws
  gt_file = os.path.join(root_dir, 'aux', 'human_caption_dict.pkl')

  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm')
  # epoch = 38
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_sc_expr', 'tf_resnet152_450.512.512.0.lstm')
  # epoch = 48
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80')
  # epoch = 35

  # expr_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_cider_sc_expr', 'bottomup.512.512.512.512.2048.add.mean.64.8.5.0.80.1.0.8.5.0')
  # epoch = 30
  expr_dir = os.path.join(root_dir, 'pytorch', 'vead_gan_simple_cider_sc_expr', 'bottomup.512.512.512.512.2048.add.64.8.5.0.80.5.0')
  epoch = 23

  pred_file = os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.json'%epoch)
  out_file = os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.precision_detail.json'%epoch)

  with open(pred_file) as f:
    vid2sent_scores = json.load(f)

  with open(gt_file) as f:
    vid2gt = pickle.load(f)

  num = 0
  for vid in vid2sent_scores:
    num+= len(vid2sent_scores[vid])
  num /= len(vid2sent_scores)
  print(num)

  vid2out = eval_precision_detail(vid2sent_scores, vid2gt, num)

  with open(out_file, 'w') as fout:
    json.dump(vid2out, fout)


def eval_precision_only():
  # root_dir = '/data1/jiac/mscoco' # mercurial
  # root_dir = '/data1/jiac/MSCOCO' # uranus
  root_dir = '/hdd/mscoco' # aws
  # root_dir = '/mnt/data1/jiac/mscoco' # neptune
  gt_file = os.path.join(root_dir, 'aux', 'human_caption_dict.pkl')

  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_ml_expr', 'tf_resnet152_450.512.512.0.lstm')
  # epoch = 38
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_sc_expr', 'tf_resnet152_450.512.512.0.lstm')
  # epoch = 48
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_rl_expr', 'tf_resnet152_450.512.512.0.0.lstm')
  # epoch = 36
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80')
  # epoch = 35
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.50.5.5.0.80.5.0')
  # epoch = 39
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.5.0')
  # epoch = 40
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_simple_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.5.50.5.0.80.5.0')
  # epoch = 16
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8')
  # epoch = 4
  # expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.2.5')
  # epoch = 21
  expr_dir = os.path.join(root_dir, 'pytorch', 'vevd_gan_cider_sc_expr', 'tf_resnet152_450.512.512.0.lstm.mean.5.50.5.0.80.1.0.8.1.0')
  epoch = 44

  pred_files = [
    os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.json'%epoch),
  ]
  out_precision_files = [
    os.path.join(expr_dir, 'pred', '%d-beam-100-100.gather.spice.json'%epoch),
  ]

  for pred_file, out_precision_file in zip(pred_files, out_precision_files):
    with open(pred_file) as f:
      vid2sent_scores = json.load(f)

    num = 0
    for vid in vid2sent_scores:
      num+= len(vid2sent_scores[vid])
    num /= len(vid2sent_scores)
    print(num)

    with open(gt_file) as f:
      vid2gt = pickle.load(f)

    precisions = eval_spice(vid2sent_scores, vid2gt, num)
    with open(out_precision_file, 'w') as fout:
      json.dump(precisions, fout)


def predict_eval_discriminator():
  root_dir = '/data1/jiac/mscoco' # mercurial
  
  model_name = 'pytorch/simple_discriminator/tf_resnet152_450.5.50.512.lstm'
  python_file = 'simple_discriminator.py'

  logdir = os.path.join(root_dir, model_name, 'log')
  preddir = os.path.join(root_dir, model_name, 'pred')
  model_cfg_file = os.path.join(root_dir, model_name + '.model.json')
  path_cfg_file = os.path.join(root_dir, model_name + '.path.json')

  gpuid = 0

  best_epoch = auto_select_discriminator(logdir)
  print(best_epoch)

  predict(python_file, model_cfg_file, path_cfg_file, [best_epoch], gpuid)


def fuse_precision():
  root_dir = '/home/jiac/data/precision_recall' # earth

  prefix = os.path.join(root_dir, 'vevd_ml', '38-beam-100-100.gather')

  precision_file = prefix + '.precision.json'
  spice_file = prefix + '.spice.json'

  with open(precision_file) as f:
    out_data = json.load(f)
  with open(spice_file) as f:
    data = json.load(f)
  out_data['spice'] = data['spice']
  with open(precision_file, 'w') as fout:
    json.dump(out_data, fout)


def eval_human():
  root_dir = '/hdd/mscoco' # aws
  vid_file = os.path.join(root_dir, 'split', 'tst_videoids.npy')
  gt_file = os.path.join(root_dir, 'aux', 'human_caption_dict.pkl')

  vids = np.load(vid_file)

  with open(gt_file) as f:
    data = pickle.load(f)

  vid2gts = {}
  for vid in vids:
    vid2gts[vid] = data[vid]

  vid2predict = {}
  for vid in vid2gts:
    vid2predict[vid] = vid2gts[vid][:1]

  bleu_scorer = Bleu(4)
  meteor_scorer = Meteor()
  rouge_scorer = Rouge()
  cider_scorer = Cider()
  spice_scorer = Spice()

  res_bleu, _ = bleu_scorer.compute_score(vid2gts, vid2predict)
  # metero handles the multi references (don't know the details yet)
  res_meteor, _ = meteor_scorer.compute_score(vid2gts, vid2predict)
  meteor_scorer.meteor_p.kill()
  # average
  res_rouge, _ = rouge_scorer.compute_score(vid2gts, vid2predict)
  # average
  res_cider, _ = cider_scorer.compute_score(vid2gts, vid2predict)
  res_spice, _ = spice_scorer.compute_score(vid2gts, vid2predict)

  upper_content = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f'%(
        res_bleu[0]*100, res_bleu[1]*100, res_bleu[2]*100, res_bleu[3]*100,
        res_meteor*100, res_rouge*100, res_cider*100, res_spice*100)
  print upper_content

  vid2leave_one_out = {}
  for vid in vid2gts:
    vid2leave_one_out[vid] = vid2gts[vid][1:]

  bleu_scorer = Bleu(4)
  meteor_scorer = Meteor()
  rouge_scorer = Rouge()
  cider_scorer = Cider()
  spice_scorer = Spice()

  res_bleu, _ = bleu_scorer.compute_score(vid2leave_one_out, vid2predict)
  # metero handles the multi references (don't know the details yet)
  res_meteor, _ = meteor_scorer.compute_score(vid2leave_one_out, vid2predict)
  meteor_scorer.meteor_p.kill()
  # average
  res_rouge, _ = rouge_scorer.compute_score(vid2leave_one_out, vid2predict)
  # average
  res_cider, _ = cider_scorer.compute_score(vid2leave_one_out, vid2predict)
  res_spice, _ = spice_scorer.compute_score(vid2leave_one_out, vid2predict)

  lower_content = '%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f'%(
        res_bleu[0]*100, res_bleu[1]*100, res_bleu[2]*100, res_bleu[3]*100,
        res_meteor*100, res_rouge*100, res_cider*100, res_spice*100)
  print lower_content

  with open('eval.txt', 'w') as fout:
    fout.write(lower_content + '\n')
    fout.write(upper_content + '\n')


if __name__ == '__main__':
  # predict_eval()
  # predict_decode()
  gather_predict_score()
  eval_precision_recall()
  # eval_precision_only()
  # predict_eval_discriminator()
  # fuse_precision()
  # eval_human()
  # eval_precision_by_sent()
