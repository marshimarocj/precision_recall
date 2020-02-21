import os
import sys
import cPickle
import time
sys.path.append('../')

import numpy as np

import model.fast_cider


'''func
'''


'''expr
'''
def precompute_df():
  root_dir = '/data1/jiac/mscoco' # mercurial
  gt_file = os.path.join(root_dir, 'aux', 'human_caption_dict.pkl')
  out_file = os.path.join(root_dir, 'aux', 'document_frequency.pkl')

  with open(gt_file) as f:
    vid2captions = cPickle.load(f)

  cider = model.fast_cider.CiderScorer()
  cider.init_refs(vid2captions)
  document_frequency = cider.document_frequency

  slim_document_frequency = {}
  for ngram in document_frequency:
    if document_frequency[ngram] > 0.:
      slim_document_frequency[ngram] = document_frequency[ngram]

  with open(out_file, 'w') as fout:
    cPickle.dump(slim_document_frequency, fout)


if __name__ == '__main__':
  precompute_df()
