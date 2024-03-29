import copy
import numpy as np
import math
import time
import pickle


def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = {}
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      if ngram not in counts:
        counts[ngram] = 0
      counts[ngram] += 1
  return counts


def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
  '''Takes a list of reference sentences for a single segment
  and returns an object that encapsulates everything that BLEU
  needs to know about them.
  :param refs: list of string : reference sentences for some image
  :param n: int : number of ngrams for which (ngram) representation is calculated
  :return: result (list of dict)
  '''
  return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
  '''Takes a test sentence and returns an object that
  encapsulates everything that BLEU needs to know about it.
  :param test: list of string : hypothesis sentence for some image
  :param n: int : number of ngrams for which (ngram) representation is calculated
  :return: result (dict)
  '''
  return precook(test, n, True)


class CiderScorer(object):
  """CIDEr scorer.
  """
  def __init__(self, n=4, sigma=6.0):
    ''' singular instance '''
    self.n = n
    self.sigma = sigma
    self.vid2refs = {}
    self.document_frequency = {}
    self.ref_len = None

    self.delta_smooth_tab ={}
    for i in range(0, 100):
      self.delta_smooth_tab[i] = np.e**(-(float(i)**2)/(2*self.sigma**2))

  def load(self, df_file, gt_file):
    with open(df_file, 'rb') as f:
      self.document_frequency = pickle.load(f)
    with open(gt_file, 'rb') as f:
      self.vid2refs = pickle.load(f)
    cnt = 0.
    for vid in self.vid2refs:
      cnt += len(self.vid2refs[vid])
    self.ref_len = np.log(cnt)

  def _counts2vec(self, cnts):
    """
    Function maps counts of ngram to vector of tfidf weights.
    The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
    The n-th entry of array denotes length of n-grams.
    :param cnts:
    :return: vec (array of dict), norm (array of float), length (int)
    """
    vec = [{} for _ in range(self.n)]
    length = 0
    norm = [0.0 for _ in range(self.n)]
    for (ngram,term_freq) in cnts.items():
      # give word count 1 if it doesn't appear in reference corpus
      if ngram not in self.document_frequency:
        df = 0.
      else:
        df = self.document_frequency[ngram]
      # ngram index
      n = len(ngram)-1
      # tf (term_freq) * idf (precomputed idf) for n-grams
      vec[n][ngram] = float(term_freq)*(self.ref_len - df)
      # compute norm for the vector.  the norm will be used for computing similarity
      norm[n] += pow(vec[n][ngram], 2)

      if n == 1:
        length += term_freq
    norm = [np.sqrt(n) for n in norm]
    return vec, norm, length

  def _sim(self, vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
    '''
    Compute the cosine similarity of two vectors.
    :param vec_hyp: array of dictionary for vector corresponding to hypothesis
    :param vec_ref: array of dictionary for vector corresponding to reference
    :param norm_hyp: array of float for vector corresponding to hypothesis
    :param norm_ref: array of float for vector corresponding to reference
    :param length_hyp: int containing length of hypothesis
    :param length_ref: int containing length of reference
    :return: array of score for each n-grams cosine similarity
    '''
    delta = abs(length_hyp - length_ref)
    # measure consine similarity
    val = np.array([0.0 for _ in range(self.n)])
    for n in range(self.n):
      # ngram
      for (ngram,count) in vec_hyp[n].items():
        # vrama91 : added clipping
        if ngram in vec_ref[n]:
          val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

      if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
        val[n] /= (norm_hyp[n]*norm_ref[n])

      assert(not math.isnan(val[n]))
      # vrama91: added a length based gaussian penalty
      # val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
      val[n] *= self.delta_smooth_tab[delta]
    return val

  def compute_cider(self, tst, vids):
    self.ctest = []
    for d in tst:
      self.ctest.append(cook_test(d))

    scores = np.zeros(len(vids))
    idx = 0
    for test, vid in zip(self.ctest, vids):
      # compute vector for test captions
      vec, norm, length = self._counts2vec(test)

      # compute vector for ref captions
      score = np.array([0.0 for _ in range(self.n)])
      refs = self.vid2refs[vid]
      for ref in refs:
        cref = cook_test(ref)
        vec_ref, norm_ref, length_ref = self._counts2vec(cref)
        score += self._sim(vec, vec_ref, norm, norm_ref, length, length_ref)
      score_avg = np.mean(score)
      # divide by number of references
      score_avg /= len(refs)
      # multiply score by 10
      score_avg *= 10.0
      # append score of an image to the score list
      scores[idx] = score_avg
      idx += 1
    return np.mean(scores), scores
