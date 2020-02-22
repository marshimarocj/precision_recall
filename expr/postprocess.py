import os
import json


'''func
'''


'''expr
'''
def diversity_topk():
  root_dir = '/data1/jiac/mscoco/pytorch' # mercurial

  topk = 5
  alpha = 2.

  expr_name = 'vevd_sc_expr/tf_resnet152_450.512.512.0.lstm'

  gather_file = os.path.join(root_dir, expr_name, 'pred', '48-beam-100-100.gather.json')
  out_file = os.path.join(root_dir, expr_name, 'pred', '48-beam-100-100.gather.diverse.%d.json'%topk)

  with open(gather_file) as f:
    vid2sent_scores = json.load(f)

  vid2out = {}
  for vid in vid2sent_scores:
    sent_scores = vid2sent_scores[vid]

    selected = set()
    diverse_sent_scores = []
    for k in range(topk):
      best_score = -1e10
      best_sent = ''
      for d in sent_scores:
        sent = d['sent']
        score = d['score']
        if sent in selected:
          continue

        words = sent.split(' ')
        avg_diff = 0.
        for ref_sent in selected:
          ref_words = ref_sent.split(' ')
          cnt = 0
          for w1, w2 in zip(words, ref_words):
            if w1 == w2:
              cnt += 1
          m = max(len(words), len(ref_words))
          diff = (m - cnt ) / float(m)
          avg_diff += diff
        if len(selected) > 0:
          avg_diff /= len(selected)
        score += alpha * avg_diff
        if score > best_score:
          best_score = score
          best_sent = sent
      selected.add(best_sent)
      diverse_sent_scores.append({'sent': best_sent, 'score': best_score})

    vid2out[vid] = diverse_sent_scores
  with open(out_file, 'w') as fout:
    json.dump(vid2out, fout, indent=2)


if __name__ == '__main__':
  diversity_topk()
