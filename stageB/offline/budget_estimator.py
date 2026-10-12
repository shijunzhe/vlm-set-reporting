"""R27 label-free report-budget estimator on B2: ratio of records listed by four separate crop calls to records listed by
one whole-image call (no labels) against the change in true positives (labels, scoring only). Same data that suggested
R27, so this is a consistency check of the estimator, not an independent test. No model calls.
Usage: python -I budget_estimator.py"""
import json
from scipy.stats import spearmanr
d = json.load(open('/home/claude/stageB/runs/b2/b2_analysis.json'))['cells']
def tp_rec(x):
    r = x['recall']; tp = x['FN'] * r / (1 - r) if r < 1 else 0.0; return tp, tp + x['FP']
out = {}
for k, c in d.items():
    tf, rf = tp_rec(c['full_D1']); tc, rc = tp_rec(c['crops_blind'])
    out[k] = dict(record_ratio=round(rc / rf, 3), tp_full=round(tf, 1), tp_crops=round(tc, 1), tp_gain=round(tc - tf, 1))
ks = list(out); s = spearmanr([out[k]['record_ratio'] for k in ks], [out[k]['tp_gain'] for k in ks])
out['_spearman'] = dict(rho=round(float(s.correlation), 3), p=round(float(s.pvalue), 4), n=len(ks))
json.dump(out, open('/home/claude/stageB/offline/budget_estimator.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
