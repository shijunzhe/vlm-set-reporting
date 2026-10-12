"""Comparator (B3 blocks) against the label-free feature bound: does the comparator discriminate true from false
candidates better than the features already available without it (support 1 or 2, class label)? A gap larger than
TV(phi) means the comparator brings new evidence (allowed by the theory: it sees rendered panels). Truth at IoU 0.75
(the E3 box metric) and 0.5 (from b3_analysis). No model calls. Usage: python -I comparator_tv.py"""
import json, collections
import numpy as np
meta = json.load(open('/home/claude/stageB/runs/b3/items.json'))['block']; A = json.load(open('/home/claude/stageB/runs/b3/b3_analysis.json'))['block']
out = {}
for key, v in A.items():
    ds, m, thr = key.split('|')
    if thr != 'iou75': continue
    B = [b for bid, b in meta.items() if bid.split('|')[0] == ds and bid.split('|')[2] == m]
    tp = np.array([b['tp'] for b in B]); cell = [f"{b['support']}|{b['label']}" for b in B]
    keys = set(cell); c = np.array(cell)
    tv = 0.5 * sum(abs(np.mean(c[tp == 1] == k) - np.mean(c[tp == 0] == k)) for k in keys) if tp.any() and (~tp.astype(bool)).any() else None
    tv_sup = abs(np.mean([b['support'] == 2 for b in B if b['tp']]) - np.mean([b['support'] == 2 for b in B if not b['tp']])) if tp.any() and (1 - tp).any() else None
    gap = (v['tp_keep_rate'] - v['fp_keep_rate']) if v['tp_keep_rate'] is not None and v['fp_keep_rate'] is not None else None
    out[f'{ds}|{m}'] = dict(blocks=len(B), tp_rate75=round(float(tp.mean()), 3), comparator_gap75=None if gap is None else round(gap, 3),
                            tv_support=None if tv_sup is None else round(float(tv_sup), 3), tv_support_label_plugin=None if tv is None else round(float(tv), 3),
                            comparator_gap50=round(A[f'{ds}|{m}|iou50']['tp_keep_rate'] - A[f'{ds}|{m}|iou50']['fp_keep_rate'], 3))
json.dump(out, open('/home/claude/stageB/offline/comparator_tv.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
