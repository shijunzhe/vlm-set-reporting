"""Own box precision of each model on E3 SKU / DOTA (L-min, three repeats): median IoU between a record and its best match in
another repeat of the same image (records matched at IoU >= 0.3). If the model's own repeat-to-repeat IoU is near the
0.75 criterion, a comparator built from the same model cannot tell IoU 0.75 boxes from IoU 0.5 boxes. No model calls.
Usage: python -I box_jitter.py"""
import json, sys, collections
import numpy as np
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import analyze_e3 as A0
from score_e3 import load_items
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl')
d3 = {(x['item'], x['model'], x['rep']): x for x in rows if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min'}
out = {}
for ds in ('sku110k', 'dota'):
    for m in ('gpt61', 'sonnet55', 'gem38'):
        ious = []; gt_ious = []
        for (i, mm, k), x in d3.items():
            if mm != m or k != 0 or items[i].get('dataset', ds) != ds or not i.startswith(ds[:3]): continue
            if (i, m, 1) not in d3: continue
            A = [r['box'] for r in A0.recs_of(x) if r['box'] is not None]; B = [r['box'] for r in A0.recs_of(d3[(i, m, 1)]) if r['box'] is not None]
            if not A or not B: continue
            M = G.iou_mat(A, B)
            best = M.max(1); ious += [float(v) for v in best if v >= 0.3]
        if ious:
            a = np.array(ious); out[f'{ds}|{m}'] = dict(pairs=len(a), median_iou=round(float(np.median(a)), 3), share_ge_075=round(float(np.mean(a >= 0.75)), 3))
json.dump(out, open('/home/claude/stageB/offline/box_jitter.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
