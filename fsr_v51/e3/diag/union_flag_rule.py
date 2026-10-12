"""Revised aggregation rule: union only when the count gap comes with an observed completion flag on a shorter report
(the completion channel), otherwise majority. Also checks directly whether the objects union adds are mostly true when the
gap is flagged (completion) and mostly false when it is not."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import e3lib as L
import analyze_e3 as A
from score_e3 import load_items

items = load_items(); rows = A.load_rows('runs_main/scored.jsonl')
cell = defaultdict(dict)
for x in rows:
    if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min': cell[(x['dataset'], x['model'])][(x['item'], x['rep'])] = x
out = {}; pooled = defaultdict(list)
for (ds, m), c in sorted(cell.items()):
    its = [i for i in items.values() if i['dataset'] == ds]; M = max(1.0, 0.05 * float(np.mean([i['n'] for i in its])))
    g = defaultdict(list)
    for i in sorted({k[0] for k in c}):
        if not all((i, r) in c for r in (0, 1, 2)): continue
        it = items[i]; rr = [c[(i, r)] for r in (0, 1, 2)]; reps = [A.recs_of(x) for x in rr]; mt = A.metric_of(it, A.primary(it))
        order = sorted(range(3), key=lambda k: len(reps[k])); cnt = [len(reps[k]) for k in order]
        gap = (cnt[2] - cnt[1]) > M; nest = A.nestedness(it, reps) >= 0.9
        short_flag = any(rr[k]['flags'][f] for k in order[:2] for f in A.COMP)
        maj = A.majority(it, reps); uni = A.union(it, reps)
        Es = float(np.mean([L.error(it, P, mt)[0] for P in reps])); Em, tm, _ = L.error(it, maj, mt); Eu, tu, _ = L.error(it, uni, mt)
        old = Eu if (gap and nest) else Em; new = Eu if (gap and nest and short_flag) else Em
        g['single'].append(Es); g['majority'].append(Em); g['old'].append(old); g['new'].append(new)
        g['fire_old'].append(int(gap and nest)); g['fire_new'].append(int(gap and nest and short_flag))
        if gap and nest:
            extra = len(uni) - len(maj)
            if extra > 0: pooled['flagged' if short_flag else 'unflagged'].append((tu - tm) / extra)
    out[f'{ds}|{m}'] = dict(n=len(g['single']), **{k: round(float(np.mean(v)), 2) for k, v in g.items()},
                           gain_old=A.boot([s - r for s, r in zip(g['single'], g['old'])]), gain_new=A.boot([s - r for s, r in zip(g['single'], g['new'])]),
                           new_minus_majority=A.boot([r - mm for r, mm in zip(g['new'], g['majority'])]))
    print(ds, m, out[f'{ds}|{m}'])
out['extra_precision'] = {k: dict(n=len(v), mean=round(float(np.mean(v)), 3)) for k, v in pooled.items()}
print(out['extra_precision'])
(ROOT / 'diag/union_flag_rule.json').write_text(json.dumps(out, indent=1))
