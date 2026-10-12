"""Why does the nested/gap union rule fail on real data for some models? For sources where the rule picks union, measure the
precision of the objects that union adds beyond the majority vote (truth used only here, for diagnosis)."""
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
out = {}
for (ds, m), c in sorted(cell.items()):
    its = [i for i in items.values() if i['dataset'] == ds]; M = max(1.0, 0.05 * float(np.mean([i['n'] for i in its])))
    res = defaultdict(list)
    for i in sorted({k[0] for k in c}):
        if not all((i, r) in c for r in (0, 1, 2)): continue
        it = items[i]; reps = [A.recs_of(c[(i, r)]) for r in (0, 1, 2)]
        cnt = sorted(len(P) for P in reps); uu = (cnt[2] - cnt[1]) > M and A.nestedness(it, reps) >= 0.9
        if not uu: continue
        maj = A.majority(it, reps); uni = A.union(it, reps)
        for name in (('primary', A.metric_of(it, A.primary(it))), ('iou50', A.metric_of(it, 'iou50')), ('point', 'point')):
            if it['task'] == 'point' and name[0] != 'primary': continue
            Em, tm, _ = L.error(it, maj, name[1]); Eu, tu, _ = L.error(it, uni, name[1])
            extra = len(uni) - len(maj)
            res[name[0]].append(dict(extra=extra, extra_prec=(tu - tm) / extra if extra > 0 else None, dE=Eu - Em))
        lo, hi = min(reps, key=len), max(reps, key=len)
        res['counts'].append(dict(n=it['n'], lo=len(lo), hi=len(hi), maj=len(maj), uni=len(uni)))
    if not res: continue
    summ = {}
    for k, v in res.items():
        if k == 'counts':
            summ[k] = {f: round(float(np.mean([x[f] for x in v])), 1) for f in v[0]}; summ['n_sources'] = len(v)
        else:
            ep = [x['extra_prec'] for x in v if x['extra_prec'] is not None]
            summ[k] = dict(mean_extra=round(float(np.mean([x['extra'] for x in v])), 1), extra_precision=round(float(np.mean(ep)), 3) if ep else None,
                           union_minus_majority=round(float(np.mean([x['dE'] for x in v])), 2))
    out[f'{ds}|{m}'] = summ; print(ds, m, summ)
(ROOT / 'diag/union_diag.json').write_text(json.dumps(out, indent=1))
