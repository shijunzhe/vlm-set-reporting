"""End-to-end held-out estimate of what the proposed procedure buys (no model calls).
Data: the 30 screening sources per dataset, where every interface has two calls. 400 random splits: 10 labelled sources
(used for every choice) and 20 evaluation sources (never used for choices). Objective declared in advance per run: set error E
(primary metric) or mean per-image F1.
Baselines (one call): API-default list (L-default, rep 0); cheapest list (L-min, rep 0); ENG = best of the 4 interfaces by the
labelled mean (rep 0).
Proposed (two calls on the chosen interface):
  emp     choose among 4 interfaces x {single, intersection, union} by the labelled mean of the declared objective
  theory  choose the interface by the labelled mean of single calls; then keep support-1 records (union) iff their labelled
          task-valid share p1 exceeds the loss threshold: 1/2 for E, F1(intersection)/2 for F1 (C1b), else intersection
Correspondence: label-free, lexicographic (offline_agg2.corr_pairs_lex)."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
import analyze_e3 as A0
import offline_agg2 as O                 # installs the lexicographic correspondence into aggregate_v2
import aggregate_v2 as G
from score_e3 import load_items

ROOT = Path(__file__).parent
items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl'); pool = json.loads((ROOT / 'runs/screening_pool.json').read_text())
cell = defaultdict(dict)
for x in rows:
    if x['block'] in ('MAIN', 'AGG'): cell[(x['dataset'], x['model'])][(x['cand'], x['item'], x['rep'])] = x
OUT = {}
for (ds, m), c in sorted(cell.items()):
    scr = sorted(pool[ds]); S = {}           # S[(cand, item)] = dict of option -> (E, F1, TP, r, n), plus p1 counts
    for cand in A0.CANDS:
        for i in scr:
            it = items[i]; mt = A0.metric_of(it, A0.primary(it)); reps = [A0.recs_of(c[(cand, i, r)]) for r in (0, 1)]
            cl = G.build_clusters(it, reps); n = it['n']
            def sc(P):
                E, tp, cnt = L.error(it, P, mt); return (E, (2 * tp / (n + cnt) if n + cnt else 1.0), tp, cnt, n)
            uni = [G.fuse(x[1]) for x in cl]; sup = [len(x[2]) for x in cl]
            if mt == 'point':
                Y = [(((t['point'][0] if 'point' in t else (t['box'][0] + t['box'][2]) / 2) + L.LABEL_OFFSET * it['labels'].index(t['label'])),
                      (t['point'][1] if 'point' in t else (t['box'][1] + t['box'][3]) / 2)) for t in it['targets']]
                ms = {a for a, _ in L.match_points(L._pts(uni, it['labels']), Y, it['tau_point'])} if uni and Y else set()
            else:
                ms = {a for a, _ in L.match_boxes(uni, [(t['box'], t['label']) for t in it['targets']], mt[1])}
            J1 = [j for j in range(len(uni)) if sup[j] == 1 and (j in ms or not L.in_ignore(uni[j], it['ignore']))]
            S[(cand, i)] = dict(single=sc(reps[0]), inter=sc([u for u, k in zip(uni, sup) if k == 2]), union=sc(uni),
                                N1=len(J1), T1=sum(j in ms for j in J1))
    rng = random.Random(f'e2e:{ds}:{m}'); res = defaultdict(lambda: defaultdict(list))
    for _ in range(400):
        lab = rng.sample(scr, 10); ev = [i for i in scr if i not in lab]
        mean = lambda cand, opt, idx, f: float(np.mean([S[(cand, i)][opt][f] for i in idx]))
        for obj, f, better in (('E', 0, min), ('F1', 1, max)):
            sgn = 1 if obj == 'E' else -1
            evalv = lambda cand, opt: mean(cand, opt, ev, f)
            eng = min(A0.CANDS, key=lambda q: sgn * mean(q, 'single', lab, f))
            emp = min(((q, o) for q in A0.CANDS for o in ('single', 'inter', 'union')), key=lambda qo: sgn * mean(qo[0], qo[1], lab, f))
            N1 = sum(S[(eng, i)]['N1'] for i in lab); T1 = sum(S[(eng, i)]['T1'] for i in lab); p1 = T1 / N1 if N1 else 0.0
            if obj == 'E': thr = 0.5
            else:
                tp = sum(S[(eng, i)]['inter'][2] for i in lab); r = sum(S[(eng, i)]['inter'][3] for i in lab); n = sum(S[(eng, i)]['inter'][4] for i in lab)
                thr = (tp / (n + r)) if n + r else 0.0            # = F1(intersection)/2, pooled
            th = (eng, 'union' if p1 > thr else 'inter')
            v = {'L-default': evalv('L-default', 'single'), 'L-min': evalv('L-min', 'single'), 'ENG': evalv(eng, 'single'),
                 'emp': evalv(*emp), 'theory': evalv(*th)}
            for k, x in v.items(): res[obj][k].append(x)
            res[obj]['emp_choice'].append(f'{emp[0]}|{emp[1]}'); res[obj]['theory_choice'].append(f'{th[0]}|{th[1]}')
    summ = {}
    for obj in ('E', 'F1'):
        r = res[obj]; d = {k: round(float(np.mean(r[k])), 4 if obj == 'F1' else 2) for k in ('L-default', 'L-min', 'ENG', 'emp', 'theory')}
        from collections import Counter
        d['emp_choice'] = dict(Counter(r['emp_choice']).most_common(3)); d['theory_choice'] = dict(Counter(r['theory_choice']).most_common(3))
        summ[obj] = d
    OUT[f'{ds}|{m}'] = summ
    print(ds, m, summ['E'], summ['F1'], flush=True)
(ROOT / 'runs/endtoend_v2.json').write_text(json.dumps(OUT, indent=1))
