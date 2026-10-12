"""Offline check: does aggregation still help on the interface a reader would choose? (no model calls)
Every candidate has two calls (rep 0, rep 1) on the 30 screening-pool sources of each dataset. For each cell x candidate:
  single  = mean of the two calls
  inter2  = records supported by both calls (two-call majority), fused coordinates
  union2  = records from either call, fused where both report
Correspondence is label-free and lexicographic: maximum number of pairs first (IoU >= 0.5, same class; FSC points within tau),
then the largest total IoU among maximum matchings. Metrics: primary set error E, mean per-image F1, TP/FP/FN.
Also: L-min three-call support rule on the same 30 sources, to calibrate the two-call result against three calls.
Best interface = lowest 60-source mean single-call error (and, separately, highest mean F1), as a reader would choose with labels."""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
import analyze_e3 as A0
import aggregate_v2 as G
from score_e3 import load_items

ROOT = Path(__file__).parent


def corr_pairs_lex(it, base, P):
    if not base or not P: return []
    if it['task'] == 'point':
        return L.match_points(L._pts(base, it['labels']), L._pts(P, it['labels']), it['tau_point'])
    M = L.iou_matrix([b['box'] for b in base], [p['box'] for p in P])
    ok = (M >= 0.5) & np.array([[b['label'] == p['label'] for p in P] for b in base])
    W = np.where(ok, 1000.0 + M, 0.0)                    # lexicographic: cardinality first, IoU second
    r, c = linear_sum_assignment(-W)
    return [(int(i), int(j)) for i, j in zip(r, c) if ok[i, j]]


G.corr_pairs = corr_pairs_lex                            # build_clusters uses the module-level name


def scores(it, P, mt):
    E, tp, cnt = L.error(it, P, mt); n = it['n']
    return dict(E=E, F1=(2 * tp / (n + cnt) if n + cnt else 1.0), TP=tp, FP=cnt - tp, FN=n - tp)


def main():
    items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl')
    pool = json.loads((ROOT / 'runs/screening_pool.json').read_text())
    cell = defaultdict(dict)
    for x in rows:
        if x['block'] in ('MAIN', 'AGG'): cell[(x['dataset'], x['model'])][(x['cand'], x['item'], x['rep'])] = x
    full = json.loads((ROOT / 'runs/table_v2.json').read_text())
    out = {}
    for (ds, m), c in sorted(cell.items()):
        scr = sorted(pool[ds]); res = {}
        for cand in A0.CANDS:
            per = []
            for i in scr:
                it = items[i]; mt = A0.metric_of(it, A0.primary(it))
                reps = [A0.recs_of(c[(cand, i, r)]) for r in (0, 1)]
                cl = G.build_clusters(it, reps)
                s1 = [scores(it, P, mt) for P in reps]
                d = {f'single_{k}': float(np.mean([s[k] for s in s1])) for k in s1[0]}
                for name, sel in (('inter2', [G.fuse(x[1]) for x in cl if len(x[2]) == 2]), ('union2', [G.fuse(x[1]) for x in cl])):
                    for k, v in scores(it, sel, mt).items(): d[f'{name}_{k}'] = v
                if cand == 'L-min' and all((cand, i, r) in c for r in (0, 1, 2)):
                    reps3 = [A0.recs_of(c[(cand, i, r)]) for r in (0, 1, 2)]; cl3 = G.build_clusters(it, reps3)
                    s3 = [scores(it, P, mt) for P in reps3]
                    d['single3_E'] = float(np.mean([s['E'] for s in s3])); d['single3_F1'] = float(np.mean([s['F1'] for s in s3]))
                    for kmin in (1, 2, 3):
                        sc = scores(it, [G.fuse(x[1]) for x in cl3 if len(x[2]) >= kmin], mt); d[f'k{kmin}of3_E'] = sc['E']; d[f'k{kmin}of3_F1'] = sc['F1']
                per.append(d)
            agg = {k: round(float(np.mean([x[k] for x in per])), 4 if 'F1' in k else 2) for k in per[0]}
            for name in ('inter2', 'union2'):
                agg[f'{name}_gainE_ci'] = A0.boot([x['single_E'] - x[f'{name}_E'] for x in per])
                agg[f'{name}_gainF1_ci'] = A0.boot([x[f'{name}_F1'] - x['single_F1'] for x in per])
            res[cand] = agg
        k60 = {cand: full[f'{ds}|{m}|{cand}'] for cand in A0.CANDS}
        bestE = min(A0.CANDS, key=lambda q: k60[q]['E']); bestF = max(A0.CANDS, key=lambda q: k60[q]['F1'])
        out[f'{ds}|{m}'] = dict(best_by_E=bestE, best_by_F1=bestF, candidates=res)
        print(ds, m, 'bestE', bestE, 'bestF1', bestF, flush=True)
    (ROOT / 'runs/offline_agg2.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
