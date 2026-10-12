"""E3 aggregation v2: label-free correspondence, support filtering separated from geometric fusion. No model calls.

Correspondence (no ground truth is read):
  FSC-147 (points): centres within tau, where tau = half the mean exemplar short side; the exemplar boxes are task input in the
                    FSC-147 protocol (the reference crop is sent to the model).
  boxes:            maximum matching with IoU >= 0.5 and the same class.
Clusters are built over the three L-min calls; a cluster's support k = number of calls that contain it.
Variants, for k_min in {1 (union), 2 (majority), 3 (intersection)}:
  rep   keep the member record from the earliest call (no geometric change for that call's records)
  fuse  average the members' coordinates (centre and box)
Outputs per cell (primary metric): mean error of single calls and of each variant, bootstrap CIs over sources for
filtering gain (rep - single), fusion gain (fuse - rep), and total; support counts N_k and support precision p_k
(precision of representative records, primary metric); the identity check; a label-free dispersion statistic for fusion;
and the guarded support rule (p_k from 10 labelled sources, +-2 cluster SE, fusion on unless the labels show harm; evaluated on the 50 unused sources).
Usage: [E3_TAG=...] python aggregate_v2.py"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
import analyze_e3 as A0
from score_e3 import load_items, TAG

ROOT = Path(__file__).parent


def corr_pairs(it, base, P):
    """label-free correspondence between base representatives and a new report"""
    if not base or not P: return []
    if it['task'] == 'point':
        return L.match_points(L._pts(base, it['labels']), L._pts(P, it['labels']), it['tau_point'])
    M = L.iou_matrix([b['box'] for b in base], [p['box'] for p in P])
    ok = (M >= 0.5) & np.array([[b['label'] == p['label'] for p in P] for b in base])
    r, c = linear_sum_assignment(-(ok * (1 + M)))
    return [(int(i), int(j)) for i, j in zip(r, c) if ok[i, j]]


def build_clusters(it, reps):
    cl = []                                                     # [rep_record, members(list), support set]
    for ri, P in enumerate(reps):
        P = [p for p in P if it['task'] == 'point' or p['box'] is not None]
        pairs = corr_pairs(it, [c[0] for c in cl], P); inv = {j: i for i, j in pairs}
        for j, p in enumerate(P):
            if j in inv: cl[inv[j]][1].append(p); cl[inv[j]][2].add(ri)
            else: cl.append([p, [p], {ri}])
    return cl


def fuse(members):
    out = dict(cx=float(np.mean([r['cx'] for r in members])), cy=float(np.mean([r['cy'] for r in members])), label=members[0]['label'], box=None)
    if members[0]['box'] is not None: out['box'] = [float(np.mean([r['box'][k] for r in members])) for k in range(4)]
    return out


def dispersion(it, members):
    f = fuse(members)
    if it['task'] == 'point' or members[0]['box'] is None:
        return float(np.mean([math.hypot(r['cx'] - f['cx'], r['cy'] - f['cy']) for r in members])) / it['tau_point']
    return float(np.mean([1 - L.iou_matrix([r['box']], [f['box']])[0, 0] for r in members]))


def main():
    items = load_items(); rows = A0.load_rows(f'runs_main/scored{TAG}.jsonl')
    cell = defaultdict(dict)
    for x in rows:
        if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min': cell[(x['dataset'], x['model'])][(x['item'], x['rep'])] = x
    out = {}
    for (ds, m), c in sorted(cell.items()):
        per = []
        for i in sorted({k[0] for k in c}):
            if not all((i, r) in c for r in (0, 1, 2)): continue
            it = items[i]; mt = A0.metric_of(it, A0.primary(it)); reps = [A0.recs_of(c[(i, r)]) for r in (0, 1, 2)]
            cl = build_clusters(it, reps)
            s = dict(single=float(np.mean([L.error(it, P, mt)[0] for P in reps])))
            for k in (1, 2, 3):
                sel = [x for x in cl if len(x[2]) >= k]
                s[f'rep{k}'] = L.error(it, [x[0] for x in sel], mt)[0]; s[f'fuse{k}'] = L.error(it, [fuse(x[1]) for x in sel], mt)[0]
            reprs = [x[0] for x in cl]
            if mt == 'point':
                Y = [(((t['point'][0] if 'point' in t else (t['box'][0] + t['box'][2]) / 2) + L.LABEL_OFFSET * it['labels'].index(t['label'])),
                      (t['point'][1] if 'point' in t else (t['box'][1] + t['box'][3]) / 2)) for t in it['targets']]
                matched = {a for a, _ in L.match_points(L._pts(reprs, it['labels']), Y, it['tau_point'])} if reprs and Y else set()
            else:
                matched = {a for a, _ in L.match_boxes(reprs, [(t['box'], t['label']) for t in it['targets']], mt[1])}
            for k in (1, 2, 3):
                J = [j for j, x in enumerate(cl) if len(x[2]) == k and (j in matched or not L.in_ignore(x[0], it['ignore']))]
                s[f'N{k}'] = len(J); s[f'T{k}'] = sum(j in matched for j in J)
            d = [dispersion(it, x[1]) for x in cl if len(x[2]) >= 2]
            s['disp'] = float(np.mean(d)) if d else 0.0
            per.append(s)
        n = len(per); g = lambda f: A0.boot([f(x) for x in per])
        S = {k: sum(x[k] for x in per) for k in per[0]}
        p = {k: (S[f'T{k}'] / S[f'N{k}'] if S[f'N{k}'] else None) for k in (1, 2, 3)}
        pred_rep2 = (S['N1'] - 2 * S['T1'] + 2 * S['T2'] - S['N2']) / 3 / n
        res = dict(sources=n, single=round(S['single'] / n, 2),
                   mean_error={f'{v}{k}': round(S[f'{v}{k}'] / n, 2) for v in ('rep', 'fuse') for k in (1, 2, 3)},
                   filtering_gain={k: g(lambda x, k=k: x['single'] - x[f'rep{k}']) for k in (1, 2, 3)},
                   fusion_gain={k: g(lambda x, k=k: x[f'rep{k}'] - x[f'fuse{k}']) for k in (1, 2, 3)},
                   total_gain={k: g(lambda x, k=k: x['single'] - x[f'fuse{k}']) for k in (1, 2, 3)},
                   N_per_source={k: round(S[f'N{k}'] / n, 1) for k in (1, 2, 3)}, p={k: (round(v, 3) if v is not None else None) for k, v in p.items()},
                   identity_majority_filter=dict(predicted=round(pred_rep2, 2), observed=round((S['single'] - S['rep2']) / n, 2)),
                   dispersion=round(S['disp'] / n, 3))
        # guarded support rule, evaluated on unused sources; fusion on by default, switched off only if the same labelled sources show harm beyond 2 SE
        rng = random.Random(f'agg2:{ds}:{m}'); out_rule = defaultdict(list); choice = defaultdict(int)
        for _ in range(400):
            idx = rng.sample(range(n), 10); sub = [per[j] for j in idx]; rest = [per[j] for j in range(n) if j not in idx]
            def ci(k):
                N = np.array([x[f'N{k}'] for x in sub], float); T = np.array([x[f'T{k}'] for x in sub], float)
                if N.sum() == 0: return 0.5, 1.0
                pp = T.sum() / N.sum(); return pp, math.sqrt(np.sum((T - pp * N) ** 2)) / N.sum()
            p1, s1 = ci(1); p2, s2 = ci(2)
            k = 1 if p1 - 2 * s1 > .5 else (3 if p2 + 2 * s2 < .5 else 2)
            fg = np.array([x[f'rep{k}'] - x[f'fuse{k}'] for x in sub], float)
            use_fuse = not (fg.mean() + 2 * fg.std(ddof=1) / math.sqrt(len(fg)) < 0)   # fuse by default (convexity, v5.7 Prop. 6); stop only on labelled evidence of harm
            v = 'fuse' if use_fuse else 'rep'; choice[f'{v}{k}'] += 1
            single = np.mean([x['single'] for x in rest])
            out_rule['rule'].append(single - np.mean([x[f'{v}{k}'] for x in rest]))
            out_rule['majority_fused'].append(single - np.mean([x['fuse2'] for x in rest]))
            out_rule['majority_rep'].append(single - np.mean([x['rep2'] for x in rest]))
        res['guarded_rule'] = dict(choice={k: round(v / 400, 3) for k, v in sorted(choice.items())},
                                   gain_unused=round(float(np.mean(out_rule['rule'])), 2),
                                   gain_unused_majority_fused=round(float(np.mean(out_rule['majority_fused'])), 2),
                                   gain_unused_majority_rep=round(float(np.mean(out_rule['majority_rep'])), 2),
                                   rule_minus_majority_fused_q05=round(float(np.percentile(np.array(out_rule['rule']) - np.array(out_rule['majority_fused']), 5)), 2))
        out[f'{ds}|{m}'] = res
        print(ds, m, res['single'], res['mean_error'], 'filter2', res['filtering_gain'][2][:3], 'fusion2', res['fusion_gain'][2][:3], 'p', res['p'],
              'id', res['identity_majority_filter'], 'disp', res['dispersion'], 'rule', res['guarded_rule'], flush=True)
    (ROOT / f'runs/aggregate_v2{TAG}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
