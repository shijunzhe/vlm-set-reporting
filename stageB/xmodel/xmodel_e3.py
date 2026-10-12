"""Cross-model fusion on E3 (LIST-min, 4 datasets x 3 models x 3 calls, the same 60 images per dataset). No model calls.
Labels are used only for scoring and for the diagnostic precisions.
Constructions (records clustered with the E3 correspondence; fused coordinates):
  K=2: same model A0 with A1 (intersection / union) vs A0 with B0 for each other model B
  K=3: same model A0, A1, A2 (unanimous / majority / union) vs one call from each model (unanimous / majority / union)
  K=6: two calls from each model, majority (>= 4 of 6) and 'model majority' (kept if >= 2 models support it)
  K=9: all calls, model majority
Baselines: single call of A; the best single model of the dataset (chosen with labels, a reference only); the best model's
same-model K=3 majority / unanimous.
Diagnostics (theory): p1 = precision of A0 records not confirmed by the partner (THM-INTERSECT predicts intersection helps iff p1 < 1/2);
confirmation rates of A0's true and false records by the partner; share of A's stable misses (missed in all three A calls)
recovered by the partner model; library ceiling (perfect selection from the union) for same-model vs cross-model K=3.
usage: python -I xmodel_e3.py"""
import json, sys, os, random, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np
E3 = Path('/home/claude/fsr_v51/e3'); sys.path.insert(0, str(E3)); os.chdir(E3)
import e3lib as L
import analyze_e3 as A0
import offline_agg2
import aggregate_v2 as G
from score_e3 import load_items
OUT = Path(__file__).parent / 'xmodel_e3.json'
M = ['gpt61', 'sonnet55', 'gem38']; PRICE = {'gpt61': (2, 10), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}
rng = np.random.default_rng(0)


def tpoint(it, t):
    return (((t['point'][0] if 'point' in t else (t['box'][0] + t['box'][2]) / 2) + L.LABEL_OFFSET * it['labels'].index(t['label'])),
            (t['point'][1] if 'point' in t else (t['box'][1] + t['box'][3]) / 2))


def matched(it, recs, mt):
    """which records are matched (one-to-one) to targets"""
    Y = it['targets']
    if not recs or not Y: return np.zeros(len(recs), bool)
    if mt == 'point':
        pr = L.match_points(L._pts(recs, it['labels']), [tpoint(it, t) for t in Y], it['tau_point'])
    else:
        pr = L.match_boxes(recs, [(t['box'], t['label']) for t in Y], mt[1])
    s = {a for a, _ in pr}; return np.array([j in s for j in range(len(recs))])


def tmatched(it, recs, mt):
    """which targets are matched"""
    Y = it['targets']
    if not recs or not Y: return np.zeros(len(Y), bool)
    if mt == 'point':
        pr = L.match_points(L._pts(recs, it['labels']), [tpoint(it, t) for t in Y], it['tau_point'])
    else:
        pr = L.match_boxes(recs, [(t['box'], t['label']) for t in Y], mt[1])
    s = {b for _, b in pr}; return np.array([j in s for j in range(len(Y))])


def ceiling(it, recs, mt):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    Y = it['targets']
    if not recs or not Y: return it['n']
    if mt == 'point':
        P = np.array(L._pts(recs, it['labels']), float); T = np.array([tpoint(it, t) for t in Y], float)
        R = np.hypot(P[:, None, 0] - T[None, :, 0], P[:, None, 1] - T[None, :, 1]) <= it['tau_point']
    else:
        R = (L.iou_matrix([a['box'] for a in recs], [t['box'] for t in Y]) >= mt[1] - 1e-12) & np.array([[a['label'] == t['label'] for t in Y] for a in recs])
    if not R.any(): return it['n']
    return it['n'] - int((maximum_bipartite_matching(csr_matrix(R.astype(np.int8)), perm_type='column') >= 0).sum())


def keep(cl, rule):
    """cl from build_clusters over a list of reps; rule(support_set) -> bool; fused output"""
    return [G.fuse(x[1]) for x in cl if rule(x[2])]


items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl')
R = defaultdict(dict); COST = defaultdict(list)
for x in rows:
    if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min':
        R[(x['dataset'], x['model'])][(x['item'], x['rep'])] = x
        COST[x['model']].append(((x.get('in_tok') or 0) * PRICE[x['model']][0] + (x.get('out_tok') or 0) * PRICE[x['model']][1]) / 1e6)
out = dict(per_cell={}, pairs={}, k3={}, cost_per_call={m: round(float(np.mean(v)), 5) for m, v in COST.items()})
for ds in sorted({k[0] for k in R}):
    ids = sorted(set.intersection(*[{i for (i, r) in R[(ds, m)] if all((i, rr) in R[(ds, m)] for rr in (0, 1, 2))} for m in M]))
    E = defaultdict(dict); diag = defaultdict(lambda: defaultdict(float))
    for i in ids:
        it = items[i]; mt = A0.metric_of(it, A0.primary(it))
        reps = {m: [[r for r in A0.recs_of(R[(ds, m)][(i, k)]) if it['task'] == 'point' or r['box'] is not None] for k in (0, 1, 2)] for m in M}
        err = lambda recs: L.error(it, recs, mt)[0]
        for m in M:
            E[f'single|{m}'][i] = err(reps[m][0])
            cl = G.build_clusters(it, reps[m][:2])
            E[f'same2_inter|{m}'][i] = err(keep(cl, lambda s: len(s) == 2)); E[f'same2_union|{m}'][i] = err(keep(cl, lambda s: True))
            cl3 = G.build_clusters(it, reps[m])
            E[f'same3_unan|{m}'][i] = err(keep(cl3, lambda s: len(s) == 3)); E[f'same3_maj|{m}'][i] = err(keep(cl3, lambda s: len(s) >= 2))
            E[f'same3_union|{m}'][i] = err(keep(cl3, lambda s: True))
            E[f'same3_ceiling|{m}'][i] = ceiling(it, [r for rr in reps[m] for r in rr] + keep(cl3, lambda s: True), mt)
            # stable misses of m: targets matched by none of m's three calls
            tm = [tmatched(it, reps[m][k], mt) for k in (0, 1, 2)]
            stable = ~(tm[0] | tm[1] | tm[2]); diag[f'stable|{m}']['n'] += stable.sum()
            for b in M:
                if b == m: continue
                tb = tmatched(it, reps[b][0], mt); diag[f'stable|{m}'][f'found_by_{b}'] += (stable & tb).sum()
                tb3 = tmatched(it, reps[b][0], mt) | tmatched(it, reps[b][1], mt) | tmatched(it, reps[b][2], mt)
                diag[f'stable|{m}'][f'found_by_{b}_3calls'] += (stable & tb3).sum()
        for a in M:
            for b in M:
                part = reps[a][1] if a == b else reps[b][0]; tag = f'{a}|{"same" if a == b else b}'
                cl = G.build_clusters(it, [reps[a][0], part])
                if a != b:
                    E[f'cross2_inter|{a}|{b}'][i] = err(keep(cl, lambda s: len(s) == 2)); E[f'cross2_union|{a}|{b}'][i] = err(keep(cl, lambda s: True))
                # diagnostics on A0's records: confirmed by partner or not
                a_cl = [x for x in cl if 0 in x[2]]; recs = [x[0] for x in a_cl]   # A0's own record as representative
                if not recs: continue
                tr = matched(it, [G.fuse([r for r in x[1] if True]) if False else x[0] for x in a_cl], mt)
                conf = np.array([len(x[2]) == 2 for x in a_cl])
                d = diag[f'pair|{tag}']
                d['A_true'] += tr.sum(); d['A_false'] += (~tr).sum(); d['true_conf'] += (tr & conf).sum(); d['false_conf'] += (~tr & conf).sum()
                # partner-only records (union adds these)
                b_only = [x for x in cl if x[2] == {1}]
                if b_only:
                    # a partner-only record is 'useful' if adding it to A0 lowers E
                    base = err(reps[a][0]); withb = err(reps[a][0] + [x[0] for x in b_only])
                    d['B_only'] += len(b_only); d['B_only_net'] += base - withb
        cross3 = [reps[m][0] for m in M]; cl = G.build_clusters(it, cross3)
        E['cross3_unan'][i] = err(keep(cl, lambda s: len(s) == 3)); E['cross3_maj'][i] = err(keep(cl, lambda s: len(s) >= 2)); E['cross3_union'][i] = err(keep(cl, lambda s: True))
        E['cross3_ceiling'][i] = ceiling(it, [r for rr in cross3 for r in rr] + keep(cl, lambda s: True), mt)
        six = [reps[m][k] for m in M for k in (0, 1)]; cl6 = G.build_clusters(it, six); owner = [0, 0, 1, 1, 2, 2]
        E['cross6_maj4'][i] = err(keep(cl6, lambda s: len(s) >= 4)); E['cross6_modelmaj'][i] = err(keep(cl6, lambda s: len({owner[k] for k in s}) >= 2))
        E['cross6_modelunan'][i] = err(keep(cl6, lambda s: len({owner[k] for k in s}) == 3))
        nine = [reps[m][k] for m in M for k in (0, 1, 2)]; cl9 = G.build_clusters(it, nine); own9 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        E['cross9_modelmaj'][i] = err(keep(cl9, lambda s: len({own9[k] for k in s}) >= 2)); E['cross9_maj5'][i] = err(keep(cl9, lambda s: len(s) >= 5))
    mean = {k: float(np.mean([v[i] for i in ids])) for k, v in E.items()}
    best = min(M, key=lambda m: mean[f'single|{m}'])
    def rel(k, base):
        a = np.array([E[base][i] for i in ids], float); b = np.array([E[k][i] for i in ids], float); bs = []
        for _ in range(2000):
            s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / a[s].mean())
        return dict(base=round(float(a.mean()), 2), val=round(float(b.mean()), 2), rel=round(float((b.mean() - a.mean()) / a.mean()) * 100, 1),
                    ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)])
    cellout = dict(images=len(ids), best_single=best, means={k: round(v, 2) for k, v in mean.items()})
    # K=2 per base model: same vs cross, relative to single A and to best single
    for a in M:
        for b in M:
            if a == b: continue
            cellout[f'K2|{a}|{b}'] = dict(cross_inter_vs_single=rel(f'cross2_inter|{a}|{b}', f'single|{a}'), same_inter_vs_single=rel(f'same2_inter|{a}', f'single|{a}'),
                                          cross_inter_vs_same_inter=rel(f'cross2_inter|{a}|{b}', f'same2_inter|{a}'),
                                          cross_union_vs_single=rel(f'cross2_union|{a}|{b}', f'single|{a}'), same_union_vs_single=rel(f'same2_union|{a}', f'single|{a}'),
                                          cross_inter_vs_best_single=rel(f'cross2_inter|{a}|{b}', f'single|{best}'))
    bm = min(['same3_unan', 'same3_maj'], key=lambda k: mean[f'{k}|{best}'])
    cellout['K3'] = dict(best_same3=f'{bm}|{best}',
                         cross3_unan_vs_best_single=rel('cross3_unan', f'single|{best}'), cross3_maj_vs_best_single=rel('cross3_maj', f'single|{best}'),
                         cross3_unan_vs_best_same3=rel('cross3_unan', f'{bm}|{best}'), cross3_maj_vs_best_same3=rel('cross3_maj', f'{bm}|{best}'),
                         **{f'cross3_{r}_vs_same3_{r}|{m}': rel(f'cross3_{r}', f'same3_{r}|{m}') for m in M for r in ('unan', 'maj')},
                         ceiling_cross3=round(1 - mean['cross3_ceiling'] / mean[f'single|{best}'], 4),
                         **{f'ceiling_same3|{m}': round(1 - mean[f'same3_ceiling|{m}'] / mean[f'single|{best}'], 4) for m in M})
    cellout['K6K9'] = {k: rel(k, f'{bm}|{best}') for k in ('cross6_maj4', 'cross6_modelmaj', 'cross6_modelunan', 'cross9_modelmaj', 'cross9_maj5')}
    cellout['diag'] = {}
    for k, d in diag.items():
        if k.startswith('pair'):
            Au = d['A_true'] - d['true_conf']; Fu = d['A_false'] - d['false_conf']
            cellout['diag'][k] = dict(p1=round(float(Au / max(1, Au + Fu)), 3), true_conf=round(float(d['true_conf'] / max(1, d['A_true'])), 3),
                                      false_conf=round(float(d['false_conf'] / max(1, d['A_false'])), 3), B_only_per_img=round(float(d['B_only'] / len(ids)), 2),
                                      union_net_gain_per_img=round(float(d['B_only_net'] / len(ids)), 2))
        else:
            cellout['diag'][k] = {kk: (round(float(v / d['n']), 3) if kk != 'n' else int(v)) for kk, v in d.items()}
    out['per_cell'][ds] = cellout
    out.setdefault('per_image', {})[ds] = {k: {str(i): v[i] for i in ids} for k, v in E.items()}
    print(ds, best, json.dumps({k: v for k, v in cellout.items() if k in ('K3', 'K6K9')})[:3000], flush=True)
    print(json.dumps(cellout['diag'])[:3000], flush=True)
OUT.write_text(json.dumps(out, indent=1))
