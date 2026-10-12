"""PREP-1: gate and information limit on the raw-record library (E3, LIST-min, three calls, 60 images per cell). No model calls.
Library per image = the three raw lists + fused clusters (as the 45.3% ceiling). For every cluster the gate may drop it,
keep the fused record, or keep one raw member; candidate score = logistic (or boosted trees) on label-free features
(cluster features of e3_baselines511 + candidate kind + member offset from the fused record). Keep the cluster if the best
candidate has p > 1/2, output that candidate. Labels enter only to fit / score.
Estimators of the information limit (all relative to the first single call, as 45.3% / 18.1%):
  insample_logit      fitted and scored on all images (the 18.1% estimator, upward biased)
  crossfit_logit      5-fold by image (downward biased: each fold sees 4/5 of the data)
  insample_gbt / crossfit_gbt   same with gradient-boosted trees (more flexible)
  fused_only_insample  the old gate (fused candidates only) re-run here, should reproduce 18.1%
  gate_L10 / L5        10 / 5 labelled images, 40 random label sets, scored on the other images (the 12.9% estimator)
Ceilings: perfect selection from library (raw + fused), and from fused only.
usage: python -I prep1_info_limit.py"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
E3 = Path('/home/claude/fsr_v51/e3'); sys.path.insert(0, str(E3))
import os; os.chdir(E3)
import e3lib as L
import analyze_e3 as A0
import offline_agg2
import aggregate_v2 as G
from score_e3 import load_items
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
OUT = Path(__file__).parent / 'prep1_info_limit.json'


def tpoint(it, t):
    return (((t['point'][0] if 'point' in t else (t['box'][0] + t['box'][2]) / 2) + L.LABEL_OFFSET * it['labels'].index(t['label'])),
            (t['point'][1] if 'point' in t else (t['box'][1] + t['box'][3]) / 2))


def can_match(it, atoms, mt):
    Y = it['targets']
    if not atoms or not Y: return np.zeros(len(atoms), bool)
    if mt == 'point':
        P = np.array(L._pts(atoms, it['labels']), float); T = np.array([tpoint(it, t) for t in Y], float)
        return (np.hypot(P[:, None, 0] - T[None, :, 0], P[:, None, 1] - T[None, :, 1]) <= it['tau_point']).any(1)
    boxes = [a['box'] for a in atoms]
    M = L.iou_matrix(boxes, [t['box'] for t in Y]) >= mt[1] - 1e-12
    lab = np.array([[a['label'] == t['label'] for t in Y] for a in atoms])
    return (M & lab).any(1)


def offset(it, r, f):
    if it['task'] == 'point' or r['box'] is None: return math.hypot(r['cx'] - f['cx'], r['cy'] - f['cy']) / it['tau_point']
    return 1 - float(L.iou_matrix([r['box']], [f['box']])[0, 0])


def build(it, reps, mt):
    W, H = it['width'], it['height']; cl = G.build_clusters(it, reps); n_rec = len(cl); lab = defaultdict(int)
    for c in cl: lab[c[0]['label']] += 1
    spread = np.std([len(r) for r in reps]) / max(1.0, np.mean([len(r) for r in reps]))
    cands, X, cid, kind = [], [], [], []
    for ci, c in enumerate(cl):
        k = len(c[2]); f = G.fuse(c[1])
        size = ((f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]) / (W * H)) if f['box'] is not None else 0.0
        base = [1.0, float(k == 2), float(k == 3), G.dispersion(it, c[1]) if k > 1 else 0.0, math.log(max(size, 1e-6)) if f['box'] is not None else 0.0,
                math.log(1 + n_rec), lab[c[0]['label']] / n_rec, spread * (k < 3)]
        opts = [(f, 1.0, 0.0)] + ([(r, 0.0, offset(it, r, f)) for r in c[1]] if k > 1 else [])
        for a, isf, off in opts:
            cands.append(a); X.append(base + [isf, off, isf * float(k > 1)]); cid.append(ci); kind.append(isf)
    y = can_match(it, cands, mt).astype(float)
    return dict(cands=cands, X=np.array(X, float).reshape(-1, 11), y=y, cid=np.array(cid, int), fused=np.array(kind, float), ncl=n_rec)


def fit_logit(X, y, lam=0.1):
    def f(w):
        z = X @ w; return np.sum(np.logaddexp(0, z) - y * z) + lam * np.sum(w[1:] ** 2)
    return minimize(f, np.zeros(X.shape[1]), method='L-BFGS-B').x


def select(d, p, fused_only=False):
    best = {}
    for j, (c, pj) in enumerate(zip(d['cid'], p)):
        if fused_only and not d['fused'][j]: continue
        if pj > 0.5 and (c not in best or pj > best[c][0]): best[c] = (pj, j)
    return [d['cands'][j] for _, j in best.values()]


def ceiling_E(it, d, mt):
    """perfect selection: max matching over all library atoms -> E = n - C_A (only true atoms kept)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    Y = it['targets']; atoms = d['cands']
    if not atoms or not Y: return it['n'], it['n']
    if mt == 'point':
        P = np.array(L._pts(atoms, it['labels']), float); T = np.array([tpoint(it, t) for t in Y], float)
        R = np.hypot(P[:, None, 0] - T[None, :, 0], P[:, None, 1] - T[None, :, 1]) <= it['tau_point']
    else:
        R = (L.iou_matrix([a['box'] for a in atoms], [t['box'] for t in Y]) >= mt[1] - 1e-12) & np.array([[a['label'] == t['label'] for t in Y] for a in atoms])
    def mm(Rm):
        if not Rm.any(): return 0
        return int((maximum_bipartite_matching(csr_matrix(Rm.astype(np.int8)), perm_type='column') >= 0).sum())
    return it['n'] - mm(R), it['n'] - mm(R[d['fused'] == 1])


def run_gate(per, fit_ids, score_ids, model='logit', fused_only=False):
    rows = [i for i in fit_ids if len(per[i]['d']['y'])]
    msk = (lambda d: d['fused'] == 1) if fused_only else (lambda d: np.ones(len(d['y']), bool))
    X = np.vstack([per[i]['d']['X'][msk(per[i]['d'])] for i in rows]); y = np.concatenate([per[i]['d']['y'][msk(per[i]['d'])] for i in rows])
    if len(set(y)) < 2: return None
    if model == 'logit':
        w = fit_logit(X, y); pr = lambda Z: 1 / (1 + np.exp(-Z @ w))
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0).fit(X[:, 1:], y)
        pr = lambda Z: clf.predict_proba(Z[:, 1:])[:, 1]
    out = {}
    for i in score_ids:
        d = per[i]['d']
        out[i] = per[i]['single'] if not len(d['y']) else L.error(per[i]['it'], select(d, pr(d['X']), fused_only), per[i]['mt'])[0]
    return out


items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl')
cell = defaultdict(dict)
for x in rows:
    if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min': cell[(x['dataset'], x['model'])][(x['item'], x['rep'])] = x
res = {}
for (ds, m), c in sorted(cell.items()):
    per = {}
    for i in sorted({k[0] for k in c}):
        if not all((i, r) in c for r in (0, 1, 2)): continue
        it = items[i]; mt = A0.metric_of(it, A0.primary(it))
        reps = [[r for r in A0.recs_of(c[(i, k)]) if it['task'] == 'point' or r['box'] is not None] for k in (0, 1, 2)]
        d = build(it, reps, mt); ce, cef = ceiling_E(it, d, mt)
        per[i] = dict(d=d, it=it, mt=mt, single=L.error(it, A0.recs_of(c[(i, 0)]), mt)[0], ceil=ce, ceil_fused=cef)
    ids = sorted(per); S = np.mean([per[i]['single'] for i in ids])
    rel = lambda E: round(float((S - np.mean([E[i] for i in ids])) / S), 4)
    out = dict(images=len(ids), single_E=round(float(S), 2), ceiling_library=rel({i: per[i]['ceil'] for i in ids}),
               ceiling_fused_only=rel({i: per[i]['ceil_fused'] for i in ids}))
    out['fused_only_insample'] = rel(run_gate(per, ids, ids, 'logit', True))
    out['insample_logit'] = rel(run_gate(per, ids, ids, 'logit'))
    out['insample_gbt'] = rel(run_gate(per, ids, ids, 'gbt'))
    rng = random.Random(hash((ds, m)) % 1000); perm = ids[:]; rng.shuffle(perm); folds = [perm[k::5] for k in range(5)]
    for model in ('logit', 'gbt'):
        E = {}
        for f in folds: E.update(run_gate(per, [i for i in ids if i not in f], f, model))
        out[f'crossfit_{model}'] = rel(E)
    for Lb in (5, 10):
        rng.seed(hash((ds, m, Lb)) % 1000); g = []
        for s in range(40):
            lab = rng.sample(ids, Lb); rest = [i for i in ids if i not in lab]
            sr = np.mean([per[i]['single'] for i in rest]); E = run_gate(per, lab, rest, 'logit')
            g.append(0.0 if E is None else (sr - np.mean([E[i] for i in rest])) / sr)
        out[f'gate_L{Lb}'] = round(float(np.median(g)), 4); out[f'gate_L{Lb}_worse_share'] = round(float(np.mean(np.array(g) < 0)), 3)
    out['raw_choice_share_insample'] = None
    res[f'{ds}|{m}'] = out; print(ds, m, out, flush=True)
summ = {}
for k in ('ceiling_library', 'ceiling_fused_only', 'fused_only_insample', 'insample_logit', 'insample_gbt', 'crossfit_logit', 'crossfit_gbt', 'gate_L5', 'gate_L10'):
    v = [r[k] for r in res.values()]
    summ[k] = dict(median=round(float(np.median(v)), 4), better=int(sum(x > 0 for x in v)), cells=len(v))
gap = [r['ceiling_library'] - max(r['insample_logit'], r['insample_gbt']) for r in res.values()]
summ['invisible_gap_vs_best_insample'] = dict(median=round(float(np.median(gap)), 4), min=round(float(min(gap)), 4), max=round(float(max(gap)), 4))
res['_summary'] = summ; print(json.dumps(summ, indent=1))
OUT.write_text(json.dumps(res, indent=1))
