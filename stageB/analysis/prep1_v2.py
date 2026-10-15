"""PREP-1 v2 (repair A). Label-fitted output-feature selector vs constrained oracle on the SAME feasible family.
Section 4.3 of the paper (Propositions 7, 8). E3 LIST-min, K = 3, 12 cells. No model calls.
Feasible family S_A: per cluster, nothing or one candidate from A(Q) = {fused record} U {members} (members only if support >= 2);
the fused-only family uses A(Q) = {fused}. Single-call baseline: call 0.
Labels (fitting only): 'approx' = candidate compatible with some target (the v1 label, an additive approximation);
'oracle' = candidate compatible with the target assigned to its cluster by the constrained oracle (Prop. 7).
Estimators: in-sample fit; 5-fold cross-fit by image with persisted fold ids (stable digest, not Python hash);
5 / 10 labelled images, 40 draws with stable seeds. Selector classes: L2 logistic, gradient-boosted trees.
Outputs: prep1_v2.json, prep1_v2_folds.json."""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fsr_core as F
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
OUT = Path(__file__).parent / 'out'; OUT.mkdir(exist_ok=True)
d = F.E3Data(); G = d.G


def offset(it, r, f):
    if it['task'] == 'point' or r['box'] is None: return math.hypot(r['cx'] - f['cx'], r['cy'] - f['cy']) / it['tau_point']
    return 1 - float(d.L.iou_matrix([r['box']], [f['box']])[0, 0])


def build(i, reps):
    it = d.items[i]; W, H = it['width'], it['height']; cl = d.clusters(i, reps); n_rec = len(cl); lab = defaultdict(int)
    for c in cl: lab[c[0]['label']] += 1
    spread = np.std([len(r) for r in reps]) / max(1.0, np.mean([len(r) for r in reps]))
    cands, X, cid, isf, per_cl = [], [], [], [], []
    for ci, c in enumerate(cl):
        k = len(c[2]); f = d.fuse(c[1])
        size = ((f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]) / (W * H)) if f['box'] is not None else 0.0
        base = [1.0, float(k == 2), float(k == 3), G.dispersion(it, c[1]) if k > 1 else 0.0, math.log(max(size, 1e-6)) if f['box'] is not None else 0.0,
                math.log(1 + n_rec), lab[c[0]['label']] / n_rec, spread * (k < 3)]
        opts = [(f, 1.0, 0.0)] + ([(r, 0.0, offset(it, r, f)) for r in c[1]] if k > 1 else [])
        idx = []
        for a, fz, off in opts:
            idx.append(len(cands)); cands.append(a); X.append(base + [fz, off, fz * float(k > 1)]); cid.append(ci); isf.append(fz)
        per_cl.append(idx)
    Cm = d.compat(i, cands)
    n = len(it['targets'])
    # constrained oracles: full family and fused-only family
    E_full, M_full, ch_full = F.constrained_oracle([Cm[ix] for ix in per_cl], n)
    E_fz, _, _ = F.constrained_oracle([Cm[[ix[0]]] for ix in per_cl], n)
    E_unr, _ = F.unrestricted_ceiling(Cm, n)
    y_approx = Cm.any(1).astype(float) if len(cands) else np.zeros(0)
    y_or = np.zeros(len(cands))
    for q, ch in enumerate(ch_full):
        if ch is None: continue
        t = ch[1]
        for j in per_cl[q]:
            if Cm[j, t]: y_or[j] = 1.0
    return dict(cands=cands, X=np.array(X, float).reshape(-1, 11), cid=np.array(cid, int), fused=np.array(isf), per_cl=per_cl,
                y={'approx': y_approx, 'oracle': y_or}, E_or_full=E_full, E_or_fused=E_fz, E_unr=E_unr)


def fit_logit(X, y, lam=0.1):
    def f(w):
        z = X @ w; return np.sum(np.logaddexp(0, z) - y * z) + lam * np.sum(w[1:] ** 2)
    return minimize(f, np.zeros(X.shape[1]), method='L-BFGS-B').x


def selector(per, fit_ids, label, model, fused_only):
    rows = [i for i in fit_ids if len(per[i]['b']['cands'])]
    msk = (lambda b: b['fused'] == 1) if fused_only else (lambda b: np.ones(len(b['cands']), bool))
    X = np.vstack([per[i]['b']['X'][msk(per[i]['b'])] for i in rows]); y = np.concatenate([per[i]['b']['y'][label][msk(per[i]['b'])] for i in rows])
    if len(set(y)) < 2: return None
    if model == 'logit':
        w = fit_logit(X, y); return lambda Z: 1 / (1 + np.exp(-Z @ w))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0).fit(X[:, 1:], y)
    return lambda Z: clf.predict_proba(Z[:, 1:])[:, 1]


def apply(per, i, pr, fused_only):
    b = per[i]['b']
    if pr is None or not len(b['cands']): return per[i]['single']
    p = pr(b['X']); out = []
    for ix in b['per_cl']:
        ix = [ix[0]] if fused_only else ix
        j = max(ix, key=lambda t: p[t])
        if p[j] > 0.5: out.append(b['cands'][j])
    return d.score(i, out)['E']


res = {}; folds_out = {}
for ds in d.datasets:
    ids = d.images(ds)
    fold = {i: F.stable_seed('prep1', ds, i) % 5 for i in ids}; folds_out[ds] = fold
    for m in F.MODELS:
        per = {}
        for i in ids:
            reps = d.reps(ds, m, i); per[i] = dict(b=build(i, reps), single=d.score(i, reps[0])['E'])
        S = np.mean([per[i]['single'] for i in ids])
        def gain(Ed): return float((S - np.mean([Ed[i] for i in ids])) / S)
        out = dict(images=len(ids), single_E=round(float(S), 3),
                   oracle_constrained=gain({i: per[i]['b']['E_or_full'] for i in ids}),
                   oracle_constrained_fused_only=gain({i: per[i]['b']['E_or_fused'] for i in ids}),
                   oracle_unrestricted_atoms=gain({i: per[i]['b']['E_unr'] for i in ids}),
                   cases_unrestricted_ne_constrained=int(sum(per[i]['b']['E_unr'] != per[i]['b']['E_or_full'] for i in ids)))
        for label in ('approx', 'oracle'):
            for model in ('logit', 'gbt'):
                pr = selector(per, ids, label, model, False)
                out[f'insample_{label}_{model}'] = gain({i: apply(per, i, pr, False) for i in ids})
                Ecf = {}
                for k in range(5):
                    te = [i for i in ids if fold[i] == k]; tr = [i for i in ids if fold[i] != k]
                    pr = selector(per, tr, label, model, False)
                    for i in te: Ecf[i] = apply(per, i, pr, False)
                out[f'crossfit_{label}_{model}'] = gain(Ecf)
                if label == 'oracle' and model == 'logit':
                    # evaluation-sampling CI of the cross-fit gap (selector fixed): image bootstrap
                    rng = np.random.default_rng(F.stable_seed('boot', ds, m)); a = np.array([per[i]['single'] for i in ids], float)
                    e = np.array([Ecf[i] for i in ids], float); o = np.array([per[i]['b']['E_or_full'] for i in ids], float); bs = []
                    for _ in range(2000):
                        s = rng.integers(0, len(ids), len(ids)); bs.append(((a[s].mean() - o[s].mean()) - (a[s].mean() - e[s].mean())) / a[s].mean())
                    out['gap_crossfit_oracle_logit_ci'] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        prf = selector(per, ids, 'approx', 'logit', True)
        out['insample_fusedonly_approx_logit'] = gain({i: apply(per, i, prf, True) for i in ids})
        for Lb in (5, 10):
            g = []
            for s in range(40):
                rng = random.Random(F.stable_seed('labels', ds, m, Lb, s)); lab = rng.sample(ids, Lb); rest = [i for i in ids if i not in lab]
                sr = np.mean([per[i]['single'] for i in rest]); pr = selector(per, lab, 'oracle', 'logit', False)
                g.append((sr - np.mean([apply(per, i, pr, False) for i in rest])) / sr)
            out[f'labels{Lb}_oracle_logit_median'] = float(np.median(g)); out[f'labels{Lb}_worse_share'] = float(np.mean(np.array(g) < 0))
        out['gap_crossfit_oracle_logit'] = out['oracle_constrained'] - out['crossfit_oracle_logit']
        res[f'{ds}|{m}'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
        print(ds, m, {k: res[f'{ds}|{m}'][k] for k in ('oracle_constrained', 'oracle_unrestricted_atoms', 'cases_unrestricted_ne_constrained', 'crossfit_oracle_logit', 'crossfit_approx_logit', 'gap_crossfit_oracle_logit')}, flush=True)
keys = [k for k in next(iter(res.values())) if isinstance(next(iter(res.values()))[k], float)]
res['_summary'] = {k: dict(median=round(float(np.median([r[k] for r in res.values() if isinstance(r, dict) and k in r])), 4)) for k in keys}
res['_summary']['cases_unrestricted_ne_constrained_total'] = int(sum(r['cases_unrestricted_ne_constrained'] for kk, r in res.items() if not kk.startswith('_')))
json.dump(res, open(OUT / 'prep1_v2.json', 'w'), indent=1)
json.dump({ds: {str(k): v for k, v in f.items()} for ds, f in folds_out.items()}, open(OUT / 'prep1_v2_folds.json', 'w'), indent=1)
print(json.dumps(res['_summary'], indent=0))
