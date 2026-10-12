"""Mediation check for Gemini's grid effect: is it carried by a more row-major listing order?
Per response (SCP, arms d, e, f, g, h, i, j): typed error E, untyped share, wrong-type share, row-major Spearman.
Within-image (demeaned) regression of E on arm indicators with and without row-major order. No model calls.
Usage: python -I grid_mediation.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from scipy.stats import spearmanr
Q = C.load_requests('b1', 'main'); RS = C.load_responses('b1', 'main')
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
GT = {i: json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
rows = []
for rid, q in Q.items():
    mt = q['meta']
    if mt['task'] != 'scp' or mt['arm'] not in 'defghij' or len(mt['arm']) != 1: continue
    r = RS.get(rid)
    if not r or r['status'] != 'completed': continue
    allrecs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent'])
    if len(allrecs) < 3: continue
    G = GT[mt['item']]; recs = [x for x in allrecs if x['type'] in T5]
    mm = C.match_typed(recs, G, 30); mag = C.match_typed(allrecs, G, 30, typed=False)
    P = np.array([[x['cx'], x['cy']] for x in allrecs]); cell = (P // 100).astype(int)
    half = len(allrecs) // 2
    unt = [x['type'] not in T5 for x in allrecs]
    rows.append(dict(m=q['model_key'], arm=mt['arm'], item=mt['item'], E=len(recs) + len(G) - 2 * len(mm),
                     untyped=float(np.mean(unt)), untyped_first=float(np.mean(unt[:half])), untyped_second=float(np.mean(unt[half:])),
                     wrongtype=(len(mag) - len(mm) - sum(unt)) / len(G), pos_miss=(len(G) - len(mag)) / len(G),
                     rowmajor=spearmanr(np.arange(len(P)), cell[:, 1] * 20 + P[:, 0] / 100).correlation))
out = {}
for m in C.MODELS:
    R = [x for x in rows if x['m'] == m]
    lev = {a: {k: round(float(np.mean([x[k] for x in R if x['arm'] == a])), 4) for k in ('E', 'untyped', 'untyped_first', 'untyped_second', 'wrongtype', 'pos_miss', 'rowmajor')} for a in 'defghij'}
    # within-image regression on arms d, f, g, h (marks experiment) : E ~ g + f + h (+ rowmajor)
    S = [x for x in R if x['arm'] in 'dfgh']
    items = sorted({x['item'] for x in S}); mu = {i: (np.mean([x['E'] for x in S if x['item'] == i]), np.mean([x['rowmajor'] for x in S if x['item'] == i])) for i in items}
    y = np.array([x['E'] - mu[x['item']][0] for x in S]); rm = np.array([x['rowmajor'] - mu[x['item']][1] for x in S])
    D = np.array([[x['arm'] == a for a in 'fgh'] for x in S], float); D -= D.mean(0)
    def ols(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ b
        # cluster bootstrap over images for the grid coefficient
        idx = collections.defaultdict(list)
        for k, x in enumerate(S): idx[x['item']].append(k)
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            ks = np.concatenate([idx[i] for i in rng.choice(items, len(items))]); bb, *_ = np.linalg.lstsq(X[ks], y[ks], rcond=None); bs.append(bb)
        bs = np.array(bs)
        return [dict(coef=round(float(b[j]), 3), ci=[round(float(np.percentile(bs[:, j], 2.5)), 3), round(float(np.percentile(bs[:, j], 97.5)), 3)]) for j in range(X.shape[1])]
    m1 = ols(D); m2 = ols(np.column_stack([D, rm]))
    # within-arm association: in arm d alone, responses with more row-major order have lower E?
    Sd = [x for x in R if x['arm'] == 'd']
    rho = spearmanr([x['rowmajor'] for x in Sd], [x['E'] for x in Sd])
    out[m] = dict(levels=lev, marks_regression=dict(no_mediator=dict(zip(['edge', 'grid', 'random'], m1)), with_rowmajor=dict(zip(['edge', 'grid', 'random', 'rowmajor'], m2))),
                  arm_d_spearman_rowmajor_E=dict(rho=round(float(rho.correlation), 3), p=round(float(rho.pvalue), 4), n=len(Sd)))
json.dump(out, open('/home/claude/stageB/offline/grid_mediation.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
