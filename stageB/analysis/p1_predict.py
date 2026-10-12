"""Predictive test of the deletion threshold with a few labels (Section 5, Step 7; the intersection threshold). No model calls.
For each of the 54 base/partner pairs of Table 9 (E3 and screen sources, 60 images each), per image: deleted scored records
dc_i = c(A) - c(S0) and lost matches dr_i = r(A) - r(S0), where S0 keeps the base call's records confirmed by the partner
(deletion only). The selection gain on an image is g_i = dc_i - 2 dr_i (equal error costs).
Test: draw L labelled images (L = 3, 5, 10; 200 seeded draws per pair), decide from them only, and score the decision on the
other 60 - L images of the same pair.
  threshold rule   delete iff the pooled marginal matching rate p1 = sum dr / sum dc on the labelled images is below 1/2
                   (equivalently mean g > 0); keep if the labelled images deleted nothing
  Step 7 rule      start from the shipped regime (delete on over-reporting families, keep on FSC-147) and switch only if the
                   other decision's mean advantage on the labelled images minus one standard error is positive (the shipped
                   label rule); 'from keep' applies the same margin starting from never deleting
Outcome on the held-out images: whether the decision has the sign of the held-out mean gain, and the regret per image
relative to the better of delete / keep there. Compared with always deleting, never deleting, and the shipped regime
(delete on over-reporting families, keep on FSC-147).
Output: out/p1_predict.json"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fsr_core as F
OUT = Path(__file__).parent / 'out'

per = {}
d = F.E3Data()
for ds in d.datasets:
    ids = d.images(ds)
    for a in F.MODELS:
        for b in F.MODELS:
            rows = []
            for i in ids:
                A = d.reps(ds, a, i); base = A[0]; part = A[1] if a == b else d.reps(ds, b, i)[0]
                cl = d.clusters(i, [base, part]); S0 = [x[1][0] for x in cl if 0 in x[2] and len(x[2]) == 2]
                sA, sS = d.score(i, base), d.score(i, S0)
                rows.append((sA['c'] - sS['c'], sA['r'] - sS['r']))
            per[f'E3|{ds}|{a}|{"same" if a == b else b}'] = rows
g = F.GUIData()
for src in ('groundcua', 'screenparse'):
    ids = g.images(src)
    for a in F.MODELS:
        for b in F.MODELS:
            rows = []
            for i in ids:
                base = g.draft(i, a, 1); part = g.draft(i, a, 2) if a == b else g.draft(i, b, 1)
                S0 = [c['members'][0] for c in g.F.clusters([base, part]) if 0 in c['sup'] and len(c['sup']) == 2]
                sA, sS = g.score(i, [x[0] for x in base]), g.score(i, S0)
                rows.append((sA['c'] - sS['c'], sA['r'] - sS['r']))
            per[f'GUI|{src}|{a}|{"same" if a == b else b}'] = rows

res = {'pairs': {}, 'summary': {}}
for L in (3, 5, 10):
    acc = {k: [] for k in ('threshold', 'step7', 'step7_from_keep', 'always', 'never', 'regime')}; reg = {k: [] for k in acc}
    near = {'threshold': [], 'step7': [], 'regime': []}
    for key, rows in per.items():
        dc = np.array([r[0] for r in rows], float); dr = np.array([r[1] for r in rows], float); gain = dc - 2 * dr; n = len(gain)
        p1_all = dr.sum() / dc.sum() if dc.sum() else float('nan')
        rng = np.random.default_rng(F.stable_seed('p1_predict', key, L))
        regime_delete = key.split('|')[1] != 'fsc147'
        pp = {'threshold': [], 'step7': [], 'regime': []}
        for _ in range(200):
            lab = rng.choice(n, L, replace=False); rest = np.setdiff1d(np.arange(n), lab)
            gl, gr = gain[lab], gain[rest]; held = gr.mean(); best = max(held, 0.0)
            se = gl.std(ddof=1) / math.sqrt(L)
            dec = dict(threshold=bool(dc[lab].sum() > 0 and dr[lab].sum() / dc[lab].sum() < 0.5),
                       step7=(not (-gl.mean() - se > 0)) if regime_delete else bool(gl.mean() - se > 0),
                       step7_from_keep=bool(gl.mean() - se > 0),
                       always=True, never=False, regime=regime_delete)
            for k in pp: pp[k].append(int(dec[k] == (held > 0)) if held != 0 else 1)
            for k, v in dec.items():
                got = held if v else 0.0
                acc[k].append(int(v == (held > 0)) if held != 0 else 1); reg[k].append(best - got)
            if abs(p1_all - 0.5) < 0.1:
                for k in near: near[k].append(int(dec[k] == (held > 0)) if held != 0 else 1)
        if L == 5: res['pairs'][key] = dict(p1=round(float(p1_all), 3), mean_gain=round(float(gain.mean()), 3), **{f'sign_correct_{k}': round(float(np.mean(v)), 3) for k, v in pp.items()})
    res['summary'][f'L={L}'] = {k: dict(sign_correct=round(float(np.mean(acc[k])), 3), regret_per_image=round(float(np.mean(reg[k])), 3)) for k in acc}
    res['summary'][f'L={L}']['pairs_with_p1_within_0.1_of_half'] = {k: round(float(np.mean(v)), 3) for k, v in near.items()}
    res['summary'][f'L={L}']['n_pairs'] = len(per)
json.dump(res, open(OUT / 'p1_predict.json', 'w'), indent=1)
print(json.dumps(res['summary'], indent=1))
