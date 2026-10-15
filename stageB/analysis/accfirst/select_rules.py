"""Accuracy-first selection rules borrowed from the literature, tested offline on the 260 pilot click items (no model calls).
Hypotheses = clusters of the three models' zoomed points (greedy, delta 1.5% of the longer side). Each cluster is scored with a
naive-Bayes log-likelihood ratio whose tables are estimated on a labelled anchor set (other image folds; label-light), so each
judge is weighted by its own discrimination (log-odds weighting) instead of averaging judges. Rules:
  gem      size of cluster + Gemini judge bin
  nb_all   size + every judge's bin (independent-judge NB = log-odds weighting)
  mean_p   size + mean p_hit over judges (unweighted average; the rule the literature warns against)
  size     cluster size only (majority vote)
Anchor size curve: tables from 50, 100 or all other-fold items. 5-fold cross-fitting by screenshot, 20 random fold splits.
Also: viability of a judge trigger on unanimous answers (no alarm). Output: accfirst/select_rules.json"""
import collections, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; AN = HERE.parent
src = open(AN / 'autoselect.py').read(); src = src[:src.index('rng = np.random.default_rng')]
ns = {'__file__': str(AN / 'autoselect.py')}; exec(src, ns)
items, Z, C, M, HIT, dist, score, DELTA = [ns[k] for k in ('items', 'Z', 'C', 'M', 'HIT', 'dist', 'score', 'DELTA')]
BINS = [0.0, 0.6, 0.85, 0.95, 1.01]
binof = lambda p: None if p is None else int(np.digitize([p], BINS)[0] - 1)


def clusters(c):
    H = []
    for m in M:
        p = Z[(c, m)]
        if p is None: continue
        for h in H:
            if any(dist(p, q, c) <= DELTA for _, q in h): h.append((m, p)); break
        else: H.append([(m, p)])
    out = []
    for h in H:
        f = {'size': len(h)}
        for j in M:
            ps = [score(c, j, m)['p'] for m, _ in h if score(c, j, m) and score(c, j, m)['p'] is not None]
            f[j] = binof(np.mean(ps)) if ps else None
        allp = [score(c, j, m)['p'] for m, _ in h for j in M if score(c, j, m) and score(c, j, m)['p'] is not None]
        f['meanp'] = binof(np.mean(allp)) if allp else None
        gem = {m: (score(c, 'gem38', m) or {}).get('p') or 0 for m, _ in h}
        rep = max(h, key=lambda x: gem[x[0]])[1]                     # output point: member with highest Gemini p_hit
        out.append(dict(f=f, point=rep, y=int(any(HIT(q, c) for _, q in h))))
    return out


CL = {c: clusters(c) for c in items}
RULES = {'gem': ['size', 'gem38'], 'nb_all': ['size'] + M, 'mean_p': ['size', 'meanp'], 'size': ['size']}
NV = {'size': 4, 'meanp': 4, **{j: 4 for j in M}}


def fit(anchor, feats):
    T = {}
    for k in feats:
        cnt = {1: np.ones(NV[k]) * .5, 0: np.ones(NV[k]) * .5}
        for c in anchor:
            for cl in CL[c]:
                v = cl['f'][k]
                if v is not None: cnt[cl['y']][v] += 1
        T[k] = np.log(cnt[1] / cnt[1].sum()) - np.log(cnt[0] / cnt[0].sum())
    return T


def choose(c, T):
    L = CL[c]
    if not L: return None
    s = [sum(T[k][cl['f'][k]] for k in T if cl['f'][k] is not None) for cl in L]
    return L[int(np.argmax(s))]['point']


groups = sorted({C[c]['image'] for c in items}); res = {}
for rule, feats in RULES.items():
    for na in (50, 100, 'all'):
        errs = []
        for rep in range(20):
            r = np.random.default_rng(rep); perm = r.permutation(len(groups)); fold = {groups[i]: n % 5 for n, i in enumerate(perm)}
            e = []
            for f in range(5):
                anchor = [c for c in items if fold[C[c]['image']] != f]
                if na != 'all': anchor = list(r.choice(anchor, na, replace=False))
                T = fit(anchor, feats)
                e += [1 - HIT(choose(c, T), c) for c in items if fold[C[c]['image']] == f]
            errs.append(100 * np.mean(e))
        res[f'{rule}|anchor={na}'] = dict(error=round(float(np.mean(errs)), 2), sd=round(float(np.std(errs)), 2))
        print(f'{rule:7s} anchor {str(na):4s}: error {np.mean(errs):.2f}% (sd over splits {np.std(errs):.2f})')
floor = 100 * np.mean([1 - max([cl['y'] for cl in CL[c]] or [0]) for c in items])
res['coverage_floor'] = round(floor, 2); print('coverage floor', round(floor, 2))
for m in M: res[f'zoom_only_{m}'] = round(100 * np.mean([1 - HIT(Z[(c, m)], c) for c in items]), 2)
# judge weights at full anchor (illustration) and pairwise accuracy -> log-odds
# unanimous trigger: single-cluster items with 3 members
un = [(c, CL[c][0]) for c in items if len(CL[c]) == 1 and CL[c][0]['f']['size'] == 3]
gp = lambda c, cl: np.mean([score(c, 'gem38', m)['p'] for m in M if score(c, 'gem38', m) and score(c, 'gem38', m)['p'] is not None])
w = sorted(gp(c, cl) for c, cl in un if not cl['y']); r_ = sorted(gp(c, cl) for c, cl in un if cl['y'])
res['unanimous'] = dict(n=len(un), wrong=len(w), gem_p_wrong=[round(x, 3) for x in w],
                        right_below_max_wrong=int(sum(x <= max(w) for x in r_)) if w else 0, right_n=len(r_))
print('unanimous (3 agree):', res['unanimous'])
json.dump(res, open(HERE / 'select_rules.json', 'w'), indent=1)
