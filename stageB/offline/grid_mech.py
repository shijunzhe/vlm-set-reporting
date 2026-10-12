"""Mechanism of the grid effect in B1 (SCP, list-only arms). No model calls; labels used only for scoring.
For each response: misses, extras, type swaps, localisation error of matched shapes, misses split by distance to the
image border, spatial clustering of misses (region skips vs isolated), list-order structure (consecutive step length,
same-or-adjacent 100 px cell share, row-major Spearman). Compares arms with paired bootstrap over images.
Usage: python -I grid_mech.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from scipy.stats import spearmanr
Q = C.load_requests('b1', 'main'); RS = C.load_responses('b1', 'main')
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']; TOL = 30; BAND = 150
GT = {i: json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
ARMS = ['d', 'e', 'f', 'g', 'h', 'i', 'j']
rng = np.random.default_rng(0)
rows = collections.defaultdict(lambda: collections.defaultdict(list))  # (m, arm) -> item -> list of metric dicts
for rid, q in Q.items():
    mt = q['meta']
    if mt['task'] != 'scp' or mt['arm'] not in ARMS: continue
    r = RS.get(rid)
    if not r or r['status'] != 'completed': continue
    allrecs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent'])
    recs = [x for x in allrecs if x['type'] in T5]; G = GT[mt['item']]
    mm = C.match_typed(recs, G, TOL); mag = C.match_typed(allrecs, G, TOL, typed=False)
    hitG = {j for _, j in mm}; g = np.array([[s['cx'], s['cy']] for s in G])
    dB = np.minimum.reduce([g[:, 0], g[:, 1], 1024 - g[:, 0], 1024 - g[:, 1]])
    miss = np.array([j not in hitG for j in range(len(G))])
    inner = dB >= BAND
    # spatial clustering of misses: share of missed shapes whose nearest GT neighbour is also missed
    D = np.sqrt(((g[:, None] - g[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf); nn = D.argmin(1)
    clus = float(np.mean(miss[nn][miss])) if miss.sum() else np.nan
    base = float(miss.mean())
    locerr = [np.hypot(recs[a]['cx'] - G[b]['cx'], recs[a]['cy'] - G[b]['cy']) for a, b in mm]
    P = np.array([[x['cx'], x['cy']] for x in allrecs]) if allrecs else np.zeros((0, 2))
    if len(P) > 2:
        step = float(np.mean(np.hypot(*(np.diff(P, axis=0).T))))
        cell = (P // 100).astype(int); adj = np.mean(np.max(np.abs(np.diff(cell, axis=0)), 1) <= 1)
        rowmajor = spearmanr(np.arange(len(P)), cell[:, 1] * 20 + P[:, 0] / 100).correlation
    else: step = adj = rowmajor = np.nan
    rows[(q['model_key'], mt['arm'])][mt['item']].append(dict(
        missed=int(miss.sum()), extras=len(recs) - len(mm), swaps=len(mag) - len(mm), E=len(recs) + len(G) - 2 * len(mm),
        miss_inner=float(miss[inner].mean()) if inner.any() else np.nan, miss_border=float(miss[~inner].mean()) if (~inner).any() else np.nan,
        clus_excess=(clus - base) if miss.sum() else np.nan, locerr=float(np.mean(locerr)) if locerr else np.nan,
        step=step, adj=adj, rowmajor=rowmajor, n=len(G), k=len(allrecs)))
KEYS = ['E', 'missed', 'extras', 'swaps', 'miss_inner', 'miss_border', 'clus_excess', 'locerr', 'step', 'adj', 'rowmajor']


def item_means(m, arm, key):
    return {i: np.nanmean([x[key] for x in L]) for i, L in rows[(m, arm)].items()}


def pdiff(m, a1, a2, key):
    A = item_means(m, a1, key); B = item_means(m, a2, key); ids = [i for i in sorted(set(A) & set(B)) if np.isfinite(A[i]) and np.isfinite(B[i])]
    a = np.array([A[i] for i in ids]); b = np.array([B[i] for i in ids]); d = b - a; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return dict(a=round(float(a.mean()), 4), b=round(float(b.mean()), 4), diff=round(float(d.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], n=len(ids))


out = {'level': {}, 'contrasts': {}}
for m in C.MODELS:
    out['level'][m] = {arm: {k: round(float(np.nanmean(list(item_means(m, arm, k).values()))), 4) for k in KEYS} for arm in ARMS if rows[(m, arm)]}
    out['contrasts'][m] = {f'{a2}_vs_{a1}': {k: pdiff(m, a1, a2, k) for k in KEYS}
                           for a1, a2 in (('f', 'g'), ('h', 'g'), ('e', 'g'), ('d', 'g'), ('d', 'i'), ('d', 'j'), ('d', 'e'), ('d', 'f'))}
json.dump(out, open('/home/claude/stageB/offline/grid_mech.json', 'w'), indent=1)
for m in C.MODELS:
    print('==', m)
    for arm, v in out['level'][m].items(): print(' ', arm, {k: v[k] for k in KEYS})
    for c, v in out['contrasts'][m].items():
        print(' ', c, '; '.join(f"{k} {x['diff']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]" for k, x in v.items()))
