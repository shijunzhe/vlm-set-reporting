"""Stricter test of redundant-field omission for Gemini (SCP only): logit of 'record has no valid type' on
same-true-type-as-previous and relative position, with arm fixed effects, standard errors by bootstrap over images
(cluster). Also the same model fitted separately within each arm. No model calls. Usage: python -I schema_decay_fe.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
Q = C.load_requests('b1', 'main'); RS = C.load_responses('b1', 'main')
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
GT = {i: json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
ARMS = list('abcdefghij'); rows = []
for rid, q in Q.items():
    mt = q['meta']
    if q['model_key'] != 'gem38' or mt['task'] != 'scp' or mt['arm'] not in ARMS: continue
    r = RS.get(rid)
    if not r or r['status'] != 'completed': continue
    recs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent'])
    G = GT[mt['item']]; mag = dict(C.match_typed(recs, G, 30, typed=False))
    true = [G[mag[k]]['type'] if k in mag else None for k in range(len(recs))]
    for k in range(1, len(recs)):
        if true[k] is None or true[k - 1] is None: continue
        rows.append((mt['item'], mt['arm'], float(true[k] == true[k - 1]), k / len(recs), float(recs[k]['type'] not in T5)))


def fit(R, arms):
    X = np.array([[float(a == b) for b in arms] + [s, p] for _, a, s, p, _ in R]); y = np.array([u for *_, u in R])
    b = np.zeros(X.shape[1])
    for _ in range(60):
        z = np.clip(X @ b, -30, 30); pr = 1 / (1 + np.exp(-z)); W = pr * (1 - pr) + 1e-9
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - pr))
    return b[-2:]


def cluster_ci(R, arms, B=300):
    by = collections.defaultdict(list)
    for x in R: by[x[0]].append(x)
    items = list(by); rng = np.random.default_rng(0); bs = []
    for _ in range(B):
        S = [x for i in rng.choice(items, len(items)) for x in by[i]]; bs.append(fit(S, arms))
    bs = np.array(bs); return [[round(float(np.percentile(bs[:, j], 2.5)), 3), round(float(np.percentile(bs[:, j], 97.5)), 3)] for j in range(2)]


out = {}
b = fit(rows, ARMS); ci = cluster_ci(rows, ARMS)
out['pooled_arm_FE'] = dict(same_type=dict(coef=round(float(b[0]), 3), ci=ci[0]), position=dict(coef=round(float(b[1]), 3), ci=ci[1]), n=len(rows))
for a in ARMS:
    R = [x for x in rows if x[1] == a]
    if len(R) < 200: continue
    b = fit(R, [a]); ci = cluster_ci(R, [a], B=200)
    out[f'arm_{a}'] = dict(same_type=dict(coef=round(float(b[0]), 3), ci=ci[0]), position=dict(coef=round(float(b[1]), 3), ci=ci[1]), n=len(R))
json.dump(out, open('/home/claude/stageB/offline/schema_decay_fe.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
