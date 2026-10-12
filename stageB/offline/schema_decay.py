"""Why does Gemini lose the type field, and why do grid / scan order reduce it? Test of the 'redundant-field omission'
hypothesis: a record drops its type mainly when its true type equals the previous record's true type (a run), so any
listing order with fewer same-type neighbours loses fewer types. Records are matched type-agnostically to ground truth
to get their true type. Also reports the same-type-neighbour share by arm and the position effect. No model calls.
Usage: python -I schema_decay.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
Q = C.load_requests('b1', 'main'); RS = C.load_responses('b1', 'main')
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
GT = {('scp', i): json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
GT.update({('scpd', i): json.load(open(f'/home/claude/stageB/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(50)})
TOL = {'scp': 30, 'scpd': 20}
cnt = collections.defaultdict(collections.Counter); arm_runs = collections.defaultdict(list); arm_unt = collections.defaultdict(list)
logit_rows = []
for rid, q in Q.items():
    mt = q['meta']
    if q['model_key'] != 'gem38' or mt['task'] not in ('scp', 'scpd') or mt['arm'] in ('k1', 'k2', 'l'): continue
    r = RS.get(rid)
    if not r or r['status'] != 'completed': continue
    recs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent'])
    G = GT[(mt['task'], mt['item'])]
    mag = dict(C.match_typed(recs, G, TOL[mt['task']], typed=False))
    true = [G[mag[k]]['type'] if k in mag else None for k in range(len(recs))]
    same = []
    for k in range(1, len(recs)):
        if true[k] is None or true[k - 1] is None: continue
        s = true[k] == true[k - 1]; u = recs[k]['type'] not in T5; same.append(s)
        cnt[(mt['task'], mt['arm'])][('run' if s else 'switch', 'untyped' if u else 'typed')] += 1
        logit_rows.append((mt['task'], mt['arm'], s, k / len(recs), u))
    if same: arm_runs[(mt['task'], mt['arm'])].append(np.mean(same))
    arm_unt[(mt['task'], mt['arm'])].append(np.mean([x['type'] not in T5 for x in recs]) if recs else np.nan)
out = {}
for key in sorted(cnt):
    c = cnt[key]
    pr = c[('run', 'untyped')] / max(1, c[('run', 'untyped')] + c[('run', 'typed')]); ps = c[('switch', 'untyped')] / max(1, c[('switch', 'untyped')] + c[('switch', 'typed')])
    out[f'{key[0]}|{key[1]}'] = dict(same_type_neighbour_share=round(float(np.mean(arm_runs[key])), 3), untyped_share=round(float(np.nanmean(arm_unt[key])), 3),
                                     p_untyped_given_run=round(pr, 3), p_untyped_given_switch=round(ps, 3), n_run=c[('run', 'untyped')] + c[('run', 'typed')],
                                     n_switch=c[('switch', 'untyped')] + c[('switch', 'typed')])
# pooled logistic: untyped ~ run + relative position (within arm fixed effects ignored; descriptive)
import math
X = np.array([[1.0, float(s), p] for _, _, s, p, _ in logit_rows]); y = np.array([float(u) for *_, u in logit_rows])
b = np.zeros(3)
for _ in range(50):
    z = X @ b; p = 1 / (1 + np.exp(-z)); W = p * (1 - p); H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(3); b += np.linalg.solve(H, X.T @ (y - p))
se = np.sqrt(np.diag(np.linalg.inv(H)))
out['pooled_logit'] = dict(terms=['intercept', 'same_type_as_previous', 'relative_position'], coef=[round(float(v), 3) for v in b], se=[round(float(v), 3) for v in se], n=len(y))
json.dump(out, open('/home/claude/stageB/offline/schema_decay.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
