"""Split the grid effect into (1) position-level detection (type-agnostic matching) and (2) type-field integrity,
and test the occlusion term: miss rate of shapes overlapped by a grid cross vs not (arm g vs arm d, same images).
Also: grid effect by list length (images with above / below median shape count). No model calls.
Usage: python -I grid_split.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
Q = C.load_requests('b1', 'main'); RS = C.load_responses('b1', 'main')
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
GT = {i: json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
CROSS = np.array([(100 * i, 100 * j) for i in range(1, 11) for j in range(1, 11)], float)
rng = np.random.default_rng(0)
per = collections.defaultdict(lambda: collections.defaultdict(list)); occ = collections.defaultdict(lambda: collections.Counter()); badtypes = collections.Counter()
for rid, q in Q.items():
    mt = q['meta']
    if mt['task'] != 'scp' or mt['arm'] not in ('d', 'g', 'f'): continue
    r = RS.get(rid)
    if not r or r['status'] != 'completed': continue
    allrecs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent'])
    G = GT[mt['item']]; recs = [x for x in allrecs if x['type'] in T5]; m = q['model_key']
    for x in allrecs:
        if x['type'] not in T5: badtypes[(m, str(x['type'])[:20])] += 1
    mm = C.match_typed(recs, G, 30); mag = C.match_typed(allrecs, G, 30, typed=False)
    hit = {j for _, j in mag}
    per[(m, mt['arm'])][mt['item']].append(dict(E=len(recs) + len(G) - 2 * len(mm), E_agn=len(allrecs) + len(G) - 2 * len(mag),
                                                 type_err=(len(mag) - len(mm)), n=len(G)))
    for j, s in enumerate(G):
        o = np.min(np.hypot(*(CROSS - [s['cx'], s['cy']]).T)) <= s['radius'] + 8
        occ[(m, mt['arm'], bool(o))]['n'] += 1; occ[(m, mt['arm'], bool(o))]['miss'] += j not in hit


def pd(m, a1, a2, key, sel=None):
    A = {i: np.mean([x[key] for x in L]) for i, L in per[(m, a1)].items()}; B = {i: np.mean([x[key] for x in L]) for i, L in per[(m, a2)].items()}
    ids = [i for i in sorted(set(A) & set(B)) if sel is None or sel(i)]
    d = np.array([B[i] - A[i] for i in ids]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return dict(a=round(float(np.mean([A[i] for i in ids])), 3), b=round(float(np.mean([B[i] for i in ids])), 3), diff=round(float(d.mean()), 3),
                ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n=len(ids))


nG = {i: len(GT[i]) for i in range(100)}; med = float(np.median(list(nG.values())))
out = dict(median_shapes=med, cells={})
for m in C.MODELS:
    c = {}
    for a1, a2 in (('d', 'g'), ('f', 'g')):
        c[f'{a2}_vs_{a1}'] = {k: pd(m, a1, a2, k) for k in ('E', 'E_agn', 'type_err')}
        c[f'{a2}_vs_{a1}']['E_long_lists'] = pd(m, a1, a2, 'E', lambda i: nG[i] > med)
        c[f'{a2}_vs_{a1}']['E_short_lists'] = pd(m, a1, a2, 'E', lambda i: nG[i] <= med)
    c['occlusion_miss_rate'] = {f'{a}|{"under_cross" if o else "clear"}': dict(rate=round(occ[(m, a, o)]['miss'] / max(1, occ[(m, a, o)]['n']), 4), n=occ[(m, a, o)]['n'])
                                for a in ('d', 'f', 'g') for o in (True, False)}
    c['off_vocab_types'] = {k[1]: v for k, v in badtypes.most_common() if k[0] == m}
    out['cells'][m] = c
json.dump(out, open('/home/claude/stageB/offline/grid_split.json', 'w'), indent=1)
for m, c in out['cells'].items():
    print('==', m)
    for k, v in c.items():
        if k == 'off_vocab_types': print(' ', k, dict(list(v.items())[:6]))
        else: print(' ', k, json.dumps(v))
