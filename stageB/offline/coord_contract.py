"""Coordinate contract of the current models on SCP (B1 arm d and experiment G arms): affine fit of reported vs true
centres for matched shapes (type-agnostic, 30 px): slope, offset, residual SD, all in pixels of the 1024 canvas. A
coordinate reference (grid, ruler) can only help if there is a systematic error to fix, relative to the 30 px matching
tolerance. No model calls. Usage: python -I coord_contract.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code'); import bcommon as C
GT = {i: json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
out = {}
for sub, arms in (('b1', ('d', 'g', 'f')), ('g1', ('d_enum', 'g_enum', 'ruler_enum'))):
    Q = C.load_requests(sub, 'main'); RS = C.load_responses(sub, 'main'); pts = collections.defaultdict(list)
    for rid, q in Q.items():
        mt = q['meta']
        if mt['task'] != 'scp' or mt['arm'] not in arms or mt.get('setting', 'min') != 'min': continue
        r = RS.get(rid)
        if not r or r['status'] != 'completed': continue
        recs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, 1024, 1024); G = GT[mt['item']]
        for a, b in C.match_typed(recs, G, 30, typed=False): pts[(q['model_key'], mt['arm'])].append((G[b]['cx'], G[b]['cy'], recs[a]['cx'], recs[a]['cy']))
    for (m, arm), P in pts.items():
        P = np.array(P, float); res = {}
        for ax, (t, p) in (('x', (0, 2)), ('y', (1, 3))):
            A = np.column_stack([P[:, t], np.ones(len(P))]); w, *_ = np.linalg.lstsq(A, P[:, p], rcond=None); e = P[:, p] - A @ w
            res[ax] = dict(slope=round(float(w[0]), 4), offset_at_centre=round(float(w[0] * 512 + w[1] - 512), 2), mean_signed=round(float(np.mean(P[:, p] - P[:, t])), 2), resid_sd=round(float(e.std()), 2))
        out[f'{sub}|{m}|{arm}'] = dict(n=len(P), **res)
json.dump(out, open('/home/claude/stageB/offline/coord_contract.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
