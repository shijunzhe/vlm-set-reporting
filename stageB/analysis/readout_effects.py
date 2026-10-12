"""Readout effects kept separate. SCP-dense 50, min, per-type count error unless noted. No model calls.
  pure readout      same response: declared count vs code count of its list (no alias map), old protocol (B1 arm a) and new (RC16)
  normalization     same list: code count before vs after the declared alias map (RC16, and B1 arm a)
  protocol          fixed readout (mapped code count): B1 arm a list vs RC16 list (different responses, same images)
  full method       B1 arm a declared count vs RC16 readout of draft 3.4 (declared if flagged and present, else mapped code count)
  totals            same response: |list length - n| vs |declared total - n| (RC16)
Paired image bootstrap (4,000). Output: out/readout_effects.json"""
import json, sys, io, contextlib, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from degen import degenerate, T5
OUT = Path(__file__).parent / 'out'; OUT.mkdir(exist_ok=True)
SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
GT = {i: collections.Counter(s['type'] for s in json.load(open(f'{C.ROOT}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes']) for i in range(50)}
NTOT = {i: sum(v.values()) for i, v in GT.items()}
rng = np.random.default_rng(20261012)


def rows(sub, arm, m):
    Q = C.load_requests(sub, 'main'); RS = C.load_responses(sub, 'main'); out = {}
    for rid, q in Q.items():
        mt = q['meta']
        if q['model_key'] != m or mt['task'] != 'scpd' or mt['arm'] != arm or mt.get('rep', 0) != 0: continue
        r = RS.get(rid)
        if not r or r['status'] != 'completed': continue
        i = mt['item']; recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, 1024, 1024)
        raw = collections.Counter(x['type'] for x in recs if x['type'] in T5)
        mrecs = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs]; mp = collections.Counter(x['type'] for x in mrecs if x['type'] in T5)
        ok = isinstance(dec, dict); w = {t: int(dec.get(t, 0) or 0) for t in T5} if ok else None
        fl = degenerate(mrecs, sum(w.values()) if ok else None, r, True)
        e = lambda cnt: sum(abs(cnt[t] - GT[i][t]) for t in T5)
        out[i] = dict(declared=e(w) if ok else np.nan, code_raw=e(raw), code_map=e(mp), readout=(e(w) if (fl and ok) else e(mp)),
                      tot_list=abs(len(mrecs) - NTOT[i]), tot_declared=abs(sum(w.values()) - NTOT[i]) if ok else np.nan)
    return out


def pd(A, ka, B, kb):
    ids = [i for i in sorted(set(A) & set(B)) if np.isfinite(A[i][ka]) and np.isfinite(B[i][kb])]
    a = np.array([A[i][ka] for i in ids], float); b = np.array([B[i][kb] for i in ids], float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(ids), len(ids)); bs.append((b[s].mean() - a[s].mean()) / a[s].mean() if a[s].mean() else 0.0)
    return dict(n=len(ids), a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / a.mean()) * 100, 1) if a.mean() else None,
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)])


res = {}
for m in C.MODELS:
    old = rows('b1', 'a', m); new = rows('rc16', 'v516', m)
    res[m] = dict(pure_readout_old_protocol=pd(old, 'declared', old, 'code_raw'), pure_readout_new_protocol=pd(new, 'declared', new, 'code_raw'),
                  normalization_new=pd(new, 'code_raw', new, 'code_map'), normalization_old=pd(old, 'code_raw', old, 'code_map'),
                  protocol_fixed_readout=pd(old, 'code_map', new, 'code_map'), full_method=pd(old, 'declared', new, 'readout'),
                  totals_same_response_new=dict(list_mae=round(float(np.mean([v['tot_list'] for v in new.values()])), 3),
                                                declared_mae=round(float(np.nanmean([v['tot_declared'] for v in new.values()])), 3),
                                                list_exact=int(sum(v['tot_list'] == 0 for v in new.values())), n=len(new)))
    print(m, json.dumps(res[m]))
json.dump(res, open(OUT / 'readout_effects.json', 'w'), indent=1)
