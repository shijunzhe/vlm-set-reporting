"""EXP-RC16D: arm V516 for Gemini 3.8 at 'default' (thinking on, output cap 32,768), SCP-dense 50. Labels only in scoring.
Readouts as EXP-RC16 (code count with synonym map, degeneracy-check readout, written count). Paired contrasts with Gemini V516 at
min (EXP-RC16), B1 arm a written count (naive, min), and B7 Gemini default arms on the 30-image subset. Usage: python -I analyze_rc16d.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
from analyze_diag import degenerate, T5
R = C.ROOT; TOL = 20; rng = np.random.default_rng(0)
SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
GT = {i: json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(50)}
def score(raw, mode, i):
    recs, dec, _ = C.decode_shapes(raw, mode, 1024, 1024, 1024, 1024)
    recs = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs]
    typed = [x for x in recs if x['type'] in T5]; G = GT[i]
    gc = collections.Counter(s['type'] for s in G); lc = collections.Counter(x['type'] for x in typed)
    code = sum(abs(lc[t] - gc[t]) for t in T5); E = len(typed) + len(G) - 2 * len(C.match_typed(typed, G, TOL))
    w = None
    if isinstance(dec, dict): w = sum(abs(int(dec.get(t, 0) or 0) - gc[t]) for t in T5)
    return recs, dec, code, E, w
def readouts(subset, model, arm=None, setting=None):
    Q = C.load_requests(subset, 'main'); RS = C.load_responses(subset, 'main'); out = {}
    for rid, q in Q.items():
        mt = q['meta']
        if q['model_key'] != model or mt['task'] != 'scpd' or (arm and mt['arm'] != arm): continue
        r = RS.get(rid)
        if not r or r['status'] != 'completed': continue
        recs, dec, code, E, w = score(r['raw_text'], mt['mode'], mt['item'])
        ok = isinstance(dec, dict); fl = degenerate(recs, sum(int(dec.get(t, 0) or 0) for t in T5) if ok else None, r, True) if True else []
        u = r.get('usage', {})
        out[mt['item']] = dict(code=code, E=E, written=w if w is not None else code, check=(w if (fl and ok) else code), flagged=bool(fl), kinds=fl,
                               untyped=float(np.mean([x['type'] not in T5 for x in recs])) if recs else 0.0, k_minus_n=len(recs) - len(GT[mt['item']]),
                               think=int((u.get('thinking_tokens') or 0) > 0), trunc='trunc' in fl, cost=C.cost(model, r))
    return out
def pd(A, B, key='check'):
    ids = sorted(set(A) & set(B)); a = np.array([A[i][key] for i in ids], float); b = np.array([B[i][key] for i in ids], float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / max(1e-9, a[s].mean()))
    return dict(a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / max(1e-9, a.mean())) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(ids))
d = readouts('rc16d', 'gem38'); m = readouts('rc16', 'gem38')
b1a = readouts('b1', 'gem38', 'a'); b7 = {a: readouts('b7', 'gem38', a) for a in ('a', 'c', 'd', 'i')}
for v in b1a.values(): v['check'] = v['written']   # naive readout = written count
for a in b7:
    for v in b7[a].values(): v['check'] = v['written'] if a == 'a' else (v['check'] if a == 'c' else v['code'])
lev = lambda D: {k: round(float(np.mean([v[k] for v in D.values()])), 3) for k in ('check', 'code', 'written', 'E', 'flagged', 'untyped', 'k_minus_n', 'think', 'trunc')} | dict(n=len(D), exact=round(float(np.mean([v['check'] == 0 for v in D.values()])), 3), cost=round(float(sum(v['cost'] for v in D.values())), 3))
out = dict(levels=dict(v516_default=lev(d), v516_min=lev(m), b1_a_naive=lev(b1a), **{f'b7_default_{a}': lev(b7[a]) for a in b7}),
           contrasts=dict(default_vs_min=pd(m, d), default_vs_naive=pd(b1a, d), default_code_vs_min_code=pd(m, d, 'code'), default_E_vs_min_E=pd(m, d, 'E'),
                          **{f'default_vs_b7_{a}': pd(b7[a], d) for a in b7}),
           flag_kinds=dict(collections.Counter(k for v in d.values() for k in v['kinds'])))
json.dump(out, open(f'{R}/runs/rc16d/rc16d_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
