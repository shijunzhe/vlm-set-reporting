"""Cap-matched setting comparison for Gemini 3.8 on arm V516 (SCP-dense 50, identical images and prompts).
min16k (EXP-RC16) <-> default16k (EXP-RC16C) <-> default32k (EXP-RC16D). Labels only in scoring.
Per call: count readout per draft 3.4 (declared count if flagged and present, else code count), code count, declared count,
set error E of the delivered records Z_valid (class valid after the alias map; equal class, 20 px; all 50 calls, no fallback),
E_retain (sensitivity: unknown-class records kept and scored as unmatched; E_retain = E + n_unknown), degeneracy flags, strict-JSON failure, truncation, thinking and output
tokens, cost. Paired image bootstrap: default16k - min16k (setting at fixed cap), default32k - default16k (cap under default).
Output: out/rc16_setting.json"""
import json, re, sys, io, contextlib, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from degen import degenerate, T5
OUT = Path(__file__).parent / 'out'; OUT.mkdir(exist_ok=True)
SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
GT = {i: json.load(open(f'{C.ROOT}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(50)}


def strict_ok(raw):
    s = raw.strip(); s = re.sub(r'^```(?:json)?', '', s).strip(); s = re.sub(r'```$', '', s).strip()
    try:
        v = json.loads(s); return isinstance(v, dict) and isinstance(v.get('shapes'), list)
    except Exception: return False


def per_call(sub):
    Q = C.load_requests(sub, 'main'); RS = C.load_responses(sub, 'main'); out = {}
    for rid, q in Q.items():
        if q['model_key'] != 'gem38': continue
        r = RS[rid]; i = q['meta']['item']; G = GT[i]
        recs, dec, _ = C.decode_shapes(r['raw_text'], q['meta']['mode'], 1024, 1024, 1024, 1024)
        recs = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs]; typed = [x for x in recs if x['type'] in T5]
        gc = collections.Counter(s['type'] for s in G); lc = collections.Counter(x['type'] for x in typed)
        ok = isinstance(dec, dict); w = {t: int(dec.get(t, 0) or 0) for t in T5} if ok else None
        fl = degenerate(recs, sum(w.values()) if ok else None, r, True)
        code = sum(abs(lc[t] - gc[t]) for t in T5); wr = sum(abs(w[t] - gc[t]) for t in T5) if ok else None
        u = r.get('usage', {})
        out[i] = dict(readout=(wr if (fl and ok) else code), branch=('declared' if (fl and ok) else ('code_flagged_no_declared' if fl else 'code')),
                      code=code, written=wr, E=len(typed) + len(G) - 2 * len(C.match_typed(typed, G, 20)),
                      E_retain=len(recs) + len(G) - 2 * len(C.match_typed(typed, G, 20)), n_unknown=len(recs) - len(typed), flagged=bool(fl),
                      strict_fail=not strict_ok(r['raw_text']), trunc='trunc' in fl, think_tok=u.get('thinking_tokens') or 0,
                      out_tok=u.get('output_tokens') or 0, cost=C.cost('gem38', r), untyped=float(np.mean([x['type'] not in T5 for x in recs])) if recs else 0.0)
    return out


cfg = {'min16k': per_call('rc16'), 'default16k': per_call('rc16c'), 'default32k': per_call('rc16d')}
rng = np.random.default_rng(20261012)


def lev(D):
    o = {}
    for k in ('readout', 'code', 'E', 'E_retain', 'n_unknown', 'flagged', 'strict_fail', 'trunc', 'untyped', 'out_tok', 'think_tok'):
        o[k] = round(float(np.mean([v[k] for v in D.values()])), 3)
    o['written'] = round(float(np.mean([v['written'] for v in D.values() if v['written'] is not None])), 3)
    o['think_share'] = round(float(np.mean([v['think_tok'] > 0 for v in D.values()])), 3)
    o['branches'] = dict(collections.Counter(v['branch'] for v in D.values())); o['cost_total'] = round(float(sum(v['cost'] for v in D.values())), 3); o['n'] = len(D)
    return o


def pd(A, B, key):
    ids = sorted(set(A) & set(B)); a = np.array([A[i][key] for i in ids], float); b = np.array([B[i][key] for i in ids], float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(ids), len(ids)); bs.append((b[s].mean() - a[s].mean()) / a[s].mean())
    return dict(a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / a.mean()) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)])


res = dict(levels={k: lev(v) for k, v in cfg.items()}, contrasts={})
for key in ('readout', 'code', 'E'):
    res['contrasts'][f'{key}: default16k vs min16k'] = pd(cfg['min16k'], cfg['default16k'], key)
    res['contrasts'][f'{key}: default32k vs default16k'] = pd(cfg['default16k'], cfg['default32k'], key)
    res['contrasts'][f'{key}: default32k vs min16k'] = pd(cfg['min16k'], cfg['default32k'], key)
json.dump(res, open(OUT / 'rc16_setting.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
