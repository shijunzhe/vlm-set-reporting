"""EXP-RC16 analysis (labels used only in scoring). Per-type count error (B1 metric) and typed set error E (tol 20 px) for
arm v516 under three readouts (code count with synonym map, degeneracy-check readout, written count), paired against B1
arms on the same 50 SCP-dense images (min): a (written count), c (code, check), d, i (one-call convention), and G d_enum
(40 images). Also: flags, diamond / off-vocab records, cost. Usage: python -I analyze_rc16.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
from analyze_diag import degenerate, T5
R = C.ROOT; TOL = 20; rng = np.random.default_rng(0)
SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square', 'rhombus/diamond': 'square'}
GT = {i: json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(50)}


def score(recs, i, mapped):
    rr = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs] if mapped else recs
    typed = [x for x in rr if x['type'] in T5]; G = GT[i]
    gc = collections.Counter(s['type'] for s in G); lc = collections.Counter(x['type'] for x in typed)
    mm = C.match_typed(typed, G, TOL)
    return dict(count=sum(abs(lc[t] - gc[t]) for t in T5), E=len(typed) + len(G) - 2 * len(mm), lc=lc, gc=gc)


def written_err(dec, i):
    gc = collections.Counter(s['type'] for s in GT[i])
    return sum(abs(int(dec.get(t, 0) or 0) - gc[t]) for t in T5)


per = collections.defaultdict(dict)
Q = C.load_requests('rc16', 'main'); RS = C.load_responses('rc16', 'main')
flags = collections.defaultdict(list); voc = collections.defaultdict(collections.Counter); spend = collections.Counter()
for rid, q in Q.items():
    r = RS.get(rid); m = q['model_key']; i = q['meta']['item']
    if not r or r['status'] != 'completed': continue
    spend[m] += C.cost(m, r)
    recs, dec, au = C.decode_shapes(r['raw_text'], q['meta']['mode'], 1024, 1024, 1024, 1024)
    for x in recs: voc[m][x['type'] if x['type'] in T5 else x['type'][:14]] += 1
    s_raw = score(recs, i, False); s_map = score(recs, i, True)
    mrecs = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs]
    ok = isinstance(dec, dict)
    w = {t: int(dec.get(t, 0) or 0) for t in T5} if ok else None
    fl = degenerate(mrecs, sum(w.values()) if ok else None, r, True); flags[m].append(fl)
    werr = written_err(dec, i) if ok else s_map['count']
    per[(m, 'v516_code_nomap')][i] = dict(count=s_raw['count'], E=s_raw['E'])
    per[(m, 'v516_code')][i] = dict(count=s_map['count'], E=s_map['E'])
    per[(m, 'v516_written')][i] = dict(count=werr, E=np.nan)
    per[(m, 'v516_check')][i] = dict(count=werr if (fl and ok) else s_map['count'], E=np.nan if (fl and ok) else s_map['E'])
    per[(m, 'v516_k_minus_n')][i] = dict(count=len(recs) - len(GT[i]), E=np.nan)
# B1 comparators
Q1 = C.load_requests('b1', 'main'); R1 = C.load_responses('b1', 'main')
for rid, q in Q1.items():
    mt = q['meta']
    if mt['task'] != 'scpd' or mt['arm'] not in ('a', 'c', 'd', 'i'): continue
    r = R1.get(rid)
    if not r or r['status'] != 'completed': continue
    m = q['model_key']; i = mt['item']
    recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, 1024, 1024)
    s = score(recs, i, False); sm = score(recs, i, True)
    per[(m, 'b1_' + mt['arm'] + '_code')][i] = dict(count=s['count'], E=s['E'])
    per[(m, 'b1_' + mt['arm'] + '_code_map')][i] = dict(count=sm['count'], E=sm['E'])
    if isinstance(dec, dict):
        per[(m, 'b1_' + mt['arm'] + '_written')][i] = dict(count=written_err(dec, i), E=np.nan)
        if mt['arm'] == 'c':
            w = {t: int(dec.get(t, 0) or 0) for t in T5}; fl = degenerate(recs, sum(w.values()), r, True)
            per[(m, 'b1_c_check')][i] = dict(count=written_err(dec, i) if fl else s['count'], E=np.nan if fl else s['E'])
Qg = C.load_requests('g1', 'main'); Rg = C.load_responses('g1', 'main')
for rid, q in Qg.items():
    mt = q['meta']
    if mt['task'] != 'scpd' or mt['arm'] != 'd_enum': continue
    r = Rg.get(rid)
    if not r or r['status'] != 'completed': continue
    recs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, 1024, 1024)
    s = score(recs, mt['item'], False); sm = score(recs, mt['item'], True)
    per[(q['model_key'], 'g_denum_code')][mt['item']] = dict(count=s['count'], E=s['E'])
    per[(q['model_key'], 'g_denum_code_map')][mt['item']] = dict(count=sm['count'], E=sm['E'])


def pd(A, B, key):
    ids = [i for i in sorted(set(A) & set(B)) if np.isfinite(A[i][key]) and np.isfinite(B[i][key])]
    if len(ids) < 5: return None
    a = np.array([A[i][key] for i in ids], float); b = np.array([B[i][key] for i in ids], float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / max(1e-9, a[s].mean()))
    return dict(a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / max(1e-9, a.mean())) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(ids))


out = dict(levels={}, contrasts={}, flags={}, vocab={}, cost={k: round(v, 3) for k, v in spend.items()})
for (m, arm), D in sorted(per.items()):
    out['levels'][f'{m}|{arm}'] = dict(n=len(D), count=round(float(np.mean([v['count'] for v in D.values()])), 3),
                                        E=round(float(np.nanmean([v['E'] for v in D.values()])), 3) if any(np.isfinite(v['E']) for v in D.values()) else None,
                                        exact=round(float(np.mean([v['count'] == 0 for v in D.values()])), 3))
for m in C.MODELS:
    out['flags'][m] = dict(flagged_share=round(float(np.mean([bool(f) for f in flags[m]])), 3), kinds=dict(collections.Counter(k for f in flags[m] for k in f)))
    out['vocab'][m] = dict(voc[m].most_common(8))
    for base in ('b1_a_written', 'b1_c_check', 'b1_c_code', 'b1_i_code', 'b1_i_code_map', 'b1_d_code', 'g_denum_code', 'g_denum_code_map', 'b1_c_code_map'):
        for arm in ('v516_check', 'v516_code'):
            A = per.get((m, base)); B = per.get((m, arm))
            if A and B:
                out['contrasts'][f'{m}|{arm}_vs_{base}'] = dict(count=pd(A, B, 'count'), E=pd(A, B, 'E'))
json.dump(out, open(f'{R}/runs/rc16/rc16_analysis.json', 'w'), indent=1)
for k, v in out['levels'].items(): print(k, v)
for k, v in out['contrasts'].items(): print(k, 'count', v['count'] and (v['count']['a'], v['count']['b'], v['count']['rel'], v['count']['ci']), '| E', v['E'] and (v['E']['rel'], v['E']['ci'], v['E']['n']))
print(out['flags']); print(out['vocab']); print(out['cost'], sum(out['cost'].values()))
