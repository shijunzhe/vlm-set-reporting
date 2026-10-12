"""Experiment G analysis (labels used only in scoring). For every response: typed set error E, type-agnostic error E_agn,
type errors (position right, type wrong or missing), records without a valid type, 'diamond' records, localisation
error of matched shapes (px), row-major order, same-type-neighbour share, untyped rate at same-type vs type-switch
neighbours, absolute count error of the code count. Paired bootstrap over images for arm contrasts; contrasts with B1
arms on the same images (earlier run, no enum) are labelled 'b1'.
Usage: python -I analyze_g.py"""
import json, os, sys, collections
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']; TOL = {'scp': 30, 'scpd': 20}
GT = {('scp', i): json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
GT.update({('scpd', i): json.load(open(f'{C.ROOT}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(50)})
rng = np.random.default_rng(0)


def metrics(raw, mode, task, item):
    allrecs, _, au = C.decode_shapes(raw, mode, 1024, 1024, 1024, 1024)
    G = GT[(task, item)]; recs = [x for x in allrecs if x['type'] in T5]; tol = TOL[task]
    mm = C.match_typed(recs, G, tol); mag = C.match_typed(allrecs, G, tol, typed=False)
    unt = [x['type'] not in T5 for x in allrecs]
    locerr = [np.hypot(recs[a]['cx'] - G[b]['cx'], recs[a]['cy'] - G[b]['cy']) for a, b in mm]
    gc = collections.Counter(s['type'] for s in G); lc = collections.Counter(x['type'] for x in recs)
    magd = dict(mag); true = [G[magd[k]]['type'] if k in magd else None for k in range(len(allrecs))]
    run_u = run_n = sw_u = sw_n = 0; same = []
    for k in range(1, len(allrecs)):
        if true[k] is None or true[k - 1] is None: continue
        s = true[k] == true[k - 1]; same.append(s)
        if s: run_n += 1; run_u += unt[k]
        else: sw_n += 1; sw_u += unt[k]
    P = np.array([[x['cx'], x['cy']] for x in allrecs]) if len(allrecs) > 2 else None
    rowmajor = spearmanr(np.arange(len(P)), (P[:, 1] // 100) * 20 + P[:, 0] / 100).correlation if P is not None else np.nan
    half = len(allrecs) // 2
    return dict(E=len(recs) + len(G) - 2 * len(mm), E_agn=len(allrecs) + len(G) - 2 * len(mag), type_err=len(mag) - len(mm),
                untyped=float(np.mean(unt)) if unt else 0.0, untyped_2nd_half=float(np.mean(unt[half:])) if unt[half:] else 0.0,
                diamond=sum(x['type'] == 'diamond' for x in allrecs), locerr=float(np.mean(locerr)) if locerr else np.nan,
                rowmajor=rowmajor, same_nb=float(np.mean(same)) if same else np.nan, run_u=run_u, run_n=run_n, sw_u=sw_u, sw_n=sw_n,
                count_err=sum(abs(lc[t] - gc[t]) for t in T5), k_minus_n=len(allrecs) - len(G), json_ok=au['json_ok'])


per = collections.defaultdict(dict)   # (model, task, setting, arm) -> item -> metrics
Qg = C.load_requests('g1', 'main'); Rg = C.load_responses('g1', 'main')
for rid, q in Qg.items():
    r = Rg.get(rid); mt = q['meta']
    if r is None or r['status'] != 'completed': continue
    per[(q['model_key'], mt['task'], mt['setting'], mt['arm'])][mt['item']] = metrics(r['raw_text'], mt['mode'], mt['task'], mt['item'])
Q1 = C.load_requests('b1', 'main'); R1 = C.load_responses('b1', 'main')
for rid, q in Q1.items():
    mt = q['meta']
    if mt['task'] not in ('scp', 'scpd') or mt['arm'] not in ('d', 'g', 'i') or mt['rep'] != 0: continue
    r = R1.get(rid)
    if r is None or r['status'] != 'completed': continue
    per[(q['model_key'], mt['task'], 'min', 'b1_' + mt['arm'])][mt['item']] = metrics(r['raw_text'], mt['mode'], mt['task'], mt['item'])
KEYS = ['E', 'E_agn', 'type_err', 'untyped', 'untyped_2nd_half', 'diamond', 'locerr', 'rowmajor', 'same_nb', 'count_err', 'k_minus_n']


def level(cell):
    L = {k: round(float(np.nanmean([v[k] for v in cell.values()])), 4) for k in KEYS}
    ru = sum(v['run_u'] for v in cell.values()); rn = sum(v['run_n'] for v in cell.values()); su = sum(v['sw_u'] for v in cell.values()); sn = sum(v['sw_n'] for v in cell.values())
    L['p_untyped_run'] = round(ru / rn, 3) if rn else None; L['p_untyped_switch'] = round(su / sn, 3) if sn else None; L['n'] = len(cell)
    return L


def pdiff(A, B, key):
    ids = [i for i in sorted(set(A) & set(B)) if np.isfinite(A[i][key]) and np.isfinite(B[i][key])]
    if len(ids) < 5: return None
    d = np.array([B[i][key] - A[i][key] for i in ids]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    a = float(np.mean([A[i][key] for i in ids]))
    return dict(a=round(a, 3), b=round(float(np.mean([B[i][key] for i in ids])), 3), diff=round(float(d.mean()), 3),
                ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n=len(ids))


out = dict(levels={}, contrasts={})
for k, cell in sorted(per.items()): out['levels']['|'.join(k)] = level(cell)
CONTRASTS = [('d_enum', 'g_enum'), ('d_enum', 'ruler_enum'), ('d_enum', 'cell_enum'), ('d_enum', 'scan_enum'), ('d_enum', 'gscan_enum'),
             ('d_enum', 'req_enum'), ('scan_enum', 'gscan_enum'), ('g_enum', 'gscan_enum'), ('g_enum', 'cell_enum'), ('scan_enum', 'cell_enum'),
             ('b1_d', 'd_enum'), ('b1_g', 'g_enum'), ('b1_i', 'scan_enum'), ('b1_d', 'b1_g')]
for (m, task, setting) in sorted({k[:3] for k in per}):
    for a1, a2 in CONTRASTS:
        A = per.get((m, task, setting, a1)); B = per.get((m, task, setting, a2))
        if not A or not B: continue
        out['contrasts'][f'{m}|{task}|{setting}|{a2}_vs_{a1}'] = {k: pdiff(A, B, k) for k in ('E', 'E_agn', 'type_err', 'untyped', 'diamond', 'locerr', 'count_err', 'rowmajor', 'same_nb')}
# reasoning: grid effect at default vs at min (same 50 images)
for key in ('E', 'type_err', 'untyped'):
    Amin, Bmin = per.get(('gem38', 'scp', 'min', 'd_enum')), per.get(('gem38', 'scp', 'min', 'g_enum'))
    Adef, Bdef = per.get(('gem38', 'scp', 'default', 'd_enum')), per.get(('gem38', 'scp', 'default', 'g_enum'))
    if Adef and Bdef:
        ids = sorted(set(Adef) & set(Bdef) & set(Amin) & set(Bmin))
        dmin = np.array([Bmin[i][key] - Amin[i][key] for i in ids]); ddef = np.array([Bdef[i][key] - Adef[i][key] for i in ids])
        bs = [(ddef[s].mean() - dmin[s].mean()) for s in (rng.integers(0, len(ids), len(ids)) for _ in range(4000))]
        out['contrasts'][f'gem38|scp|grid_effect_default_minus_min|{key}'] = dict(grid_effect_min=round(float(dmin.mean()), 3), grid_effect_default=round(float(ddef.mean()), 3),
                                                                                diff=round(float(ddef.mean() - dmin.mean()), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n=len(ids))
spend = collections.Counter()
for rid, q in Qg.items():
    r = Rg.get(rid)
    if r: spend[(q['model_key'], q['setting'])] += C.cost(q['model_key'], r)
out['cost'] = {f'{k[0]}|{k[1]}': round(v, 2) for k, v in spend.items()}; out['cost_total'] = round(sum(spend.values()), 2)
json.dump(out, open(f'{C.ROOT}/runs/g1/g_analysis.json', 'w'), indent=1)
for k, v in out['levels'].items(): print(k, {kk: v[kk] for kk in ('n', 'E', 'E_agn', 'type_err', 'untyped', 'diamond', 'locerr', 'rowmajor', 'same_nb', 'count_err', 'p_untyped_run', 'p_untyped_switch')})
for k, v in out['contrasts'].items():
    if 'grid_effect' in k: print(k, v); continue
    print(k, '; '.join(f"{kk} {x['diff']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]" for kk, x in v.items() if x))
print('cost', out['cost'], out['cost_total'])
