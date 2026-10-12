"""Descriptive tables from region_rows / pred_rows / tau_sweep (v2)."""
import json, statistics as st
from collections import defaultdict, Counter
from pathlib import Path
ROOT = Path(__file__).parent
R = [json.loads(l) for l in open(ROOT / 'runs_v2/region_rows.jsonl')]
P = [json.loads(l) for l in open(ROOT / 'runs_v2/pred_rows.jsonl')]
T = [json.loads(l) for l in open(ROOT / 'runs_v2/tau_sweep.jsonl')]
NM = {'gpt56': 'GPT-5.6', 'sonnet55': 'Sonnet 5.5', 'opus55': 'Opus 5.5', 'gpt61': 'GPT-6.1', 'qwen3vl': 'Qwen3-VL'}
CL = ['gpt56', 'sonnet55', 'opus55', 'gpt61']; ALL = CL + ['qwen3vl']; ARMS = ['F', 'J', 'S1', 'C1']
out = {}
def mean(v): return round(sum(v) / len(v), 3) if v else None

# T1 per-region d by k bin
bins = [(0, 0), (1, 4), (5, 12), (13, 20), (21, 24)]
t1 = {}
for m in ALL:
    for arm in ARMS:
        for lo, hi in bins:
            v = [r for r in R if r['model'] == m and r['arm'] == arm and lo <= r['k'] <= hi]
            t1[f'{m}|{arm}|{lo}-{hi}'] = (mean([r['d'] for r in v]), mean([r['r'] - r['k'] for r in v]), len(v))
out['T1_d_by_k'] = t1

# T2 J row position at high load
t2 = {}
for m in ALL:
    for dz in (192, 384):
        for a, b in [(1, 4), (5, 8), (9, 12), (13, 16)]:
            v = [r for r in R if r['model'] == m and r['arm'] == 'J' and r['dose'] == dz and a <= (r['pos'] or 0) <= b]
            t2[f'{m}|{dz}|{a}-{b}'] = (mean([r['d'] for r in v]), mean([r['k'] for r in v]), len(v))
out['T2_J_position'] = t2

# T3 catastrophic under-report hazard: r <= k/2 among k >= 4
t3 = {}
for m in ALL:
    for arm in ARMS:
        for dz in (16, 64, 192, 384, 'all'):
            v = [r for r in R if r['model'] == m and r['arm'] == arm and r['k'] >= 4 and (dz == 'all' or r['dose'] == dz)]
            c = [r for r in v if r['r'] <= r['k'] / 2]
            t3[f'{m}|{arm}|{dz}'] = (len(c), len(v), round(len(c) / len(v), 3) if v else None,
                                     sum(r['k'] - r['t'] for r in c), Counter(r.get('status') for r in c).most_common())
out['T3_hazard'] = t3

# T4 prediction class + FN split
t4 = {}
for m in ALL:
    for arm in ARMS:
        for dz in (192, 384):
            pc = Counter(p['cls'] for p in P if p['model'] == m and p['arm'] == arm and p['dose'] == dz)
            rr = [r for r in R if r['model'] == m and r['arm'] == arm and r['dose'] == dz]
            fn_stop = sum(r['fn'] for r in rr if r['k'] >= 4 and r['r'] <= r['k'] / 2)
            fn_all = sum(r['fn'] for r in rr)
            t4[f'{m}|{arm}|{dz}'] = dict(pc, fn=fn_all, fn_stop=fn_stop, scenes=6)
out['T4_taxonomy'] = t4

# T5 localisation of matched points
t5 = {}
for m in ALL:
    for arm in ARMS:
        v = [p for p in P if p['model'] == m and p['arm'] == arm and p['cls'] == 'tp']
        if v:
            ds = sorted(p['dist'] for p in v)
            t5[f'{m}|{arm}'] = dict(n=len(v), med=round(st.median(ds), 2), p90=round(ds[int(.9 * len(ds))], 2),
                                    dx=round(mean([p['dx'] for p in v]), 2), dy=round(mean([p['dy'] for p in v]), 2),
                                    frac_gt6=round(sum(d > 6 for d in ds) / len(ds), 3))
out['T5_loc'] = t5

# T6 tau sweep, mean E per scene
t6 = {}
for m in ALL:
    for arm in ARMS:
        for dz in (16, 64, 192, 384):
            v = [t for t in T if t['model'] == m and t['arm'] == arm and t['dose'] == dz]
            t6[f'{m}|{arm}|{dz}'] = tuple(mean([t[f'E{x}'] for t in v]) for x in (6, 11, 16))
out['T6_tau'] = t6

# T7 status usage
t7 = {}
for m in ALL:
    for arm in ('J', 'S1', 'C1'):
        v = [r for r in R if r['model'] == m and r['arm'] == arm]
        t7[f'{m}|{arm}'] = dict(Counter(r['status'] for r in v),
                                found_under=sum(1 for r in v if r['status'] == 'found' and r['r'] < r['k']),
                                empty_but_k=sum(1 for r in v if r['status'] == 'empty' and r['k'] > 0),
                                k0_fp=sum(1 for r in v if r['k'] == 0 and r['r'] > 0), k0=sum(1 for r in v if r['k'] == 0))
out['T7_status'] = t7

# T8 paired region-level S1 vs J
t8 = {}
J = {(r['model'], r['sid'], r['region']): r for r in R if r['arm'] == 'J'}
for m in ALL:
    for dz in (16, 64, 192, 384):
        dd = [r['d'] - J[(m, r['sid'], r['region'])]['d'] for r in R if r['model'] == m and r['arm'] == 'S1' and r['dose'] == dz]
        pos = sorted([x for x in dd if x > 0], reverse=True)
        top = sum(pos[:max(1, len(dd) // 10)]) if pos else 0
        t8[f'{m}|{dz}'] = dict(worse=sum(x > 0 for x in dd), same=sum(x == 0 for x in dd), better=sum(x < 0 for x in dd),
                               sum_pos=sum(pos), sum_neg=sum(x for x in dd if x < 0), top10pct_share=round(top / sum(pos), 2) if pos else None)
out['T8_paired_regions'] = t8
(ROOT / 'runs_v2/diag.json').write_text(json.dumps(out, indent=1, default=str))
