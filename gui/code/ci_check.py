"""Bootstrap confidence intervals for the main claims (no model calls). Paired bootstrap over items (2,000 resamples, 95%)."""
import json, random, numpy as np, sys
rng = np.random.default_rng(0)
def boot(a, b=None, rel=False, B=2000):
    a = np.asarray(a, float); b = None if b is None else np.asarray(b, float); n = len(a); out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if b is None: out.append(a[i].mean())
        elif rel: out.append((a[i].mean() - b[i].mean()) / a[i].mean())
        else: out.append(b[i].mean() - a[i].mean())
    est = ((a.mean() - b.mean()) / a.mean()) if rel else (b.mean() - a.mean())
    return round(float(est), 4), [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
res = {}
# 1 click (ScreenSpot-Pro): accuracy differences B4-B1, B3-B1
C = json.load(open('runs/main/click_rows.json'))
for m in ('gpt61', 'sonnet55', 'gem38'):
    R = [r for r in C if r['model'] == m and r['source'] == 'sspro']
    res[f'click|{m}|B4-B1'] = boot([r['h1'] for r in R], [r['h4'] for r in R])
    res[f'click|{m}|B3-B1'] = boot([r['h1'] for r in R], [r['h3'] for r in R])
# 2 GUI parsing: relative E reduction vs A1 (positive = better), and F1 change
P = json.load(open('runs/main/parse_rows.json'))['rows']
for src in ('groundcua', 'screenparse'):
    for m in ('gpt61', 'sonnet55', 'gem38'):
        by = {}
        for r in P:
            if r['source'] == src and r['model'] == m: by.setdefault(r['arm'], {})[r['item']] = r
        ids = sorted(set.intersection(*[set(v) for v in by.values()]))
        for arm in ('A2i', 'A2u', 'A3', 'A5i'):
            res[f'parse|{src}|{m}|{arm}|relE'] = boot([by['A1'][i]['E'] for i in ids], [by[arm][i]['E'] for i in ids], rel=True)
            res[f'parse|{src}|{m}|{arm}|dF1'] = boot([by['A1'][i]['f1'] for i in ids], [by[arm][i]['f1'] for i in ids])
# 3 E3 Direct mini test
E = json.load(open('runs/e3direct/e3direct_analysis.json'))['rows']
for ds in ('sku110k', 'dota', 'fsc147'):
    for m in ('gpt61', 'sonnet55', 'gem38'):
        by = {}
        for r in E:
            if r['ds'] == ds and r['model'] == m: by.setdefault(r['arm'], {})[r['item']] = r
        ids = sorted(set.intersection(*[set(v) for v in by.values()]))
        for arm in ('inter2', 'A3'):
            res[f'e3direct|{ds}|{m}|{arm}|relE'] = boot([by['draft'][i]['E'] for i in ids], [by[arm][i]['E'] for i in ids], rel=True)
# 4 E3 zero-label intersection on the default interface (30 screening sources)
sys.path.insert(0, 'code'); import fsrgui as G
C5 = json.load(open('/home/claude/fsr_v51/model510/cache510.json'))['iface2']
for cell, I in sorted(C5.items()):
    def E_(i, o):
        pk = I[i]['L-default'][o]; tp = sum(p[2] for p in pk['P']); cnt = sum(p[3] for p in pk['P']); return cnt + I[i]['n'] - 2 * tp
    ids = sorted(I); res[f'e3zero|{cell}|inter|relE'] = boot([E_(i, 'single') for i in ids], [E_(i, 'inter') for i in ids], rel=True)
json.dump(res, open('runs/ci_check.json', 'w'), indent=1)
def excl0(ci, sign=1): return ci[0] > 0 if sign > 0 else ci[1] < 0
for k, (e, ci) in res.items():
    flag = 'better' if ci[0] > 0 else ('worse' if ci[1] < 0 else 'n.s.')
    print(k, e, ci, flag)
