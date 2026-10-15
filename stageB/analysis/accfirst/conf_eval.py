"""Evaluation for the 600-item confirmation (protocol v3); also runs on the pilot to re-validate the frozen rules.
Usage: python3 accfirst/conf_eval.py --subset conf_v1 | main
Flows: GPT-min-zoom, GPT-default-zoom, Sonnet-default-zoom, GLF, GPTx3, M-S, M-C, D-S, D-C (rules in gui/code/decision.py).
Primary test H1: D-C vs GPT-default-zoom (paired bootstrap over screenshots, 4,000 draws, seed 20261014; exact McNemar).
H1b: D-C vs GLF and vs GPTx3, Holm over the two exact McNemar p-values. H2 (D-C vs D-S, M-C vs M-S), H3 (default vs min
zoom for GPT and Sonnet): differences with intervals only. Items whose flow is incomplete (requests never executed) are
reported separately and the primary test is run only when no item is incomplete for D-C or GPT-default-zoom.
Output: accfirst/conf_eval_<subset>.json"""
import argparse, collections, json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent
BASE = os.environ.get('FSR_BASE', '/home/claude'); GUI = BASE + '/gui'
sys.path.insert(0, GUI + '/code'); sys.path.insert(0, BASE + '/fsr_v51/e3')
cwd = os.getcwd(); import fusion_stages as F; import gl_stages as GLS; import decision as DC; os.chdir(cwd)
from score_e3 import PRICE
G, B = F.G, F.B
ap = argparse.ArgumentParser(); ap.add_argument('--subset', required=True); a = ap.parse_args(); SUB = a.subset
os.chdir(GUI)
C = F.items_of('main' if SUB == 'main' else 'conf')
if SUB == 'main':
    keep = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}; C = {k: v for k, v in C.items() if k in keep}
items = sorted(C); run = DC.Run(GUI, SUB, G, B)
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))
FLOWS = {
    'GPT-min-zoom': lambda c: DC.single_zoom(run, 'M', 'gpt61', c),
    'GPT-default-zoom': lambda c: DC.single_zoom(run, 'D', 'gpt61', c),
    'Sonnet-default-zoom': lambda c: DC.single_zoom(run, 'D', 'sonnet55', c),
    'GLF': lambda c: DC.glf(run, c, GLS.final_point, C),
    'GPTx3': lambda c: DC.gpt_x3(run, c, C[c]),
    'M-S': lambda c: DC.s_rule(run, 'M', c, C[c]), 'M-C': lambda c: DC.c_rule(run, 'M', c, C[c]),
    'D-S': lambda c: DC.s_rule(run, 'D', c, C[c]), 'D-C': lambda c: DC.c_rule(run, 'D', c, C[c])}
OUT, FLAGS = {}, {}
for fl, fn in FLOWS.items():
    OUT[fl], FLAGS[fl] = {}, {}
    for c in items:
        p, f = fn(c); OUT[fl][c] = p; FLAGS[fl][c] = sorted(f)
E = {fl: {c: 1 - hit(p, c) for c, p in v.items()} for fl, v in OUT.items()}
inc = {fl: [c for c in items if 'incomplete' in FLAGS[fl][c]] for fl in FLOWS}
groups = collections.defaultdict(list)
for i, c in enumerate(items): groups[C[c]['image']].append(i)
G_IDX = list(groups.values())


def boot(b, n, B_=4000, seed=20261014):
    rr = np.random.default_rng(seed); b, n = np.array(b, float), np.array(n, float); out = []
    for _ in range(B_):
        s = rr.integers(0, len(G_IDX), len(G_IDX)); ii = [i for k in s for i in G_IDX[k]]
        out.append(100 * (n[ii].mean() - b[ii].mean()))
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def paired(x, y):
    """y relative to x"""
    ex, ey = [E[x][c] for c in items], [E[y][c] for c in items]
    fx = sum(1 for u, v in zip(ex, ey) if u and not v); br = sum(1 for u, v in zip(ex, ey) if not u and v)
    return dict(base=x, flow=y, diff_pp=round(100 * (np.mean(ey) - np.mean(ex)), 2), ci95=boot(ex, ey), fixed=fx, broken=br,
                p_exact=round(binomtest(fx, fx + br).pvalue, 5) if fx + br else 1.0)


res = dict(subset=SUB, n=len(items), flows={})
for fl in FLOWS:
    flags = collections.Counter(f for c in items for f in FLAGS[fl][c])
    res['flows'][fl] = dict(errors=int(sum(E[fl].values())), error_pct=round(100 * np.mean(list(E[fl].values())), 2),
                            no_output=int(sum(OUT[fl][c] is None for c in items)), incomplete=len(inc[fl]), flags=dict(flags))
complete_h1 = not inc['D-C'] and not inc['GPT-default-zoom']
res['H1'] = dict(paired('GPT-default-zoom', 'D-C'), complete=complete_h1,
                 passes=bool(complete_h1 and paired('GPT-default-zoom', 'D-C')['ci95'][1] < 0))
h1b = [paired('GLF', 'D-C'), paired('GPTx3', 'D-C')]
ps = sorted(range(2), key=lambda i: h1b[i]['p_exact']); holm = [None, None]
for r, i in enumerate(ps): holm[i] = min(1.0, max(h1b[i]['p_exact'] * (2 - r), holm[ps[r - 1]] if r else 0))
for i in range(2): h1b[i]['p_holm'] = round(holm[i], 5)
res['H1b'] = h1b
res['H2'] = [paired('D-S', 'D-C'), paired('M-S', 'M-C')]
OUT['Sonnet-min-zoom'] = {c: DC.single_zoom(run, 'M', 'sonnet55', c)[0] for c in items}
E['Sonnet-min-zoom'] = {c: 1 - hit(p, c) for c, p in OUT['Sonnet-min-zoom'].items()}
res['H3'] = [paired('GPT-min-zoom', 'GPT-default-zoom'), paired('Sonnet-min-zoom', 'Sonnet-default-zoom')]
# three-term decomposition for the selection flows
for cfg, fl in (('M', 'M-S'), ('M', 'M-C'), ('D', 'D-S'), ('D', 'D-C')):
    d = collections.Counter()
    for c in items:
        if not E[fl][c]: continue
        A, sc, cls, _ = DC.structure(run, cfg, c, C[c])
        if not any(hit(x['point'], c) for x in A): d['no_correct_candidate'] += 1
        elif not any(hit(cl['rep']['point'], c) for cl in cls): d['representative_lost_correct'] += 1
        else: d['wrong_choice_among_representatives'] += 1
    res['flows'][fl]['decomposition'] = dict(d)
print(json.dumps({k: (v['errors'], v['incomplete']) for k, v in res['flows'].items()}))
print('H1', res['H1']); print('H1b', res['H1b']); print('H2', res['H2']); print('H3', res['H3'])
json.dump(res, open(HERE / f'conf_eval_{SUB}.json', 'w'), indent=1)
