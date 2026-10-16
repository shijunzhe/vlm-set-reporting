"""A plain vote over the three models of the click tier, from stored calls (no model calls).
The candidates are those of Algorithm 2 (GPT-6.1 and Sonnet 5.5 default, Gemini 3.8 lowest, two-stage zoom). They are
clustered with the same rule (connected components within delta); the largest cluster wins, ties go to the cluster with
the earliest source (GPT-6.1, Sonnet 5.5, Gemini 3.8), and its earliest member is output. This is the rule of the
three-zoom GPT-6.1 vote applied to the three models; no scorer and no comparison are used.
Paired tests against the other flows use the same bootstrap (screenshots, 4,000 draws, seed 20261014) and exact McNemar
as conf_eval.py.
Usage: python3 accfirst/conf_vote.py --subset conf_v1 | main
Output: accfirst/conf_vote_<subset>.json"""
import argparse, collections, json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent
BASE = os.environ.get('FSR_BASE', '/home/claude'); GUI = BASE + '/gui'
sys.path.insert(0, GUI + '/code')
cwd = os.getcwd(); import fusion_stages as F; import decision as DC; os.chdir(cwd)
G, B = F.G, F.B
ap = argparse.ArgumentParser(); ap.add_argument('--subset', required=True); a = ap.parse_args(); SUB = a.subset
os.chdir(GUI)
C = F.items_of('main' if SUB == 'main' else 'conf')
if SUB == 'main':
    keep = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}; C = {k: v for k, v in C.items() if k in keep}
items = sorted(C); run = DC.Run(GUI, SUB, G, B)
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))


def vote(c):
    flags = set(); A = DC.atoms(run, 'D', c, flags)
    if not A: return None
    cls = DC.clusters(C[c], A)
    best = max(cls, key=lambda g: (len(g), -min(DC.ORDER[x['model']] for x in g)))
    return min(best, key=lambda x: DC.ORDER[x['model']])['point']


FLOWS = {'vote-3-models': vote,
         'GPT-default-zoom': lambda c: DC.single_zoom(run, 'D', 'gpt61', c)[0],
         'GPTx3': lambda c: DC.gpt_x3(run, c, C[c])[0],
         'D-S': lambda c: DC.s_rule(run, 'D', c, C[c])[0],
         'D-C': lambda c: DC.c_rule(run, 'D', c, C[c])[0]}
P = {fl: {c: fn(c) for c in items} for fl, fn in FLOWS.items()}
E = {fl: {c: 1 - hit(p, c) for c, p in v.items()} for fl, v in P.items()}
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
    ex, ey = [E[x][c] for c in items], [E[y][c] for c in items]
    fx = sum(1 for u, v in zip(ex, ey) if u and not v); br = sum(1 for u, v in zip(ex, ey) if not u and v)
    return dict(base=x, flow=y, diff_pp=round(100 * (np.mean(ey) - np.mean(ex)), 2), ci95=boot(ex, ey), fixed=fx, broken=br,
                p_exact=round(binomtest(fx, fx + br).pvalue, 5) if fx + br else 1.0)


res = dict(subset=SUB, n=len(items),
           vote=dict(errors=int(sum(E['vote-3-models'].values())), error_pct=round(100 * np.mean(list(E['vote-3-models'].values())), 2),
                     no_output=int(sum(p is None for p in P['vote-3-models'].values()))),
           paired=[paired('vote-3-models', 'D-C'), paired('vote-3-models', 'D-S'), paired('GPT-default-zoom', 'vote-3-models'),
                   ] + ([paired('GPTx3', 'vote-3-models')] if SUB != 'main' else []))  # no repeated GPT-6.1 zooms on the 260 items
print(json.dumps(res, indent=1))
json.dump(res, open(HERE / f'conf_vote_{SUB}.json', 'w'), indent=1)
