"""Evaluation of the four-source tier (D4 = D + crop-first point) on the 600 confirmation items. No model calls.
Same statistics as conf_eval.py (paired screenshot bootstrap, 4,000 draws, seed 20261014; exact McNemar).
The 600 items were used to measure the crop-first source's complementarity before D4 was defined, so these are results on
the items the extension was designed on, not a confirmation on new items.
Output: accfirst/d4_eval_conf_v1.json"""
import collections, glob, itertools, json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent
GUI = '/home/claude/gui'; sys.path.insert(0, GUI + '/code'); sys.path.insert(0, '/home/claude/fsr_v51/e3')
cwd = os.getcwd(); os.chdir(GUI)
import fusion_stages as F, gl_stages as GLS, decision as DC, d4_stages as D4
from score_e3 import PRICE
G, B = F.G, F.B
C = F.items_of('conf'); items = sorted(C); run = DC.Run(GUI, 'conf_v1', G, B)
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))
OUT = {
    'GPT-default-zoom': {c: DC.single_zoom(run, 'D', 'gpt61', c)[0] for c in items},
    'GPTx3': {c: DC.gpt_x3(run, c, C[c])[0] for c in items},
    'GLF': {c: DC.glf(run, c, GLS.final_point, C)[0] for c in items},
    'D-S': {c: DC.s_rule(run, 'D', c, C[c])[0] for c in items},
    'D-C': {c: DC.c_rule(run, 'D', c, C[c])[0] for c in items}}
FL = collections.Counter(); D4S, D4C, STR = {}, {}, {}
for c in items:
    p, f, st = D4.rule4(run, c, C[c], C, 'S'); D4S[c] = p
    p, f, st = D4.rule4(run, c, C[c], C, 'C'); D4C[c] = p; STR[c] = st
    for x in f: FL[x] += 1
OUT['D4-S'] = D4S; OUT['D4-C'] = D4C
E = {k: {c: 1 - hit(p, c) for c, p in v.items()} for k, v in OUT.items()}
groups = collections.defaultdict(list)
for i, c in enumerate(items): groups[C[c]['image']].append(i)
GI = list(groups.values())


def paired(x, y):
    ex = np.array([E[x][c] for c in items], float); ey = np.array([E[y][c] for c in items], float)
    rr = np.random.default_rng(20261014); bs = []
    for _ in range(4000):
        s = rr.integers(0, len(GI), len(GI)); ii = [i for k in s for i in GI[k]]; bs.append(100 * (ey[ii].mean() - ex[ii].mean()))
    fx = int(((ex == 1) & (ey == 0)).sum()); br = int(((ex == 0) & (ey == 1)).sum())
    return dict(base=x, flow=y, diff_pp=round(100 * (ey.mean() - ex.mean()), 2), ci95=[round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
                fixed=fx, broken=br, p_exact=round(binomtest(fx, fx + br).pvalue, 5) if fx + br else 1.0)


res = dict(n=len(items), flags=dict(FL), errors={k: int(sum(v.values())) for k, v in E.items()})
res['paired'] = [paired('D-C', 'D4-C'), paired('D-S', 'D4-S'), paired('D4-S', 'D4-C'), paired('GPT-default-zoom', 'D4-C'), paired('GPTx3', 'D4-C'), paired('GLF', 'D4-C')]
# three-term split
dec = collections.Counter(); ncl = collections.Counter(); single_err = 0
for c in items:
    A, sc, cls = STR[c]; ncl[len(cls)] += 1
    if len(cls) == 1: single_err += E['D4-C'][c]
    if not E['D4-C'][c]: continue
    if not any(hit(x['point'], c) for x in A): dec['F'] += 1
    elif not any(hit(cl['rep']['point'], c) for cl in cls): dec['L'] += 1
    else: dec['W'] += 1
res['decomposition_D4C'] = dict(dec); res['single_cluster_items'] = ncl[1]; res['single_cluster_errors'] = int(single_err)
# how the GLF source changed things
moves = collections.Counter()
for c in items:
    A, sc, cls = STR[c]; g = [x for x in A if x['model'] == 'glf']
    if not g: moves['no_glf_point'] += 1; continue
    gl = [cl for cl in cls if g[0]['id'] in cl['members']][0]
    moves['glf_own_cluster' if gl['size'] == 1 else 'glf_joins_cluster'] += 1
    if gl['rep']['id'] == g[0]['id'] and gl['size'] > 1: moves['glf_became_representative_of_joined'] += 1
    if D4C[c] is not None and abs(D4C[c][0] - g[0]['point'][0]) < 1e-12 and abs(D4C[c][1] - g[0]['point'][1]) < 1e-12: moves['output_is_glf_point'] += 1
res['glf_effects'] = dict(moves)
# deployment cost per item of D4-C: the D-S calls + GLF calls + SG + the comparisons D4-C reads
cost = {}
for f in glob.glob('runs/conf_v1/responses_*_*.jsonl'):
    for l in open(f):
        r = json.loads(l)
        if r['status'] == 'completed':
            m = r['request_id'].split('|')[1]; u = r['usage']; cost[r['request_id']] = (u['input_tokens'] * PRICE[m][0] + u['output_tokens'] * PRICE[m][1]) / 1e6
dep = json.load(open(HERE / 'deploy_costs_conf_v1.json'))
extra = collections.defaultdict(float); ncalls = collections.defaultdict(int)
for c in items:
    extra[c] += cost.get(f'{c}|{DC.GEM}|SG|glf', 0); ncalls[c] += 1
    A, sc, cls = STR[c]; comp = DC.compared(sc, cls)
    if len(comp) < 2: continue
    for x, y in itertools.combinations(comp, 2):
        for o, (u, v) in (('ab', (x, y)), ('ba', (y, x))):
            for st in ('CMPA_D', 'CMPA_D4'):
                ks = [f'{c}|{DC.GEM}|{st}|{u["rep"]["id"]}|{v["rep"]["id"]}|{o}', f'{c}|{DC.GEM}|{st}|{v["rep"]["id"]}|{u["rep"]["id"]}|{o}']
                k = next((k for k in ks if k in cost), None)
                if k: extra[c] += cost[k]; ncalls[c] += 1; break
res['deploy'] = dict(usd_per_item=round(dep['D-S']['usd_per_item'] + dep['GLF']['usd_per_item'] + float(np.mean([extra[c] for c in items])), 5),
                     calls_per_item=round(dep['D-S']['calls_per_item'] + dep['GLF']['calls_per_item'] + float(np.mean([ncalls[c] for c in items])), 2))
# new calls of this step
att = 0; usd = 0.0
for st in ('SG', 'CMPA_D4'):
    for l in open(f'runs/conf_v1/responses_{st}_gem38.jsonl'):
        r = json.loads(l); att += r.get('attempts', 1) or 1; usd += cost.get(r['request_id'], 0) if r['status'] == 'completed' else 0
res['new_calls'] = dict(attempts=att, usd=round(usd, 2))
os.chdir(cwd); json.dump(res, open(HERE / 'd4_eval_conf_v1.json', 'w'), indent=1); print(json.dumps(res, indent=1))
