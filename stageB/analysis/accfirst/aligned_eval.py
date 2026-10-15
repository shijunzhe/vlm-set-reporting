"""Pilot evaluation after the candidate-ID fix (260 click items; every item in the denominator).
Flows (each item's output is one registered candidate point, or a baseline point):
  GPT-min-zoom, GPT-default-zoom   single model, two-stage zoom
  GL, GLF                          GUI-Lens-style self-crop on GPT-6.1 default ('Efficient' preset; GLF = crop first)
  M-S, D-S   Gemini-score rule: the cluster whose representative has the highest Gemini p_hit (ties: larger cluster, then
             model order); output = that representative (chosen before any comparison)
  M-C, D-C   aligned comparison: on alarmed items, a cluster wins a pair only when both orders prefer its representative;
             the cluster with most wins is output (ties and no decision: the S rule); output = the representative that was drawn
M = minimal setting for all three models; D = GPT and Sonnet default setting, Gemini minimal.
Reports: errors (all, ScreenSpot-Pro, ScreenSpot-v2), decomposition (no correct candidate / wrong cluster / wrong
representative), exact McNemar against GPT-default-zoom and GPT-min-zoom, comparator accuracy on the drawn points, calls and
historical generation cost per item. Output: accfirst/aligned_eval.json"""
import collections, json, math, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
HERE = Path(__file__).resolve().parent
GUI = os.environ.get('FSR_BASE', '/home/claude') + '/gui'
sys.path.insert(0, GUI + '/code'); sys.path.insert(0, os.environ.get('FSR_BASE', '/home/claude') + '/fsr_v51/e3')
cwd = os.getcwd(); import aligned_stages as AL; import gl_stages as GLS; os.chdir(cwd)
from score_e3 import PRICE
F, G = AL.F, AL.G
os.chdir(GUI)
C = AL.items_main(); items = sorted(C); M = AL.M
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))
cost = lambda m, r: (r['usage']['input_tokens'] * PRICE[m][0] + r['usage']['output_tokens'] * PRICE[m][1]) / 1e6 if r else 0.0
res = {}


LOADED = {st: F.load('main', st) for st in ('A', 'B', 'S', 'DA', 'DB', 'SD', 'CMPA_M', 'CMPA_D', 'GL1', 'GL2', 'GL3', 'GLF1', 'GLF2', 'GLF3')
          if os.path.exists(f'runs/main/requests_{st}.jsonl')}
REG = {cfg: AL.registry(cfg, C) for cfg in ('M', 'D')}


def s_rule(cfg, c):
    A, sc, reg = REG[cfg]; cls = reg[c]
    if not cls: return None, None
    key = lambda k: (sc.get(cls[k]['rep']['id']) if sc.get(cls[k]['rep']['id']) is not None else -1, len(cls[k]['members']), -k)
    k = max(range(len(cls)), key=key); return cls[k]['rep'], k


def c_rule(cfg, c):
    A, sc, reg = REG[cfg]; cls = reg[c]
    if len(cls) < 2: return s_rule(cfg, c)
    Q, R = LOADED[f'CMPA_{cfg}']; votes = collections.defaultdict(dict)
    for rid, r in R.items():
        if not rid.startswith(c + '|'): continue
        q = Q[rid]; v = G.first_json(r['raw_text']); lab = v.get('preferred_id') if isinstance(v, dict) else None
        win = q['meta']['labels'].get(lab)
        votes[tuple(q['meta']['pair'])][q['meta']['order']] = win
    wins = collections.Counter(); rep_of = {cl['rep']['id']: k for k, cl in enumerate(cls)}
    for pair, d in votes.items():
        if d.get('ab') is not None and d.get('ab') == d.get('ba'): wins[rep_of[d['ab']]] += 1
    if not wins: return s_rule(cfg, c)
    top = max(wins.values()); best = [k for k, v in wins.items() if v == top]
    if len(best) > 1:
        _, ks = s_rule(cfg, c)
        if ks in best: return cls[ks]['rep'], ks
        return cls[best[0]]['rep'], best[0]
    return cls[best[0]]['rep'], best[0]


def decomposition(cfg, rule):
    A, sc, reg = REG[cfg]; d = collections.Counter()
    for c in items:
        rep, k = rule(cfg, c)
        if rep is not None and hit(rep['point'], c): continue
        atoms = {a['id']: a for a in A[c]}
        if not any(hit(a['point'], c) for a in A[c]): d['no_correct_candidate'] += 1
        elif k is not None and any(hit(atoms[i]['point'], c) for i in reg[c][k]['members']): d['wrong_representative'] += 1
        else: d['wrong_cluster'] += 1
    return dict(d)


out = {}
out['GPT-min-zoom'] = {c: next((a['point'] for a in REG['M'][0][c] if a['model'] == 'gpt61'), None) for c in items}
out['GPT-default-zoom'] = {c: next((a['point'] for a in REG['D'][0][c] if a['model'] == 'gpt61'), None) for c in items}
out['Sonnet-default-zoom'] = {c: next((a['point'] for a in REG['D'][0][c] if a['model'] == 'sonnet55'), None) for c in items}
for V in ('GL', 'GLF'):
    if f'{V}1' in LOADED:
        L = {n: LOADED.get(f'{V}{n}', ({}, {})) for n in (1, 2, 3)}
        out[V] = {c: GLS.final_point(c, C, L, V)[0] for c in items}
for cfg in ('M', 'D'):
    out[f'{cfg}-S'] = {c: (s_rule(cfg, c)[0] or {}).get('point') for c in items}
    if f'CMPA_{cfg}' in LOADED: out[f'{cfg}-C'] = {c: (c_rule(cfg, c)[0] or {}).get('point') for c in items}
E = {k: {c: 1 - hit(p, c) for c, p in v.items()} for k, v in out.items()}
pro = [c for c in items if c.startswith('sspro')]; v2 = [c for c in items if c.startswith('ssv2')]
for k, e in E.items():
    res[k] = dict(errors=sum(e.values()), errors_pro=sum(e[c] for c in pro), errors_v2=sum(e[c] for c in v2), error_pct=round(100 * np.mean(list(e.values())), 2))
for k in E:
    for b in ('GPT-default-zoom', 'GPT-min-zoom'):
        if k == b: continue
        fx = sum(1 for c in items if E[b][c] and not E[k][c]); br = sum(1 for c in items if not E[b][c] and E[k][c])
        res[k][f'vs_{b}'] = dict(fixed=fx, broken=br, p_exact=round(binomtest(fx, fx + br).pvalue, 4) if fx + br else 1.0)
for cfg in ('M', 'D'):
    res[f'{cfg}-S']['decomposition'] = decomposition(cfg, s_rule)
    if f'{cfg}-C' in res: res[f'{cfg}-C']['decomposition'] = decomposition(cfg, c_rule)
# comparator accuracy on the drawn representatives (pairs with exactly one correct representative)
for cfg in ('M', 'D'):
    if f'CMPA_{cfg}' not in LOADED: continue
    Q, R = LOADED[f'CMPA_{cfg}']; per = collections.defaultdict(dict)
    for rid, r in R.items():
        q = Q[rid]; v = G.first_json(r['raw_text']); lab = v.get('preferred_id') if isinstance(v, dict) else None
        per[(q['meta']['item'], tuple(q['meta']['pair']))][q['meta']['order']] = (q['meta']['labels'].get(lab), q['meta']['points'])
    single, agree, agree_ok, n = [], 0, 0, 0
    for (c, pair), d in per.items():
        pts = next(iter(d.values()))[1]; h = {i: hit(pts[i], c) for i in pair}
        if sum(h.values()) != 1: continue
        n += 1; right = [i for i in pair if h[i]][0]
        single += [int(v[0] == right) for v in d.values()]
        if d.get('ab', (None,))[0] is not None and d.get('ab')[0] == d.get('ba', (None,))[0]: agree += 1; agree_ok += int(d['ab'][0] == right)
    res[f'comparator_{cfg}'] = dict(discordant_pairs=n, single_call_accuracy=round(float(np.mean(single)), 3) if single else None,
                                   both_orders_agree=agree, agree_accuracy=round(agree_ok / agree, 3) if agree else None)
# cost and calls per item (historical generation cost)
FLOWS = {'GPT-min-zoom': [('A', ['gpt61']), ('B', ['gpt61'])],
         'GPT-default-zoom': [('DA', ['gpt61']), ('DB', ['gpt61'])],
         'GL': [('GL1', ['gpt61']), ('GL2', ['gpt61']), ('GL3', ['gpt61'])],
         'GLF': [('GLF1', ['gpt61']), ('GLF2', ['gpt61']), ('GLF3', ['gpt61'])],
         'M-S': [('A', M), ('B', M), ('S', ['gem38'])], 'M-C': [('A', M), ('B', M), ('S', ['gem38']), ('CMPA_M', ['gem38'])],
         'D-S': [('DA', ['gpt61', 'sonnet55']), ('DB', ['gpt61', 'sonnet55']), ('A', ['gem38']), ('B', ['gem38']), ('S', ['gem38']), ('SD', ['gem38'])],
         'D-C': [('DA', ['gpt61', 'sonnet55']), ('DB', ['gpt61', 'sonnet55']), ('A', ['gem38']), ('B', ['gem38']), ('S', ['gem38']), ('SD', ['gem38']), ('CMPA_D', ['gem38'])]}


def flow_cost(flow, c):
    tot, n = 0.0, 0
    for st, models in FLOWS[flow]:
        if st not in LOADED: continue
        Q, R = LOADED[st]
        for k, r in R.items():
            parts = k.split('|')
            if parts[0] != c or parts[1] not in models: continue
            if st == 'A' and parts[2] != 'C1': continue                      # only the first click is part of these flows
            if st == 'B' and parts[2] != 'B4': continue
            if st == 'S':                                                      # Gemini scores of the flow's own minimal candidates
                cand = parts[3]
                if flow.startswith('D') and cand != 'gem38': continue
            tot += cost(parts[1], r); n += 1
    return tot, n


for fl in FLOWS:
    if fl not in res: continue
    v = [flow_cost(fl, c) for c in items]
    res[fl]['usd_per_item'] = round(float(np.mean([x[0] for x in v])), 5); res[fl]['calls_per_item'] = round(float(np.mean([x[1] for x in v])), 2)
for k, v in res.items():
    if isinstance(v, dict) and 'errors' in v:
        print(f"{k:20s} err {v['errors']:3d} ({v['error_pct']:.2f}%) pro {v['errors_pro']} v2 {v['errors_v2']}  vs GPT-default {v.get('vs_GPT-default-zoom')}  "
              f"${v.get('usd_per_item', '-')} calls {v.get('calls_per_item', '-')}  {v.get('decomposition', '')}")
for cfg in ('M', 'D'):
    if f'comparator_{cfg}' in res: print('comparator', cfg, res[f'comparator_{cfg}'])
json.dump(res, open(HERE / 'aligned_eval.json', 'w'), indent=1)
