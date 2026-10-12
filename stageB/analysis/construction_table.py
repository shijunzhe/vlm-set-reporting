"""One table of constructions with every loss: E, F1, recall, precision, calls and cost. Cost per image is the mean logged cost of the stored calls each construction uses on each image (request level).
E3 (LIST-min, 4 datasets x 3 models x 60 images, K <= 3) and GUI screen parsing (GroundCUA, ScreenParse; drafts D1..D3).
Constructions for base model m (calls m0, m1, m2) and partners:
  single                     m0
  inter2_del / inter2_fus    m0 records confirmed by m1, original / fused coordinates
  union2                     all clusters of (m0, m1), fused
  maj3_del / maj3_fus        support >= 2 of (m0, m1, m2): m0's own confirmed records / all such clusters fused
  unan3_del / unan3_fus      support 3
  union3                     all clusters of three calls, fused
  x2_inter_del|p / x2_inter_fus|p / x2_union|p   m0 with partner model p's call 0
  x3_maj_fus / x3_unan_fus / x3_unan_del / x3_union   call 0 of each of the three models (del: base m's own records)
Deployable default (draft 3.5): for each loss, the construction with the best mean relative change over the OTHER five sources is
applied to the evaluated source (leave-one-source-out transfer), reported with its change over single, all other metrics and cost; the
best construction of each cell is reported only as a reference selected with that cell's labels. Output: out/construction_table.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fsr_core as F
OUT = Path(__file__).parent / 'out'; OUT.mkdir(exist_ok=True)
SAME = ['single', 'inter2_del', 'inter2_fus', 'union2', 'maj3_del', 'maj3_fus', 'unan3_del', 'unan3_fus', 'union3']


def constructions_e3(d, ds, i):
    """returns {model: {name: records}}, and costs {model: [c0, c1, c2]}"""
    R = {m: d.reps(ds, m, i) for m in F.MODELS}; out = {}
    def own(cl, need): return [x[1][0] for x in cl if 0 in x[2] and len(x[2]) >= need]
    def fus(cl, need): return [d.fuse(x[1]) for x in cl if len(x[2]) >= need]
    for m in F.MODELS:
        A = R[m]; o = {'single': A[0]}
        c2 = d.clusters(i, A[:2]); c3 = d.clusters(i, A)
        o.update(inter2_del=own(c2, 2), inter2_fus=fus(c2, 2), union2=fus(c2, 1), maj3_del=own(c3, 2), maj3_fus=fus(c3, 2),
                 unan3_del=own(c3, 3), unan3_fus=fus(c3, 3), union3=fus(c3, 1))
        for p in F.MODELS:
            if p == m: continue
            cx = d.clusters(i, [A[0], R[p][0]])
            o[f'x2_inter_del|{p}'] = own(cx, 2); o[f'x2_inter_fus|{p}'] = fus(cx, 2); o[f'x2_union|{p}'] = fus(cx, 1)
        order = [m] + [p for p in F.MODELS if p != m]
        cx3 = d.clusters(i, [R[q][0] for q in order])
        o.update(x3_maj_fus=fus(cx3, 2), x3_unan_fus=fus(cx3, 3), x3_unan_del=own(cx3, 3), x3_union=fus(cx3, 1))
        out[m] = o
    return out


def constructions_gui(g, i):
    D = {m: [g.draft(i, m, j) for j in (1, 2, 3)] for m in F.MODELS}; out = {}
    def own(cl, need): return [c['members'][0] for c in cl if 0 in c['sup'] and len(c['sup']) >= need]
    def fus(cl, need): return [list(c['fused']) for c in cl if len(c['sup']) >= need]
    for m in F.MODELS:
        A = D[m]; o = {'single': [b for b, _ in A[0]]}
        c2 = g.F.clusters(A[:2]); c3 = g.F.clusters(A)
        o.update(inter2_del=own(c2, 2), inter2_fus=fus(c2, 2), union2=fus(c2, 1), maj3_del=own(c3, 2), maj3_fus=fus(c3, 2),
                 unan3_del=own(c3, 3), unan3_fus=fus(c3, 3), union3=fus(c3, 1))
        for p in F.MODELS:
            if p == m: continue
            cx = g.F.clusters([A[0], D[p][0]])
            o[f'x2_inter_del|{p}'] = own(cx, 2); o[f'x2_inter_fus|{p}'] = fus(cx, 2); o[f'x2_union|{p}'] = fus(cx, 1)
        order = [m] + [p for p in F.MODELS if p != m]
        cx3 = g.F.clusters([D[q][0] for q in order])
        o.update(x3_maj_fus=fus(cx3, 2), x3_unan_fus=fus(cx3, 3), x3_unan_del=own(cx3, 3), x3_union=fus(cx3, 1))
        out[m] = o
    return out


def calls_of(name):
    if name == 'single': return {'self': 1}
    if name in ('inter2_del', 'inter2_fus', 'union2'): return {'self': 2}
    if name in ('maj3_del', 'maj3_fus', 'unan3_del', 'unan3_fus', 'union3'): return {'self': 3}
    if name.startswith('x2_'): return {'self': 1, name.split('|')[1]: 1}
    return {'all3': 1}


PRICE = {'gpt61': (2, 10), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}
res = {'cells': {}, 'per_image': {}}
d = F.E3Data()
# mean logged cost per call of each model on the E3 calls; used only to name the cheapest other model when partner-specific
# constructions are transferred across sources (Gemini 3.8 for GPT-6.1 and Sonnet 5.5; Sonnet 5.5 for Gemini 3.8)
PRICE_PER_CALL = {m: float(np.mean([c for (ds, mm), v in d.cost.items() if mm == m for c in v.values()])) for m in F.MODELS}


def call_cost(fam, ds, m, i, j, g=None):
    """logged cost of one stored call: E3 rep j, or GUI draft D(j+1)"""
    if fam == 'E3': return d.cost[(ds, m)].get((i, j), 0.0)
    r = g.R.get(f'{i}|{m}|D{j + 1}'); u = (r or {}).get('usage') or {}
    return ((u.get('input_tokens') or 0) * PRICE[m][0] + (u.get('output_tokens') or 0) * PRICE[m][1]) / 1e6


def cost_of(fam, ds, m, i, cl, g=None):
    """cost of the calls a construction actually uses on image i (request-level, from logged usage)"""
    if 'all3' in cl: return sum(call_cost(fam, ds, q, i, 0, g) for q in F.MODELS)
    c = sum(call_cost(fam, ds, m, i, j, g) for j in range(cl.get('self', 0)))
    return c + sum(call_cost(fam, ds, q, i, 0, g) for q in cl if q in PRICE)
sources = [('E3', ds) for ds in d.datasets] + [('GUI', s) for s in ('groundcua', 'screenparse')]
g = F.GUIData()
for fam, ds in sources:
    ids = d.images(ds) if fam == 'E3' else g.images(ds)
    per = {m: {} for m in F.MODELS}
    for i in ids:
        cons = constructions_e3(d, ds, i) if fam == 'E3' else constructions_gui(g, i)
        for m in F.MODELS:
            for name, recs in cons[m].items():
                s = d.score(i, recs) if fam == 'E3' else g.score(i, recs)
                per[m].setdefault(name, {})[i] = s
    for m in F.MODELS:
        cell = {}
        for name, D in per[m].items():
            cl = calls_of(name)
            cost = float(np.mean([cost_of(fam, ds, m, i, cl, g) for i in D]))
            cell[name] = dict(E=round(float(np.mean([s['E'] for s in D.values()])), 3), F1=round(float(np.mean([s['F1'] for s in D.values()])), 4),
                              recall=round(float(np.mean([s['recall'] for s in D.values()])), 4), precision=round(float(np.mean([s['precision'] for s in D.values()])), 4),
                              calls=sum(cl.values()) if 'all3' not in cl else 3, cost_per_image=round(cost, 5))
        res['cells'][f'{fam}|{ds}|{m}'] = cell
        res['per_image'][f'{fam}|{ds}|{m}'] = {name: {str(i): [s['E'], s['F1']] for i, s in D.items()} for name, D in per[m].items()}
    print(fam, ds, {m: (res['cells'][f'{fam}|{ds}|{m}']['single']['E'], res['cells'][f'{fam}|{ds}|{m}']['single']['F1']) for m in F.MODELS}, flush=True)

# deployable default by leave-one-source-out transfer (draft 3.5): for each loss, the generic construction with the best mean
# relative change over the OTHER five sources (all three models) is applied to the evaluated source. Partner-specific names map to roles.
def generic(name, m):
    if '|' in name:
        base, p = name.split('|'); return f'{base}|{"cheapest_other" if p == min((q for q in F.MODELS if q != m), key=lambda q: PRICE_PER_CALL[q]) else "other"}'
    return name
rng = np.random.default_rng(20261012)
res['default_transfer'] = {}
for pool in ('same_model_only', 'all'):
  for loss in ('E', 'F1'):
      sign = 1 if loss == 'E' else -1
      for fam, ds in sources:
          cand = {}
          for fam2, ds2 in sources:
              if (fam2, ds2) == (fam, ds): continue
              for q in F.MODELS:
                  cell = res['cells'][f'{fam2}|{ds2}|{q}']; base = cell['single'][loss]
                  for name, v in cell.items():
                      rel = (v[loss] - base) / base if base else 0.0
                      if pool == 'all' or name in SAME: cand.setdefault(generic(name, q), []).append(sign * rel)
          cand = {k: float(np.mean(v)) for k, v in cand.items() if len(v) == 15}
          choice = min(cand, key=cand.get)
          for m in F.MODELS:
              mine = {generic(n, m): n for n in res['cells'][f'{fam}|{ds}|{m}']}
              name = mine[choice]; cell = res['cells'][f'{fam}|{ds}|{m}']
              best = min(cell, key=lambda n: sign * cell[n][loss])
              pi = res['per_image'][f'{fam}|{ds}|{m}']; k = 0 if loss == 'E' else 1
              a = np.array([pi['single'][i][k] for i in pi['single']], float); b = np.array([pi[name][i][k] for i in pi['single']], float)
              bs = [b[s].mean() - a[s].mean() for s in (rng.integers(0, len(a), len(a)) for _ in range(4000))]
              res['default_transfer'][f'{pool}|{fam}|{ds}|{loss}|{m}'] = dict(choice=choice, applied=name, single=round(float(a.mean()), 4), value=round(float(b.mean()), 4),
                  diff=round(float(b.mean() - a.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                  cell_best_reference=best, cell_best_value=cell[best][loss], other_metrics=dict(E=cell[name]['E'], F1=cell[name]['F1'], recall=cell[name]['recall'], precision=cell[name]['precision'], cost_per_image=cell[name]['cost_per_image'], calls=cell[name]['calls']),
                  single_metrics=dict(E=cell['single']['E'], F1=cell['single']['F1'], recall=cell['single']['recall'], precision=cell['single']['precision'], cost_per_image=cell['single']['cost_per_image']))
json.dump(res, open(OUT / 'construction_table.json', 'w'), indent=1)
summ = {}
for k, v in res['default_transfer'].items():
    pool, fam, ds, loss, m = k.split('|'); good = (v['ci'][1] < 0) if loss == 'E' else (v['ci'][0] > 0); bad = (v['ci'][0] > 0) if loss == 'E' else (v['ci'][1] < 0)
    t = summ.setdefault(f'{pool}|{loss}', dict(better=0, worse=0, ns=0)); t['better' if good else 'worse' if bad else 'ns'] += 1
res['default_transfer_summary'] = summ
json.dump(res, open(OUT / 'construction_table.json', 'w'), indent=1)
print(summ)
for k, v in res['default_transfer'].items(): print(k, v['applied'], v['single'], '->', v['value'], v['ci'], 'ref best', v['cell_best_reference'], v['cell_best_value'], 'E', v['other_metrics']['E'], 'F1', v['other_metrics']['F1'], 'rec', v['other_metrics']['recall'])
