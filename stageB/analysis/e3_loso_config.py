"""E3: leave-one-source-out choice of the single-call configuration (no model calls, no new images).
Menu: the four single-call configurations run on every E3 image (MAIN block, first repetition):
  L-min (list interface, lowest setting; the paper's current single-call default), R-min (regions interface, lowest),
  L-default, R-default.
Rule (fixed before evaluation): for a model and a loss, the configuration is chosen on the OTHER three sources only, and
applied to the held-out source, whose labels are used only for scoring.
  F1: highest mean of per-source mean F1 over the other sources.
  E : lowest mean of per-source relative change in E against L-min over the other sources (scale-free across sources).
Two variants: per model (primary; the user knows the model), and shared across the three models (one configuration per
held-out source and loss).
Scoring: primary metric of each source (boxes IoU >= 0.75 with class; FSC-147 points), as in the paper.
Intervals: paired bootstrap over images, 4,000 draws, seed from SHA-256 of a fixed string.
Output: stageB/analysis/out/e3_loso_config.json"""
import collections, hashlib, json
from pathlib import Path
import numpy as np
E3 = Path('/home/claude/fsr_v51/e3'); OUT = Path('/home/claude/stageB/analysis/out'); OUT.mkdir(exist_ok=True)
PRICE = {'gpt61': (2, 10), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}
CANDS = ['L-min', 'R-min', 'L-default', 'R-default']; REF = 'L-min'
MODELS = ['gpt61', 'sonnet55', 'gem38']


def seed(*p): return int(hashlib.sha256('|'.join(map(str, p)).encode()).hexdigest()[:8], 16)


V = collections.defaultdict(dict)          # (ds, model, cand) -> item -> (E, F1, cost)
for l in open(E3 / 'runs_main/scored.jsonl'):
    r = json.loads(l)
    if r['block'] != 'MAIN' or r['rep'] != 0: continue
    k = 'point' if r['dataset'] == 'fsc147' else 'iou75'
    c = ((r.get('in_tok') or 0) * PRICE[r['model']][0] + (r.get('out_tok') or 0) * PRICE[r['model']][1]) / 1e6
    V[(r['dataset'], r['model'], r['cand'])][r['item']] = (r['E_' + k], r['F1_' + k], c)
DS = sorted({k[0] for k in V})
items = {ds: sorted(set.intersection(*[set(V[(ds, m, c)]) for m in MODELS for c in CANDS])) for ds in DS}
mean = lambda ds, m, c, j: float(np.mean([V[(ds, m, c)][i][j] for i in items[ds]]))


def choose(loss, held, models):
    other = [d for d in DS if d != held]
    def crit(c):
        v = []
        for m in models:
            for d in other:
                if loss == 'F1': v.append(mean(d, m, c, 1))
                else: v.append(-(mean(d, m, c, 0) - mean(d, m, REF, 0)) / max(mean(d, m, REF, 0), 1e-9))
        return float(np.mean(v))
    return max(CANDS, key=lambda c: (crit(c), -CANDS.index(c)))


def boot(ds, m, a, b, j, tag):
    x = np.array([V[(ds, m, a)][i][j] for i in items[ds]]); y = np.array([V[(ds, m, b)][i][j] for i in items[ds]])
    rng = np.random.default_rng(seed('e3-loso', tag, ds, m, a, b, j)); n = len(x); out = []
    for _ in range(4000):
        s = rng.integers(0, n, n); out.append(y[s].mean() - x[s].mean())
    lo, hi = np.percentile(out, [2.5, 97.5]); return float(y.mean() - x.mean()), [round(float(lo), 4), round(float(hi), 4)]


res = dict(n_images={d: len(items[d]) for d in DS}, menu=CANDS, reference=REF, cells={}, summary={})
for variant in ('per_model', 'shared'):
    for loss, j in (('F1', 1), ('E', 0)):
        rows = []
        for ds in DS:
            for m in MODELS:
                pick = choose(loss, ds, [m] if variant == 'per_model' else MODELS)
                d, ci = boot(ds, m, REF, pick, j, variant + loss)
                if loss == 'F1': sig = 'better' if ci[0] > 0 else ('worse' if ci[1] < 0 else 'ns')
                else: sig = 'better' if ci[1] < 0 else ('worse' if ci[0] > 0 else 'ns')
                best = (max if loss == 'F1' else min)(CANDS, key=lambda c: mean(ds, m, c, j))
                rows.append(dict(source=ds, model=m, chosen=pick, ref=round(mean(ds, m, REF, j), 4), chosen_value=round(mean(ds, m, pick, j), 4),
                                 diff=round(d, 4), ci95=ci, result=sig, hindsight_best=best, hindsight_value=round(mean(ds, m, best, j), 4),
                                 cost_ref=round(mean(ds, m, REF, 2), 5), cost_chosen=round(mean(ds, m, pick, 2), 5)))
        res['cells'][f'{variant}_{loss}'] = rows
        cnt = collections.Counter(r['result'] for r in rows)
        s = dict(better=cnt['better'], worse=cnt['worse'], ns=cnt['ns'],
                 macro_ref=round(float(np.mean([r['ref'] for r in rows])), 4), macro_chosen=round(float(np.mean([r['chosen_value'] for r in rows])), 4),
                 macro_hindsight=round(float(np.mean([r['hindsight_value'] for r in rows])), 4),
                 cost_ratio=round(float(np.mean([r['cost_chosen'] for r in rows]) / np.mean([r['cost_ref'] for r in rows])), 3),
                 chosen_counts=dict(collections.Counter(r['chosen'] for r in rows)))
        if loss == 'E':
            s['median_rel_change_pct'] = round(float(np.median([100 * r['diff'] / r['ref'] for r in rows])), 1)
        res['summary'][f'{variant}_{loss}'] = s
# all-menu means for reference
res['menu_means'] = {f'{ds}|{m}|{c}': dict(E=round(mean(ds, m, c, 0), 4), F1=round(mean(ds, m, c, 1), 4), cost=round(mean(ds, m, c, 2), 5))
                     for ds in DS for m in MODELS for c in CANDS}
json.dump(res, open(OUT / 'e3_loso_config.json', 'w'), indent=1)
for k, v in res['summary'].items(): print(k, v)
for k in ('per_model_F1', 'per_model_E'):
    print(k)
    for r in res['cells'][k]: print('  ', r['source'], r['model'], r['chosen'], r['ref'], '->', r['chosen_value'], r['ci95'], r['result'], '| best', r['hindsight_best'], r['hindsight_value'])
