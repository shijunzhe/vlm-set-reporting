"""Against common practice at equal or lower cost (Table 7). No model calls; reuses stored responses.
  dense count (SCP-dense, 50 images, per-type count error):
      naive            = stated count of a count-first response (1 call)
      self-consistency = per-type median of the stated counts of 3 count-first responses (3 calls)
      ours             = list-first response, Step 3 readout (1 call; readout confirmation run)
  clicks (ScreenSpot-Pro, 200 items, click error):
      naive            = one click (1 call)
      self-consistency = vote of 3 clicks, points within 1.5% of the longer side agree (3 calls)
      gated zoom       = zoom only when two clicks disagree by more than 1.5% (uncertainty-gated zoom; 2 or 3 calls)
      ours             = two-stage zoom on every item (2 calls)
  set tasks (6 sources x 3 models, E and F1, 3 calls each except single):
      majority vote    = clusters supported by >= 2 of 3 calls, base-record coordinates (self-consistency for sets)
      box fusion       = same clusters, coordinates averaged with equal weights (weighted box fusion without scores)
      ours             = leave-one-source-out default: 3-call unanimity with fused coordinates on over-reporting sources,
                         single call on the under-reporting (counting) source
Paired image bootstrap (4,000). Output: out/common_practice.json"""
import json, sys, math, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from degen import degenerate, T5
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
OUT = Path(__file__).parent / 'out'; rng = np.random.default_rng(20261013)
SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
R = C.ROOT


def boot(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / max(1e-9, a[s].mean()))
    return dict(a=round(float(a.mean()), 4), b=round(float(b.mean()), 4), rel=round(float((b.mean() - a.mean()) / max(1e-9, a.mean())) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(a))


res = {}
# ---------------------------------------------------------------- dense count
GT = {i: collections.Counter(s['type'] for s in json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes']) for i in range(50)}
err = lambda cnt, i: sum(abs(cnt.get(t, 0) - GT[i][t]) for t in T5)
stated = collections.defaultdict(dict)
for sub in ('b1', 'b6'):
    Q = C.load_requests(sub, 'main'); RS = C.load_responses(sub, 'main')
    for rid, q in Q.items():
        mt = q['meta']
        if mt.get('task') != 'scpd' or mt['arm'] != 'a': continue
        r = RS.get(rid)
        if not r or r['status'] != 'completed': continue
        _, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent'])
        if isinstance(dec, dict):
            stated[(q['model_key'], mt['item'])][mt['rep']] = ({t: int(dec.get(t, 0) or 0) for t in T5}, C.cost(q['model_key'], r))
Q = C.load_requests('rc16', 'main'); RS = C.load_responses('rc16', 'main'); ours = {}
for rid, q in Q.items():
    mt = q['meta']; r = RS.get(rid)
    if not r or r['status'] != 'completed': continue
    recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, 1024, 1024)
    mrecs = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs]; mp = collections.Counter(x['type'] for x in mrecs if x['type'] in T5)
    ok = isinstance(dec, dict); w = {t: int(dec.get(t, 0) or 0) for t in T5} if ok else None
    fl = degenerate(mrecs, sum(w.values()) if ok else None, r, True)
    ours[(q['model_key'], mt['item'])] = (w if (fl and ok) else mp, C.cost(q['model_key'], r))
res['dense_count'] = {}
for m in C.MODELS:
    ids = [i for i in range(50) if all(k in stated.get((m, i), {}) for k in (0, 1, 2)) and (m, i) in ours]
    nv = [err(stated[(m, i)][0][0], i) for i in ids]
    sc = [err({t: float(np.median([stated[(m, i)][k][0][t] for k in (0, 1, 2)])) for t in T5}, i) for i in ids]
    ou = [err(ours[(m, i)][0], i) for i in ids]
    cost = dict(naive=float(np.mean([stated[(m, i)][0][1] for i in ids])), self_consistency=float(np.mean([sum(stated[(m, i)][k][1] for k in (0, 1, 2)) for i in ids])),
                ours=float(np.mean([ours[(m, i)][1] for i in ids])))
    res['dense_count'][m] = dict(n=len(ids), naive=round(float(np.mean(nv)), 2), self_consistency=round(float(np.mean(sc)), 2), ours=round(float(np.mean(ou)), 2),
                                 self_consistency_vs_naive=boot(nv, sc), ours_vs_naive=boot(nv, ou), ours_vs_self_consistency=boot(sc, ou),
                                 cost={k: round(v, 4) for k, v in cost.items()})

# ---------------------------------------------------------------- clicks
GUI = '/home/claude/gui'
CI = {c['id']: c for c in json.load(open(f'{GUI}/data/click_items_prepped.json'))}
QA, RA, QB, RBc = {}, {}, {}, {}
for l in open(f'{GUI}/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for l in open(f'{GUI}/runs/main/requests_B.jsonl'): q = json.loads(l); QB[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'{GUI}/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
    for l in open(f'{GUI}/runs/main/responses_B_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RBc[r['request_id']] = r
Qb6 = C.load_requests('b6', 'main'); Rb6 = C.load_responses('b6', 'main')
def scc(q): return None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
res['clicks'] = {}
for m in C.MODELS:
    per = collections.defaultdict(list); cst = collections.defaultdict(list)
    for cid, c in CI.items():
        if c['source'] != 'sspro': continue
        k = lambda s: f'{cid}|{m}|{s}'
        if not (k('C1') in RA and k('C2') in RA and k('C3') in Rb6 and k('B4') in RBc): continue
        pts = [G.decode_click(RA[k('C1')]['raw_text'], scc(QA[k('C1')]))[0], G.decode_click(RA[k('C2')]['raw_text'], scc(QA[k('C2')]))[0],
               G.decode_click(Rb6[k('C3')]['raw_text'], scc(Qb6[k('C3')]))[0]]
        bb = c['bbox_norm']; W, H = c['width'], c['height']
        def d(p, q): return math.hypot((p[0] - q[0]) * W, (p[1] - q[1]) * H) / max(W, H)
        valid = [p for p in pts if p is not None]
        vote = valid[int(np.argmax([sum(1 for q in valid if d(p, q) <= 0.015) for p in valid]))] if valid else None
        q4 = QB[k('B4')]; pt, _ = G.decode_click(RBc[k('B4')]['raw_text'], scc(q4)); w = q4['meta']['window']
        p4 = [w[0] + pt[0] * (w[2] - w[0]), w[1] + pt[1] * (w[3] - w[1])] if pt is not None else pts[0]
        disagree = pts[0] is None or pts[1] is None or d(pts[0], pts[1]) > 0.015
        gate = p4 if disagree else pts[0]
        c1, c2, c3, cz = (C.cost(m, RA[k('C1')]), C.cost(m, RA[k('C2')]), C.cost(m, Rb6[k('C3')]), C.cost(m, RBc[k('B4')]))
        per['naive'].append(1 - int(G.hit(pts[0], bb))); per['self_consistency'].append(1 - int(G.hit(vote, bb)))
        per['gated_zoom'].append(1 - int(G.hit(gate, bb))); per['ours'].append(1 - int(G.hit(p4, bb)))
        cst['naive'].append(c1); cst['self_consistency'].append(c1 + c2 + c3); cst['gated_zoom'].append(c1 + c2 + (cz if disagree else 0)); cst['ours'].append(c1 + cz)
    out = dict(n=len(per['naive']), **{a: round(float(np.mean(v)), 4) for a, v in per.items()}, cost={a: round(float(np.mean(v)), 4) for a, v in cst.items()})
    for a in ('self_consistency', 'gated_zoom', 'ours'): out[f'{a}_vs_naive'] = boot(per['naive'], per[a])
    out['ours_vs_self_consistency'] = boot(per['self_consistency'], per['ours']); out['ours_vs_gated_zoom'] = boot(per['gated_zoom'], per['ours'])
    res['clicks'][m] = out

# ---------------------------------------------------------------- set tasks
ct = json.load(open(OUT / 'construction_table.json')); pi = ct['per_image']; cells = ct['cells']
SRC = [('E3', 'doclaynet'), ('E3', 'dota'), ('E3', 'fsc147'), ('E3', 'sku110k'), ('GUI', 'groundcua'), ('GUI', 'screenparse')]
methods = {'majority_vote': 'maj3_del', 'box_fusion': 'maj3_fus'}
res['sets'] = {}
for fam, src in SRC:
    for m in C.MODELS:
        key = f'{fam}|{src}|{m}'; P = pi[key]; ids = sorted(P['single'])
        ours_c = {'E': 'single' if src == 'fsc147' else 'unan3_fus', 'F1': 'single'}
        row = {}
        for name, cc in list(methods.items()) + [('ours', ours_c)]:
            for li, loss in ((0, 'E'), (1, 'F1')):
                c = cc[loss] if isinstance(cc, dict) else cc
                a = [P['single'][i][li] for i in ids]; b = [P[c][i][li] for i in ids]
                row[f'{name}|{loss}'] = boot(a, b) if c != 'single' else dict(a=round(float(np.mean(a)), 4), b=round(float(np.mean(a)), 4), rel=0.0, ci=[0.0, 0.0], n=len(ids))
            row[f'{name}|cost'] = cells[key][cc['E'] if isinstance(cc, dict) else cc]['cost_per_image']
        a = [P['maj3_fus'][i][0] for i in ids]; b = [P[ours_c['E']][i][0] for i in ids]
        row['ours_vs_box_fusion|E'] = boot(a, b)
        row['single|cost'] = cells[key]['single']['cost_per_image']
        res['sets'][key] = row
summ = {}
for name in ('majority_vote', 'box_fusion', 'ours'):
    for loss in ('E', 'F1'):
        rows = [v[f'{name}|{loss}'] for v in res['sets'].values()]
        better = sum((r['ci'][1] < 0) if loss == 'E' else (r['ci'][0] > 0) for r in rows); worse = sum((r['ci'][0] > 0) if loss == 'E' else (r['ci'][1] < 0) for r in rows)
        summ[f'{name}|{loss}'] = dict(median_rel=round(float(np.median([r['rel'] for r in rows])), 1), better=better, worse=worse, ns=len(rows) - better - worse)
rows = [v['ours_vs_box_fusion|E'] for v in res['sets'].values()]
summ['ours_vs_box_fusion|E'] = dict(median_rel=round(float(np.median([r['rel'] for r in rows])), 1), better=sum(r['ci'][1] < 0 for r in rows), worse=sum(r['ci'][0] > 0 for r in rows))
res['sets_summary'] = summ
json.dump(res, open(OUT / 'common_practice.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'sets'}, indent=1)[:6000])
