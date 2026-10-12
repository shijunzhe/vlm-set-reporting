"""B3 analysis: comparator go / no-go. Usage: python -I analyze_b3.py <subset>"""
import json, os, sys, re, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
SUB = sys.argv[1] if len(sys.argv) > 1 else 'b3'
Q = C.load_requests(SUB, 'main'); RS = C.load_responses(SUB, 'main'); meta = json.load(open(f'{C.ROOT}/runs/{SUB}/items.json'))
rng = np.random.default_rng(0)


def choice(raw):
    objs = re.findall(r'\{[^{}]*"choice"[^{}]*\}', raw or '')
    for o in reversed(objs):
        try: return str(json.loads(o)['choice']).upper().strip()
        except Exception: continue
    m = re.findall(r'\b(FIRST|SECOND|NEITHER|TIE)\b', (raw or '').upper())
    return m[-1] if m else None


spend = collections.Counter(); ans = {}
for rid, q in Q.items():
    r = RS.get(rid)
    if r is None or r['status'] != 'completed': continue
    spend[q['model_key']] += C.cost(q['model_key'], r); ans[rid] = choice(r['raw_text'])
out = dict(cost={k: round(v, 2) for k, v in spend.items()}, click={}, block={})
# ---------------- click
for m in C.MODELS:
    rows = []
    for key, v in meta['click'].items():
        cid, mm = key.split('|')
        if mm != m: continue
        a1 = ans.get(f'click|{cid}|{m}|c1b4'); a2 = ans.get(f'click|{cid}|{m}|b4c1')
        if a1 is None or a2 is None: continue
        pick1 = {'FIRST': 'c1', 'SECOND': 'b4'}.get(a1, 'none'); pick2 = {'FIRST': 'b4', 'SECOND': 'c1'}.get(a2, 'none')
        h1 = int(G.hit(v['p1'], v['bbox'])); h4 = int(G.hit(v['p4'], v['bbox']))
        consistent = pick1 == pick2 and pick1 != 'none'
        final = pick1 if consistent else 'b4'
        rows.append(dict(h1=h1, h4=h4, comp=(h1 if final == 'c1' else h4), first_pos=int(a1 == 'FIRST') + int(a2 == 'FIRST'),
                         consistent=int(consistent), single=(h1 if pick1 == 'c1' else h4 if pick1 == 'b4' else h4), oracle=max(h1, h4)))
    if not rows: continue
    n = len(rows)
    out['click'][m] = dict(differing_items=n, acc_c1=round(np.mean([r['h1'] for r in rows]), 3), acc_b4=round(np.mean([r['h4'] for r in rows]), 3),
                           acc_comparator=round(np.mean([r['comp'] for r in rows]), 3), acc_single_order=round(np.mean([r['single'] for r in rows]), 3),
                           acc_oracle=round(np.mean([r['oracle'] for r in rows]), 3), consistency=round(np.mean([r['consistent'] for r in rows]), 3),
                           first_position_rate=round(np.mean([r['first_pos'] for r in rows]) / 2, 3),
                           comp_minus_b4=int(sum(r['comp'] - r['h4'] for r in rows)), extra_calls=2 * n)
# ---------------- truth at IoU 0.5 for the block candidates (the E3 primary metric for boxes is IoU 0.75)
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import e3lib as L, analyze_e3 as A0, offline_agg2, aggregate_v2 as AG
from score_e3 import load_items
sys.path.insert(0, '/home/claude/fsr_v51/model512'); from stageA_e3_rel import relation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
items3 = load_items(); rows3 = A0.load_rows('runs_main/scored.jsonl')
d3 = {(x['item'], x['model'], x['rep']): x for x in rows3 if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min'}
tp50 = {}
for bid in meta['block']:
    ds, i, m, ci_ = bid.split('|'); ci_ = int(ci_); it = items3[i]
    reps = [[r for r in A0.recs_of(d3[(i, m, k)]) if r['box'] is not None] for k in (0, 1, 2)]
    cl = AG.build_clusters(it, reps); fused = [AG.fuse(x[1]) for x in cl]
    Rm = relation(it, fused, ('box', 0.5))
    mm = maximum_bipartite_matching(csr_matrix(Rm.astype(np.int8)), perm_type='row') if Rm.size and Rm.any() else np.array([-1])
    tp50[bid] = int(ci_ in set(int(v) for v in mm if v >= 0))
for bid in meta['block']: meta['block'][bid]['tp75'] = meta['block'][bid]['tp']
# ---------------- blocks
for key in sorted({(v['block'].split('|')[0], v['block'].split('|')[2], thr) for v in (q['meta'] for q in Q.values()) if v['task'] == 'block' for thr in ('iou75', 'iou50')}):
    ds, m, thr = key; rows = []
    for bid, v in meta['block'].items():
        b_ds, _, b_m, _ = bid.split('|')
        if (b_ds, b_m) != (ds, m): continue
        a1 = ans.get(f'block|{bid}|keepdrop'); a2 = ans.get(f'block|{bid}|dropkeep')
        if a1 is None or a2 is None: continue
        k1 = {'FIRST': 1, 'SECOND': 0}.get(a1); k2 = {'FIRST': 0, 'SECOND': 1}.get(a2)
        consistent = k1 is not None and k1 == k2
        maj = int(v['support'] == 2)
        comp = k1 if consistent else maj
        rows.append(dict(tp=(v['tp75'] if thr == 'iou75' else tp50[bid]), sup=v['support'], comp=comp, maj=maj, consistent=int(consistent), first=int(a1 == 'FIRST') + int(a2 == 'FIRST'), strict=(k1 if consistent else 0)))
    if not rows: continue
    def acc(key): return float(np.mean([int(r[key] == r['tp']) for r in rows]))
    def gain_vs_drop(key): return float(np.mean([(1 if r['tp'] else -1) * r[key] for r in rows]))   # E reduction per decision vs dropping all
    keep_all = float(np.mean([(1 if r['tp'] else -1) for r in rows]))
    tp_rate = float(np.mean([r['tp'] for r in rows]))
    bs = []
    for _ in range(2000):
        s = rng.integers(0, len(rows), len(rows)); bs.append(np.mean([(1 if rows[i]['tp'] else -1) * (rows[i]['comp'] - rows[i]['maj']) for i in s]))
    out['block'][f'{ds}|{m}|{thr}'] = dict(blocks=len(rows), tp_rate=round(tp_rate, 3), acc_comparator=round(acc('comp'), 3), acc_majority=round(acc('maj'), 3),
                                     acc_drop_all=round(1 - tp_rate, 3), acc_keep_all=round(tp_rate, 3), acc_strict=round(acc('strict'), 3),
                                     consistency=round(float(np.mean([r['consistent'] for r in rows])), 3),
                                     first_position_rate=round(float(np.mean([r['first'] for r in rows])) / 2, 3),
                                     gain_per_decision_vs_drop=dict(comparator=round(gain_vs_drop('comp'), 3), majority=round(gain_vs_drop('maj'), 3),
                                                                    keep_all=round(keep_all, 3), oracle=round(tp_rate, 3)),
                                     comp_minus_majority_per_decision=dict(mean=round(float(np.mean([(1 if r['tp'] else -1) * (r['comp'] - r['maj']) for r in rows])), 3),
                                                                            ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]),
                                     tp_keep_rate=round(float(np.mean([r['comp'] for r in rows if r['tp']])), 3) if any(r['tp'] for r in rows) else None,
                                     fp_keep_rate=round(float(np.mean([r['comp'] for r in rows if not r['tp']])), 3) if any(not r['tp'] for r in rows) else None)
json.dump(out, open(f'{C.ROOT}/runs/{SUB}/b3_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
