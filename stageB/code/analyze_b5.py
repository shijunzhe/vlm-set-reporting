"""B5 analysis: hit rate of the second-stage click at three sizes of the same window (native from the GUI run, up, down).
Usage: python -I analyze_b5.py <subset>"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
SUB = sys.argv[1] if len(sys.argv) > 1 else 'b5'
Q = C.load_requests(SUB, 'main'); RS = C.load_responses(SUB, 'main'); rng = np.random.default_rng(0)
GUI = '/home/claude/gui'; QB, RB = {}, {}
for l in open(f'{GUI}/runs/main/requests_B.jsonl'):
    q = json.loads(l)
    if q['meta']['arm'] == 'B4': QB[(q['meta']['item'], q['model_key'])] = q
for m in C.MODELS:
    for l in open(f'{GUI}/runs/main/responses_B_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed' and r['request_id'].endswith('|B4'): RB[(r['request_id'].split('|')[0], m)] = r


def hit(raw, mode, sent, w, bb):
    pt, _ = G.decode_click(raw, None if mode == 'grid' else tuple(sent))
    if pt is None: return 0
    return int(G.hit([w[0] + pt[0] * (w[2] - w[0]), w[1] + pt[1] * (w[3] - w[1])], bb))


rows = collections.defaultdict(dict); spend = collections.Counter(); scale = collections.defaultdict(dict)
for rid, q in Q.items():
    r = RS.get(rid); mt = q['meta']; m = q['model_key']
    if r is None or r['status'] != 'completed': continue
    spend[m] += C.cost(m, r)
    rows[(mt['item'], m)][mt['cond']] = hit(r['raw_text'], mt['mode'], mt['sent'], mt['window'], mt['bbox'])
    scale[(mt['item'], m)][mt['cond']] = mt['sent'][0] / mt['native_sent'][0]
    if 'native' not in rows[(mt['item'], m)] and (mt['item'], m) in RB:
        qb = QB[(mt['item'], m)]; rows[(mt['item'], m)]['native'] = hit(RB[(mt['item'], m)]['raw_text'], qb['meta']['mode'], qb['meta']['sizes'][0], qb['meta']['window'], mt['bbox'])
out = dict(cost={k: round(v, 2) for k, v in spend.items()}, cells={})
for m in C.MODELS:
    R_ = [v for (i, mm), v in rows.items() if mm == m and all(k in v for k in ('native', 'up', 'down'))]
    if not R_: continue
    cell = {k: round(float(np.mean([v[k] for v in R_])), 3) for k in ('down', 'native', 'up')}
    cell['n'] = len(R_)
    cell['size_ratio_vs_native'] = {k: round(float(np.median([scale[(i, mm)][k] for (i, mm), v in rows.items() if mm == m and k in scale[(i, mm)]])), 3) for k in ('down', 'up')}
    for a, b in (('down', 'native'), ('native', 'up')):
        d = np.array([v[b] - v[a] for v in R_], float); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        cell[f'{b}_minus_{a}'] = dict(mean=round(float(d.mean()), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
    out['cells'][m] = cell
json.dump(out, open(f'{C.ROOT}/runs/{SUB}/b5_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
