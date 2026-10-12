"""B7: default reasoning vs the 'min' setting on the same items. SCP-dense arms a, c, d, i, k1 (B1 items 0-29); FSC cf, lo,
tl (30 items); GroundCUA D1, D2 (20 screens). Usage: python -I analyze_b7.py"""
import json, os, sys, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
R = C.ROOT; T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']; rng = np.random.default_rng(0)
Q7 = C.load_requests('b7', 'main'); R7 = C.load_responses('b7', 'main'); Q1 = C.load_requests('b1', 'main'); R1 = C.load_responses('b1', 'main')
Qp = C.load_requests('b1p', 'main'); Rp = C.load_responses('b1p', 'main'); items = json.load(open(f'{R}/runs/b1p/items.json'))
GTd = {i: json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json')) for i in range(50)}
out = dict(cost={}, scpd={}, fsc={}, gui={})
spend = collections.Counter()
for rid, r in R7.items():
    if r['status'] == 'completed': spend[Q7[rid]['model_key']] += C.cost(Q7[rid]['model_key'], r)
out['cost'] = {k: round(v, 2) for k, v in spend.items()}
def scpd_metrics(q, r):
    allr, dec, _ = C.decode_shapes(r['raw_text'], q['meta']['mode'], 1024, 1024, *q['meta']['sent'])
    g = collections.Counter(s['type'] for s in GTd[q['meta']['item']]['shapes']); lc = collections.Counter(x['type'] for x in allr if x['type'] in T5)
    code = sum(abs(lc[t] - g[t]) for t in T5); written = sum(abs(int(dec.get(t, 0) or 0) - g[t]) for t in T5) if isinstance(dec, dict) else None
    hint = q['meta'].get('hint'); n = sum(g.values())
    return dict(code=code, written=written, k=len(allr), n=n, hint=hint, think=(r['usage'].get('thinking_tokens') or r['usage'].get('reasoning_tokens')), out_tok=r['usage'].get('output_tokens'))
for m in C.MODELS:
    cell = {}
    for arm in ('a', 'c', 'd', 'i', 'k1'):
        A_, B_ = [], []
        for i in range(30):
            rid = f'scpd|{i:03d}|{m}|{arm}|r0'
            r0, r7 = R1.get(rid), R7.get('def|' + rid)
            if not r0 or not r7 or r0['status'] != 'completed' or r7['status'] != 'completed': continue
            A_.append(scpd_metrics(Q1[rid], r0)); B_.append(scpd_metrics(Q7['def|' + rid], r7))
        if not A_: continue
        cell[arm] = dict(n=len(A_), code_min=round(float(np.mean([x['code'] for x in A_])), 2), code_default=round(float(np.mean([x['code'] for x in B_])), 2),
                         written_min=(round(float(np.mean([x['written'] for x in A_ if x['written'] is not None])), 2) if A_[0]['written'] is not None else None),
                         written_default=(round(float(np.mean([x['written'] for x in B_ if x['written'] is not None])), 2) if B_[0]['written'] is not None and any(x['written'] is not None for x in B_) else None),
                         mean_out_tokens_default=round(float(np.mean([x['out_tok'] or 0 for x in B_])), 0))
        if arm == 'k1':
            xs = np.array([x['hint'] - x['n'] for x in B_], float); ys = np.array([x['k'] - x['n'] for x in B_], float)
            cell['commitment_weight_default'] = round(float(np.polyfit(xs, ys, 1)[0]), 3) if len(set(xs)) > 1 else None
    out['scpd'][m] = cell
def merge(parts, tau):
    o = []
    for P in parts:
        for p in P:
            if not any(q['crop'] != p['crop'] and math.hypot(q['cx'] - p['cx'], q['cy'] - p['cy']) <= tau for q in o): o.append(p)
    return o
def fsc_counts(QQ, RR, prefix, m, iid):
    it = items[iid]; res = {}
    for arm in ('cf', 'lo'):
        rid = f'{prefix}fsc|{iid}|{m}|{arm}|c0'; r = RR.get(rid)
        if not r or r['status'] != 'completed': return None
        recs, dec, _ = C.decode_shapes(r['raw_text'], QQ[rid]['meta']['mode'], it['W'], it['H'], *QQ[rid]['meta']['sent'], list_key='points', count_key='count')
        res[arm + '_list'] = abs(len(recs) - it['n'])
        if arm == 'cf':
            try: res['cf_written'] = abs(int(dec) - it['n'])
            except Exception: return None
    parts = []
    for ci in range(4):
        rid = f'{prefix}fsc|{iid}|{m}|tl|c{ci}'; r = RR.get(rid)
        if not r or r['status'] != 'completed': return None
        c = QQ[rid]['meta']['crop']; cw, ch = (c[2] - c[0]) * it['W'], (c[3] - c[1]) * it['H']
        recs, _, _ = C.decode_shapes(r['raw_text'], QQ[rid]['meta']['mode'], cw, ch, *QQ[rid]['meta']['sent'], list_key='points', count_key='count')
        parts.append([dict(cx=x['cx'] + c[0] * it['W'], cy=x['cy'] + c[1] * it['H'], crop=tuple(c)) for x in recs])
    res['tl'] = abs(len(merge(parts, it['tau'])) - it['n'])
    return res
for m in C.MODELS:
    A_, B_ = [], []
    for iid in items:
        b = fsc_counts(Q7, R7, 'def|', m, iid)
        if b is None: continue
        a = fsc_counts(Qp, Rp, '', m, iid)
        if a is None: continue
        A_.append(a); B_.append(b)
    if A_: out['fsc'][m] = {k: dict(min=round(float(np.mean([x[k] for x in A_])), 2), default=round(float(np.mean([x[k] for x in B_])), 2)) for k in A_[0]} | {'n': len(A_)}
QA, RA = {}, {}
for l in open('/home/claude/gui/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'/home/claude/gui/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
P = {i['id']: i for i in json.load(open('/home/claude/gui/data/parse_items.json'))}
for m in C.MODELS:
    rows = collections.defaultdict(list)
    for rid, q in Q7.items():
        if q['model_key'] != m or q['meta'].get('task') != 'gui_draft' or not rid.endswith('|D1'): continue
        iid = q['meta']['item']; gt = P[iid]['targets']
        r1, r2 = R7.get(rid), R7.get(rid[:-1] + '2'); o1, o2 = RA.get(rid[4:]), RA.get(rid[4:-1] + '2')
        if not (r1 and r2 and o1 and o2): continue
        Dd = [G.decode_list(x['raw_text'], sc(Q7[k])[0])[0] for x, k in ((r1, rid), (r2, rid[:-1] + '2'))]
        Dm = [G.decode_list(x['raw_text'], sc(QA[k])[0])[0] for x, k in ((o1, rid[4:]), (o2, rid[4:-1] + '2'))]
        for name, S in (('D1_min', [b for b, n in Dm[0]]), ('D1_default', [b for b, n in Dd[0]]), ('inter_min', G.construct(Dm, 'inter')), ('inter_default', G.construct(Dd, 'inter'))):
            s = G.score_boxes(S, gt); rows[name + '_E'].append(s['E']); rows[name + '_F1'].append(s['f1'])
    out['gui'][m] = {k: round(float(np.mean(v)), 3) for k, v in rows.items()} | {'n': len(rows.get('D1_min_E', []))}
json.dump(out, open(f'{R}/runs/b7/b7_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
