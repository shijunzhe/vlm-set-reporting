"""Click: cross-model agreement gate combined with two-stage zoom, simulated on existing calls (every item has C1, C2 and the
zoom refinement B4 for every model). No model calls. Policies for model A:
  single: C1 | zoom_all: B4 for every item | same_gate: zoom only if A's C1 and C2 disagree | cross_gate(B): zoom only if A's C1 and
  B's C1 disagree. Accuracy (hit rate), paired bootstrap vs zoom_all, mean cost per item.
usage: python -I xmodel_click_zoom.py"""
import json, os, sys, math
import numpy as np
sys.path.insert(0, '/home/claude/gui/code')
import fsrgui as G
os.chdir(G.ROOT)
M = ['gpt61', 'sonnet55', 'gem38']; TAU = 0.015; rng = np.random.default_rng(0)
rows = {(r['model'], r['item']): r for r in json.load(open('runs/main/click_rows.json'))}
C = {c['id']: c for c in json.load(open('data/click_items_prepped.json'))}
Q, R = {}, {}
for l in open('runs/main/requests_A.jsonl'): q = json.loads(l); Q[q['request_id']] = q
for m in M:
    for l in open(f'runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': R[r['request_id']] = r
sc = lambda q: None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
def pt(cid, m, s):
    k = f'{cid}|{m}|{s}'; return G.decode_click(R[k]['raw_text'], sc(Q[k]))[0] if k in R else None
def agree(p, q, c): return p is not None and q is not None and math.hypot((p[0] - q[0]) * c['width'], (p[1] - q[1]) * c['height']) / max(c['width'], c['height']) <= TAU
items = [cid for cid in C if all((m, cid) in rows for m in M)]
P = {(cid, m, s): pt(cid, m, s) for cid in items for m in M for s in ('C1', 'C2')}
def boot(a, b):
    a = np.array(a, float); b = np.array(b, float); d = b - a; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return dict(diff=round(float(d.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
out = dict(n=len(items))
for a in M:
    h1 = [rows[(a, c)]['h1'] for c in items]; h4 = [rows[(a, c)]['h4'] for c in items]
    c1 = [rows[(a, c)]['c1'] for c in items]; c2 = [rows[(a, c)]['c2'] for c in items]; c4 = [rows[(a, c)]['c4'] for c in items]
    res = dict(single=dict(acc=round(float(np.mean(h1)), 4), cost=round(float(np.mean(c1)), 5)),
               zoom_all=dict(acc=round(float(np.mean(h4)), 4), cost=round(float(np.mean(np.add(c1, c4))), 5)))
    dis = [not agree(P[(c, a, 'C1')], P[(c, a, 'C2')], C[c]) for c in items]
    hs = [h4[k] if dis[k] else h1[k] for k in range(len(items))]
    res['same_gate'] = dict(acc=round(float(np.mean(hs)), 4), zoom_share=round(float(np.mean(dis)), 3), cost=round(float(np.mean([c1[k] + c2[k] + c4[k] * dis[k] for k in range(len(items))])), 5), vs_zoom_all=boot(h4, hs))
    for b in M:
        if b == a: continue
        cb = [rows[(b, c)]['c1'] for c in items]
        dis = [not agree(P[(c, a, 'C1')], P[(c, b, 'C1')], C[c]) for c in items]
        hc = [h4[k] if dis[k] else h1[k] for k in range(len(items))]
        res[f'cross_gate|{b}'] = dict(acc=round(float(np.mean(hc)), 4), zoom_share=round(float(np.mean(dis)), 3),
                                      cost=round(float(np.mean([c1[k] + cb[k] + c4[k] * dis[k] for k in range(len(items))])), 5), vs_zoom_all=boot(h4, hc), vs_single=boot(h1, hc))
    out[a] = res; print(a, json.dumps(res))
json.dump(out, open('/home/claude/stageB/xmodel/xmodel_click_zoom.json', 'w'), indent=1)
# alarm after zoom: does disagreement between two zoomed points (A's B4 vs B's B4, or A's B4 vs A's C2-based? only one zoom per model)
QB, RB = {}, {}
for l in open('runs/main/requests_B.jsonl'): q = json.loads(l); QB[q['request_id']] = q
for m in M:
    for l in open(f'runs/main/responses_B_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RB[r['request_id']] = r
def zpt(cid, m):
    p1 = P[(cid, m, 'C1')]; k = f'{cid}|{m}|B4'; r4 = RB.get(k)
    if r4 is None or p1 is None: return p1
    q4 = QB[k]; p, _ = G.decode_click(r4['raw_text'], sc(q4)); w = q4['meta']['window']
    return [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else p1
Z = {(c, m): zpt(c, m) for c in items for m in M}
post = {}
for a in M:
    errs = [c for c in items if not G.hit(Z[(c, a)], C[c]['bbox_norm'])]
    for b in M:
        if b == a: continue
        al = [c for c in items if not agree(Z[(c, a)], Z[(c, b)], C[c])]
        caught = sum(1 for c in errs if c in set(al))
        acc_act = np.mean([G.hit(Z[(c, a)], C[c]['bbox_norm']) for c in items if c not in set(al)])
        post[f'{a}|{b}'] = dict(errors=len(errs), caught=caught, caught_share=round(caught / max(1, len(errs)), 3), coverage=round(1 - len(al) / len(items), 3), acc_when_acting=round(float(acc_act), 4))
    # same-model after zoom: A's zoom point vs A's C2 (second unzoomed click) as the only available same-model second report
    al = [c for c in items if not agree(Z[(c, a)], P[(c, a, 'C2')], C[c])]
    post[f'{a}|same_C2'] = dict(errors=len(errs), caught=sum(1 for c in errs if c in set(al)), coverage=round(1 - len(al) / len(items), 3))
out['alarm_after_zoom'] = post
print(json.dumps(post, indent=0))
json.dump(out, open('/home/claude/stageB/xmodel/xmodel_click_zoom.json', 'w'), indent=1)
