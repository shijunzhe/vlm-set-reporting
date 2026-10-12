"""B6 end-to-end, cost-tracked comparison (labels used only in scoring and in the optional 10-label arm).
Tasks x arms:
  SCP-dense count  naive = written count, count-first, 1 call | common = median of 3 written counts | FSR = code count of the
                   scan-ordered list (1 call) and median of 3 code counts | FSR+10 labels = per-type ratio calibration
  FSC count        naive = written count (1 call) | common = median of 3 written counts | FSR = tiled list, code count (4 calls)
                   | FSR+10 labels = ratio calibration of the tiled count
  SKU detection    naive = single call | common = majority of 3 | FSR = unanimous of 3 (one-bit rule, over-report regime)
                   | FSR cross-view = single call  intersect separate-call crops (B2 images) | FSR+10 labels = record gate (E3 L10)
  GroundCUA parse  naive = D1 | common = majority of D1-D3 | FSR = D1 intersect D2 (one-bit rule) | FSR cross-view = D1 intersect crops
                   (B2 screens) | FSR+10 labels = menu choice on 10 labelled screens
  Click (SS-Pro)   naive = C1 | common = 3-click vote | FSR = two-stage zoom
Usage: python -I analyze_b6.py"""
import json, os, sys, collections, random, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
R = C.ROOT; rng = np.random.default_rng(0); rr = random.Random(0)
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
OUT = {}


def boot_rel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / max(1e-9, a[s].mean()))
    return dict(a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / max(1e-9, a.mean())) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(a))


def summarize(per, lower_better=True, key='err'):
    """per: {arm: {unit: value}} -> means and relative changes vs naive and vs common."""
    arms = list(per); common_units = sorted(set.intersection(*[set(per[a]) for a in arms]))
    res = {a: round(float(np.mean([per[a][u] for u in common_units])), 3) for a in arms}; res['units'] = len(common_units)
    for ref in ('naive', 'common'):
        if ref not in per: continue
        for a in arms:
            if a in ('naive', 'common') and a == ref: continue
            res[f'{a}_vs_{ref}'] = boot_rel([per[ref][u] for u in common_units], [per[a][u] for u in common_units])
    return res


# ------------------------------------------------------------- SCP-dense counting
Qb1 = C.load_requests('b1', 'main'); Rb1 = C.load_responses('b1', 'main'); Qb6 = C.load_requests('b6', 'main'); Rb6 = C.load_responses('b6', 'main')
GTd = {i: json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json')) for i in range(50)}
spend = collections.defaultdict(float)
def scp_counts(q, r):
    recs, dec, _ = C.decode_shapes(r['raw_text'], q['meta']['mode'], 1024, 1024, *q['meta']['sent'])
    code = collections.Counter(x['type'] for x in recs if x['type'] in T5)
    written = {t: int(dec.get(t, 0) or 0) for t in T5} if isinstance(dec, dict) else None
    return code, written
calls = collections.defaultdict(dict)
for Q_, R_ in ((Qb1, Rb1), (Qb6, Rb6)):
    for rid, q in Q_.items():
        if q['meta'].get('task') != 'scpd' or q['meta']['arm'] not in ('a', 'i'): continue
        r = R_.get(rid)
        if r is None or r['status'] != 'completed': continue
        calls[(q['meta']['item'], q['model_key'])][(q['meta']['arm'], q['meta']['rep'])] = (scp_counts(q, r), C.cost(q['model_key'], r))
for m in C.MODELS:
    per = collections.defaultdict(dict); cst = collections.defaultdict(list); exact = collections.defaultdict(dict); fsr_counts = {}
    for i in range(50):
        c = calls.get((i, m), {})
        if not all(k in c for k in (('a', 0), ('a', 1), ('a', 2), ('i', 0), ('i', 1), ('i', 2))): continue
        g = collections.Counter(s['type'] for s in GTd[i]['shapes'])
        def err(cnt): return sum(abs(cnt.get(t, 0) - g[t]) for t in T5)
        wa = [c[('a', k)][0][1] for k in range(3)]
        if any(w is None for w in wa): continue
        med_w = {t: float(np.median([w[t] for w in wa])) for t in T5}
        ci_ = [c[('i', k)][0][0] for k in range(3)]; med_c = {t: float(np.median([x.get(t, 0) for x in ci_])) for t in T5}
        per['naive'][i] = err(wa[0]); per['common'][i] = err(med_w); per['fsr_1call'][i] = err(ci_[0]); per['fsr_3call'][i] = err(med_c)
        for a, cnt in (('naive', wa[0]), ('common', med_w), ('fsr_1call', ci_[0]), ('fsr_3call', med_c)): exact[a][i] = int(err(cnt) == 0)
        cst['naive'].append(c[('a', 0)][1]); cst['common'].append(sum(c[('a', k)][1] for k in range(3)))
        cst['fsr_1call'].append(c[('i', 0)][1]); cst['fsr_3call'].append(sum(c[('i', k)][1] for k in range(3)))
        fsr_counts[i] = (ci_[0], g)
    if not per: continue
    # FSR + 10 labels: per-type ratio calibration of the single-call code count, 200 random label sets
    ids = sorted(fsr_counts); cal = []
    for _ in range(200):
        lab = rr.sample(ids, 10); rest = [i for i in ids if i not in lab]
        ratio = {t: (sum(fsr_counts[i][1][t] for i in lab) / max(1, sum(fsr_counts[i][0].get(t, 0) for i in lab))) for t in T5}
        cal.append(np.mean([sum(abs(round(fsr_counts[i][0].get(t, 0) * ratio[t]) - fsr_counts[i][1][t]) for t in T5) for i in rest]) /
                   max(1e-9, np.mean([per['fsr_1call'][i] for i in rest])) - 1)
    s = summarize(per); s['exact_count_rate'] = {a: round(float(np.mean(list(v.values()))), 3) for a, v in exact.items()}
    s['cost_per_image'] = {a: round(float(np.mean(v)), 4) for a, v in cst.items()}
    s['fsr_10label_calibration_vs_fsr_1call'] = dict(median_rel=round(float(np.median(cal)) * 100, 1), worse_share=round(float(np.mean(np.array(cal) > 0)), 3))
    OUT[f'scpd|{m}'] = s
# ------------------------------------------------------------- FSC counting
Qp = C.load_requests('b1p', 'main'); Rp = C.load_responses('b1p', 'main'); items = json.load(open(f'{R}/runs/b1p/items.json'))
fsc = collections.defaultdict(dict)
for Q_, R_ in ((Qp, Rp), (Qb6, Rb6)):
    for rid, q in Q_.items():
        if q['meta'].get('task') != 'fsc': continue
        r = R_.get(rid)
        if r is None or r['status'] != 'completed': continue
        mt = q['meta']; it = items[mt['item']]
        if mt['arm'] == 'cf':
            recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], it['W'], it['H'], *mt['sent'], list_key='points', count_key='count')
            try: dec = int(dec)
            except Exception: dec = None
            fsc[(mt['item'], q['model_key'])][('cf', mt.get('rep', 0))] = (dec, len(recs), C.cost(q['model_key'], r))
        elif mt['arm'] == 'lo':
            recs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], it['W'], it['H'], *mt['sent'], list_key='points', count_key='count')
            fsc[(mt['item'], q['model_key'])][('lo', 0)] = (None, len(recs), C.cost(q['model_key'], r))
        else:
            c = mt['crop']; cw, ch = (c[2] - c[0]) * it['W'], (c[3] - c[1]) * it['H']
            recs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], cw, ch, *mt['sent'], list_key='points', count_key='count')
            fsc[(mt['item'], q['model_key'])][('tl', tuple(c))] = ([dict(cx=x['cx'] + c[0] * it['W'], cy=x['cy'] + c[1] * it['H'], crop=tuple(c)) for x in recs], C.cost(q['model_key'], r))
def merge(parts, tau):
    out = []
    for P in parts:
        for p in P:
            if not any(q['crop'] != p['crop'] and math.hypot(q['cx'] - p['cx'], q['cy'] - p['cy']) <= tau for q in out): out.append(p)
    return out
for m in C.MODELS:
    per = collections.defaultdict(dict); cst = collections.defaultdict(list); tlc = {}
    for iid, it in items.items():
        d = fsc.get((iid, m), {})
        cfs = [d.get(('cf', k)) for k in (0, 1, 2)]; tls = [v for k, v in d.items() if k[0] == 'tl']
        if any(x is None or x[0] is None for x in cfs) or len(tls) != 4 or ('lo', 0) not in d: continue
        n = it['n']; tl_n = len(merge([v[0] for v in tls], it['tau']))
        per['naive'][iid] = abs(cfs[0][0] - n); per['common'][iid] = abs(float(np.median([x[0] for x in cfs])) - n)
        per['fsr_list_1call'][iid] = abs(d[('lo', 0)][1] - n); per['fsr_tiles'][iid] = abs(tl_n - n); tlc[iid] = (tl_n, n)
        cst['naive'].append(cfs[0][2]); cst['common'].append(sum(x[2] for x in cfs)); cst['fsr_list_1call'].append(d[('lo', 0)][2]); cst['fsr_tiles'].append(sum(v[1] for v in tls))
    if not per: continue
    ids = sorted(tlc); cal = []
    for _ in range(200):
        lab = rr.sample(ids, 10); rest = [i for i in ids if i not in lab]
        ratio = sum(tlc[i][1] for i in lab) / max(1, sum(tlc[i][0] for i in lab))
        cal.append(np.mean([abs(round(tlc[i][0] * ratio) - tlc[i][1]) for i in rest]) / max(1e-9, np.mean([per['fsr_tiles'][i] for i in rest])) - 1)
    s = summarize(per); s['cost_per_image'] = {a: round(float(np.mean(v)), 4) for a, v in cst.items()}
    s['fsr_10label_calibration_vs_fsr_tiles'] = dict(median_rel=round(float(np.median(cal)) * 100, 1), worse_share=round(float(np.mean(np.array(cal) > 0)), 3))
    OUT[f'fsc|{m}'] = s
# ------------------------------------------------------------- SKU (E3, IoU 0.75 primary) and GroundCUA (IoU 0.5)
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import e3lib as L, analyze_e3 as A0, offline_agg2, aggregate_v2 as AG
from score_e3 import load_items
items3 = load_items(); rows3 = A0.load_rows('runs_main/scored.jsonl')
d3 = {}
for x in rows3:
    if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min' and x['dataset'] == 'sku110k': d3[(x['item'], x['model'], x['rep'])] = x
tok = {(x['item'], x['model'], x['rep']): C.PRICE[x['model']][0] * x['in_tok'] / 1e6 + C.PRICE[x['model']][1] * x['out_tok'] / 1e6 for x in rows3 if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min'}
B2 = json.load(open(f'{R}/runs/b2/b2_analysis.json')) if os.path.exists(f'{R}/runs/b2/b2_analysis.json') else None
b2items = json.load(open(f'{R}/runs/b2/items.json')); Qb2 = C.load_requests('b2', 'main'); Rb2 = C.load_responses('b2', 'main')
def b2_crops(kind, iid, m):
    """merged blind crops (normalized full-image boxes) and their cost, or None."""
    parts = []; cost_ = 0.0
    for ci in range(4):
        rid = f'{kind}|{iid}|{m}|c{ci}|blind'; r = Rb2.get(rid); q = Qb2.get(rid)
        if r is None or r['status'] != 'completed': return None, None
        cost_ += C.cost(m, r); c = b2items['meta'][rid.rsplit('|', 1)[0]]['crop']
        P = []
        for x1, y1, x2, y2 in C.boxes_positional(r['raw_text'])[0]:
            b = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000] if q['meta']['mode'] == 'grid' else [x1 / q['meta']['sent'][0], y1 / q['meta']['sent'][1], x2 / q['meta']['sent'][0], y2 / q['meta']['sent'][1]]
            if b[2] <= b[0] or b[3] <= b[1]: continue
            w, h = c[2] - c[0], c[3] - c[1]; P.append([c[0] + b[0] * w, c[1] + b[1] * h, c[0] + b[2] * w, c[1] + b[3] * h])
        parts.append((ci, P))
    out = []
    for ci, P in parts:
        for b in P:
            if not any(cj != ci and G.iou(b, q) >= 0.5 for cj, q in out): out.append((ci, b))
    return [b for _, b in out], cost_
base = json.load(open('/home/claude/fsr_v51/model511/e3_baselines511.json'))
for m in C.MODELS:
    per = collections.defaultdict(dict); perf = collections.defaultdict(dict); cst = collections.defaultdict(list); per_x = collections.defaultdict(dict)
    for i in sorted({k[0] for k in d3 if k[1] == m}):
        if not all((i, m, r) in d3 for r in (0, 1, 2)): continue
        it = items3[i]; mt = ('box', 0.75); reps = [[r for r in A0.recs_of(d3[(i, m, k)]) if r['box'] is not None] for k in (0, 1, 2)]
        cl3 = AG.build_clusters(it, reps); fz = [AG.fuse(x[1]) for x in cl3]
        cons = {'naive': reps[0], 'common': [f for f, x in zip(fz, cl3) if len(x[2]) >= 2], 'fsr_unanimous3': [f for f, x in zip(fz, cl3) if len(x[2]) == 3]}
        for a, S in cons.items():
            E, tp, cnt = L.error(it, S, mt); per[a][i] = E; perf[a][i] = 2 * tp / (it['n'] + cnt) if it['n'] + cnt else 1.0
        cst['naive'].append(tok[(i, m, 0)]); cst['common'].append(sum(tok[(i, m, k)] for k in range(3))); cst['fsr_unanimous3'].append(sum(tok[(i, m, k)] for k in range(3)))
        if i in b2items['sku_ids']:
            cr, cc = b2_crops('sku', i, m)
            if cr is not None:
                W, H = it['width'], it['height']
                crec = [dict(cx=(b[0] + b[2]) / 2 * W, cy=(b[1] + b[3]) / 2 * H, label=reps[0][0]['label'] if reps[0] else 'product', box=[b[0] * W, b[1] * H, b[2] * W, b[3] * H]) for b in cr]
                cl = AG.build_clusters(it, [reps[0], crec]); S = [AG.fuse(x[1]) for x in cl if len(x[2]) == 2]
                E, tp, cnt = L.error(it, S, mt); per_x['fsr_crossview'][i] = E; per_x['naive'][i] = per['naive'][i]; per_x['common'][i] = per['common'][i]
                cst['fsr_crossview'].append(tok[(i, m, 0)] + cc)
    s = summarize(per); s['F1'] = {a: round(float(np.mean(list(v.values()))), 4) for a, v in perf.items()}
    s['cost_per_image'] = {a: round(float(np.mean(v)), 4) for a, v in cst.items()}
    if per_x: s['crossview_subset'] = summarize(per_x)
    s['fsr_10label_gate_rel_vs_single'] = base[f'sku110k|{m}']['gate_L10']; s['common_rel_vs_single_E3'] = base[f'sku110k|{m}']['maj3']
    OUT[f'sku|{m}'] = s
GUI = '/home/claude/gui'; Pg = {i['id']: i for i in json.load(open(f'{GUI}/data/parse_items.json'))}
QA, RA = {}, {}
for l in open(f'{GUI}/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'{GUI}/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
for m in C.MODELS:
    per = collections.defaultdict(dict); perf = collections.defaultdict(dict); cst = collections.defaultdict(list); per_x = collections.defaultdict(dict); allcons = {}
    for iid, it in Pg.items():
        if it['source'] != 'groundcua': continue
        k = lambda s: f'{iid}|{m}|{s}'
        if not all(k(f'D{j}') in RA for j in (1, 2, 3)): continue
        D = [G.decode_list(RA[k(f'D{j}')]['raw_text'], sc(QA[k(f'D{j}')])[0])[0] for j in (1, 2, 3)]; gt = it['targets']
        cons = {'naive': [b for b, n in D[0]], 'common': G.construct(D, 'majority'), 'fsr_inter2': G.construct(D[:2], 'inter')}
        for a, S in cons.items():
            s_ = G.score_boxes(S, gt); per[a][iid] = s_['E']; perf[a][iid] = s_['f1']
        c1 = C.cost(m, RA[k('D1')]); cst['naive'].append(c1); cst['common'].append(sum(C.cost(m, RA[k(f'D{j}')]) for j in (1, 2, 3))); cst['fsr_inter2'].append(c1 + C.cost(m, RA[k('D2')]))
        if iid in b2items['gui_ids']:
            cr, cc = b2_crops('gui', iid, m)
            if cr is not None:
                S = G.construct([D[0], [(b, '') for b in cr]], 'inter'); s_ = G.score_boxes(S, gt)
                per_x['fsr_crossview'][iid] = s_['E']; per_x['naive'][iid] = per['naive'][iid]; per_x['common'][iid] = per['common'][iid]
                per_x['fsr_crops_only'][iid] = G.score_boxes(cr, gt)['E']; cst['fsr_crossview'].append(c1 + cc)
                allcons[iid] = dict(naive=per['naive'][iid], common=per['common'][iid], fsr_inter2=per['fsr_inter2'][iid], fsr_crossview=s_['E'], crops=per_x['fsr_crops_only'][iid])
    s = summarize(per); s['F1'] = {a: round(float(np.mean(list(v.values()))), 4) for a, v in perf.items()}
    s['cost_per_image'] = {a: round(float(np.mean(v)), 4) for a, v in cst.items()}
    if per_x: s['crossview_subset'] = summarize(per_x)
    # FSR + 10 labels: choose among constructions on 10 labelled screens (screens with all options), score on the rest
    ids = sorted(allcons); gains = []
    for _ in range(200):
        if len(ids) < 15: break
        lab = rr.sample(ids, 10); rest = [i for i in ids if i not in lab]
        best = min(('naive', 'common', 'fsr_inter2', 'fsr_crossview', 'crops'), key=lambda a: np.mean([allcons[i][a] for i in lab]))
        gains.append(np.mean([allcons[i][best] for i in rest]) / np.mean([allcons[i]['naive'] for i in rest]) - 1)
    if gains: s['fsr_10label_menu_vs_naive'] = dict(median_rel=round(float(np.median(gains)) * 100, 1), worse_share=round(float(np.mean(np.array(gains) > 0)), 3))
    OUT[f'groundcua|{m}'] = s
# ------------------------------------------------------------- click
CI = {c['id']: c for c in json.load(open(f'{GUI}/data/click_items_prepped.json'))}
QB, RBc = {}, {}
for l in open(f'{GUI}/runs/main/requests_B.jsonl'): q = json.loads(l); QB[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'{GUI}/runs/main/responses_B_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RBc[r['request_id']] = r
def scc(q): return None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
for m in C.MODELS:
    per = collections.defaultdict(dict); cst = collections.defaultdict(list)
    for cid, c in CI.items():
        if c['source'] != 'sspro': continue
        k = lambda s: f'{cid}|{m}|{s}'
        if not (k('C1') in RA and k('C2') in RA and k('C3') in Rb6 and k('B4') in RBc): continue
        pts = [G.decode_click(RA[k('C1')]['raw_text'], scc(QA[k('C1')]))[0], G.decode_click(RA[k('C2')]['raw_text'], scc(QA[k('C2')]))[0],
               G.decode_click(Rb6[k('C3')]['raw_text'], scc(Qb6[k('C3')]))[0]]
        bb = c['bbox_norm']; W, H = c['width'], c['height']
        def d(p, q): return math.hypot((p[0] - q[0]) * W, (p[1] - q[1]) * H) / max(W, H)
        valid = [p for p in pts if p is not None]
        if valid:
            sup = [sum(1 for q in valid if d(p, q) <= 0.015) for p in valid]; vote = valid[int(np.argmax(sup))]
        else: vote = None
        q4 = QB[k('B4')]; pt, _ = G.decode_click(RBc[k('B4')]['raw_text'], scc(q4)); w = q4['meta']['window']
        p4 = [w[0] + pt[0] * (w[2] - w[0]), w[1] + pt[1] * (w[3] - w[1])] if pt is not None else pts[0]
        per['naive'][cid] = 1 - int(G.hit(pts[0], bb)); per['common'][cid] = 1 - int(G.hit(vote, bb)); per['fsr_zoom'][cid] = 1 - int(G.hit(p4, bb))
        cst['naive'].append(C.cost(m, RA[k('C1')])); cst['common'].append(C.cost(m, RA[k('C1')]) + C.cost(m, RA[k('C2')]) + C.cost(m, Rb6[k('C3')])); cst['fsr_zoom'].append(C.cost(m, RA[k('C1')]) + C.cost(m, RBc[k('B4')]))
    if not per: continue
    s = summarize(per); s['note'] = 'values are click error rates'; s['cost_per_image'] = {a: round(float(np.mean(v)), 4) for a, v in cst.items()}
    OUT[f'click|{m}'] = s
json.dump(OUT, open(f'{R}/runs/b6/b6_analysis.json', 'w'), indent=1)
for k, v in OUT.items():
    print(k, {a: v[a] for a in v if not isinstance(v[a], dict)}, {a: (v[a]['rel'], v[a]['ci']) for a in v if isinstance(v[a], dict) and 'rel' in v[a]})
