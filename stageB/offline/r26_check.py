"""Direct test of R26 (selective copying) on B2. For every unperturbed draft record shown to the model in the 'draft' arm
of a crop: was it kept in the model's output; is it true (matches ground truth at IoU 0.5); does the model's own blind
listing of the same crop report it ('agree')? R26 predicts that keeping depends on agreement with the model's own
judgement and hardly on truth once agreement is fixed; foreign records (injected false boxes) are rarely agreed with and
rarely kept. No model calls. Usage: python -I r26_check.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code'); import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
R = C.ROOT; Q = C.load_requests('b2', 'main'); RS = C.load_responses('b2', 'main'); M = json.load(open(f'{R}/runs/b2/items.json')); meta = M['meta']
P = {i['id']: i for i in json.load(open('/home/claude/gui/data/parse_items.json'))}
QA, RA = {}, {}
for l in open('/home/claude/gui/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'/home/claude/gui/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
sys.path.insert(0, '/home/claude/fsr_v51/e3'); os.chdir('/home/claude/fsr_v51/e3')
import analyze_e3 as A0
from score_e3 import load_items
items3 = load_items(); rows3 = A0.load_rows('runs_main/scored.jsonl')
d3 = {(x['item'], x['model'], x['rep']): x for x in rows3 if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min' and x['dataset'] == 'sku110k'}
def decode(raw, mode, sent):
    bx, _ = C.boxes_positional(raw); out = []
    for x1, y1, x2, y2 in bx:
        b = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000] if mode == 'grid' else [x1 / sent[0], y1 / sent[1], x2 / sent[0], y2 / sent[1]]
        if b[2] > b[0] and b[3] > b[1]: out.append(b)
    return out
def to_full(b, c): w, h = c[2] - c[0], c[3] - c[1]; return [c[0] + b[0] * w, c[1] + b[1] * h, c[0] + b[2] * w, c[1] + b[3] * h]
def inside(b, c): cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; return c[0] <= cx < c[2] and c[1] <= cy < c[3]
dec = {}
for rid, q in Q.items():
    r = RS.get(rid)
    if r is None or r['status'] != 'completed': continue
    key = rid.rsplit('|', 1)[0]; dec[rid] = [to_full(b, meta[key]['crop']) for b in decode(r['raw_text'], q['meta']['mode'], q['meta']['sent'])]
cnt = collections.defaultdict(lambda: collections.Counter())
for kind in ('gui', 'sku'):
    for m in C.MODELS:
        for iid in (M['gui_ids'] if kind == 'gui' else M['sku_ids']):
            if kind == 'gui':
                k = f'{iid}|{m}|D1'
                if k not in RA: continue
                draft = [b for b, n in G.decode_list(RA[k]['raw_text'], sc(QA[k])[0])[0]]; gt = P[iid]['targets']
            else:
                x = d3.get((iid, m, 0))
                if x is None: continue
                it = items3[iid]; W, H = it['width'], it['height']
                draft = [[r['box'][0] / W, r['box'][1] / H, r['box'][2] / W, r['box'][3] / H] for r in A0.recs_of(x) if r['box']]
                gt = [[t['box'][0] / W, t['box'][1] / H, t['box'][2] / W, t['box'][3] / H] for t in it['targets']]
            tpd = {a for a, b in G.lex_match(draft, gt, 0.5)}
            for ci in range(4):
                key = f'{kind}|{iid}|{m}|c{ci}'
                if key not in meta or f'{key}|blind' not in dec or f'{key}|draft' not in dec: continue
                mk = meta[key]; c = mk['crop']; B = dec[f'{key}|blind']; D = dec[f'{key}|draft']
                pert = [list(map(float, b)) for b in mk['deleted'] + mk['shifted']]
                for j, b in enumerate(draft):
                    if not inside(b, c) or any(G.iou(b, p) > 0.99 for p in pert): continue
                    keep = any(G.iou(o, b) >= 0.5 for o in D); agree = any(G.iou(o, b) >= 0.5 for o in B)
                    cnt[(kind, m)][('true' if j in tpd else 'false', 'agree' if agree else 'disagree', 'keep' if keep else 'drop')] += 1
                for inj in mk['injected']:
                    keep = any(G.iou(o, inj) >= 0.5 for o in D); agree = any(G.iou(o, inj) >= 0.5 for o in B)
                    cnt[(kind, m)][('injected', 'agree' if agree else 'disagree', 'keep' if keep else 'drop')] += 1
out = {}
for (kind, m), c in cnt.items():
    cell = {}
    for t in ('true', 'false', 'injected'):
        for a in ('agree', 'disagree'):
            k_, d_ = c[(t, a, 'keep')], c[(t, a, 'drop')]
            if k_ + d_: cell[f'{t}|{a}'] = dict(n=k_ + d_, keep_rate=round(k_ / (k_ + d_), 3))
    # logistic-free summary: keep-rate difference due to agreement vs due to truth
    def rate(t, a): v = cell.get(f'{t}|{a}'); return v['keep_rate'] if v and v['n'] >= 10 else None
    eff_agree = [rate(t, 'agree') - rate(t, 'disagree') for t in ('true', 'false') if rate(t, 'agree') is not None and rate(t, 'disagree') is not None]
    eff_truth = [rate('true', a) - rate('false', a) for a in ('agree', 'disagree') if rate('true', a) is not None and rate('false', a) is not None]
    cell['effect_of_agreement'] = round(float(np.mean(eff_agree)), 3) if eff_agree else None
    cell['effect_of_truth_given_agreement'] = round(float(np.mean(eff_truth)), 3) if eff_truth else None
    out[f'{kind}|{m}'] = cell
json.dump(out, open('/home/claude/stageB/offline/r26_check.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
