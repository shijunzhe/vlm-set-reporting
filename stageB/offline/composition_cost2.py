"""Composition cost, second check: are the extra errors of the one-call tiles (GUI T1) localisation errors (box near a
true element, 0.1 <= IoU < 0.5) or invented elements (no true element nearby)? Compared with separate crop calls on the
same 30 GroundCUA screens (B2 blind arm). Also centre offset of near boxes (systematic frame error). No model calls.
Usage: python -I composition_cost2.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code'); import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
P = {i['id']: i for i in json.load(open('/home/claude/gui/data/parse_items.json'))}
QA, RA = {}, {}
for l in open('/home/claude/gui/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'/home/claude/gui/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
Q2 = C.load_requests('b2', 'main'); R2 = C.load_responses('b2', 'main'); M = json.load(open('/home/claude/stageB/runs/b2/items.json')); meta = M['meta']
def to_full(b, c): w, h = c[2] - c[0], c[3] - c[1]; return [c[0] + b[0] * w, c[1] + b[1] * h, c[0] + b[2] * w, c[1] + b[3] * h]
def dec2(raw, mode, sent):
    bx, _ = C.boxes_positional(raw); out = []
    for x1, y1, x2, y2 in bx:
        b = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000] if mode == 'grid' else [x1 / sent[0], y1 / sent[1], x2 / sent[0], y2 / sent[1]]
        if b[2] > b[0] and b[3] > b[1]: out.append(b)
    return out
def breakdown(S, gt):
    mt = dict(G.lex_match(S, gt, 0.5)); used = set(mt.values()); near = far = 0; offs = []
    for a, b in enumerate(S):
        if a in mt: continue
        best = max(((G.iou(b, g), j) for j, g in enumerate(gt) if j not in used), default=(0, None))
        if best[0] >= 0.1:
            near += 1; g = gt[best[1]]; offs.append(((b[0] + b[2]) / 2 - (g[0] + g[2]) / 2, (b[1] + b[3]) / 2 - (g[1] + g[3]) / 2))
        else: far += 1
    return len(mt), near, far, offs
out = {}
for m in C.MODELS:
    acc = {'onecall': collections.Counter(), 'separate': collections.Counter()}; offs = {'onecall': [], 'separate': []}
    for iid in M['gui_ids']:
        gt = P[iid]['targets']; k = f'{iid}|{m}|T1'
        keys = [f'gui|{iid}|{m}|c{ci}|blind' for ci in range(4)]
        if k not in RA or not all(x in R2 and R2[x]['status'] == 'completed' for x in keys): continue
        s = sc(QA[k]); T1 = [b for b, n in G.decode_tiles(RA[k]['raw_text'], s if s[0] else None)[0]]
        SEP = []
        for x in keys:
            c = meta[x.rsplit('|', 1)[0]]['crop']; SEP += [to_full(b, c) for b in dec2(R2[x]['raw_text'], Q2[x]['meta']['mode'], Q2[x]['meta']['sent'])]
        for name, S in (('onecall', T1), ('separate', SEP)):
            tp, near, far, o = breakdown(S, gt); acc[name]['tp'] += tp; acc[name]['near'] += near; acc[name]['far'] += far; acc[name]['n'] += 1; offs[name] += o
    if acc['onecall']['n']:
        out[m] = {name: dict(screens=a['n'], tp=round(a['tp'] / a['n'], 1), fp_near=round(a['near'] / a['n'], 1), fp_far=round(a['far'] / a['n'], 1),
                             mean_offset_near=[round(float(np.mean([x for x, _ in offs[name]])) * 1000, 1), round(float(np.mean([y for _, y in offs[name]])) * 1000, 1)] if offs[name] else None)
                  for name, a in acc.items()}
json.dump(out, open('/home/claude/stageB/offline/composition_cost2.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
