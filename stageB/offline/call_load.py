"""Per-call load: is the gain of separate crop calls over one call with the same four crops larger on screens with more
elements? Per screen (B2 GroundCUA 30 + GUI T1): true positives of separate calls minus one call, against the number of
ground-truth elements. Also whole image vs one call. No model calls. Usage: python -I call_load.py"""
import json, sys
import numpy as np
from scipy.stats import spearmanr
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
out = {}
for m in C.MODELS:
    n, g_sep, g_full = [], [], []
    for iid in M['gui_ids']:
        gt = P[iid]['targets']; k = f'{iid}|{m}|T1'; kd = f'{iid}|{m}|D1'; keys = [f'gui|{iid}|{m}|c{ci}|blind' for ci in range(4)]
        if k not in RA or kd not in RA or not all(x in R2 for x in keys): continue
        s = sc(QA[k]); T1 = [b for b, _ in G.decode_tiles(RA[k]['raw_text'], s if s[0] else None)[0]]
        D1 = [b for b, _ in G.decode_list(RA[kd]['raw_text'], sc(QA[kd])[0])[0]]
        SEP = []
        for x in keys: SEP += [to_full(b, meta[x.rsplit('|', 1)[0]]['crop']) for b in dec2(R2[x]['raw_text'], Q2[x]['meta']['mode'], Q2[x]['meta']['sent'])]
        tp = lambda S: len(G.lex_match(S, gt, 0.5))
        n.append(len(gt)); g_sep.append(tp(SEP) - tp(T1)); g_full.append(tp(D1) - tp(T1))
    r1 = spearmanr(n, g_sep); r2 = spearmanr(n, g_full)
    out[m] = dict(screens=len(n), mean_gain_separate_vs_onecall=round(float(np.mean(g_sep)), 1), spearman_n_vs_gain=dict(rho=round(float(r1.correlation), 3), p=round(float(r1.pvalue), 4)),
                  mean_gain_whole_vs_onecall=round(float(np.mean(g_full)), 1), spearman_n_vs_whole_gain=dict(rho=round(float(r2.correlation), 3), p=round(float(r2.pvalue), 4)))
json.dump(out, open('/home/claude/stageB/offline/call_load.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
