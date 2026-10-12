"""B7 GUI parsing under 'default' vs 'min': paired bootstrap of D1 error, FP / FN split, thinking tokens actually used,
and the same-setting noise floor (D1 vs D2, both 'min'). No model calls. Usage: python -I reasoning_gui.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code'); import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
Q7 = C.load_requests('b7', 'main'); R7 = C.load_responses('b7', 'main')
QA, RA = {}, {}
for l in open('/home/claude/gui/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'/home/claude/gui/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
P = {i['id']: i for i in json.load(open('/home/claude/gui/data/parse_items.json'))}
rng = np.random.default_rng(0); out = {}
def boot(d):
    d = np.array(d); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return dict(mean=round(float(d.mean()), 2), ci=[round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)], n=len(d))
for m in C.MODELS:
    dd, noise, fp, fn, th = [], [], [], [], []
    for rid, q in Q7.items():
        if q['model_key'] != m or q['meta'].get('task') != 'gui_draft' or not rid.endswith('|D1'): continue
        iid = q['meta']['item']; gt = P[iid]['targets']; r1 = R7.get(rid); o1, o2 = RA.get(rid[4:]), RA.get(rid[4:-1] + '2')
        if not (r1 and o1 and o2): continue
        sd = G.score_boxes([b for b, n in G.decode_list(r1['raw_text'], sc(Q7[rid])[0])[0]], gt)
        s1 = G.score_boxes([b for b, n in G.decode_list(o1['raw_text'], sc(QA[rid[4:]])[0])[0]], gt)
        s2 = G.score_boxes([b for b, n in G.decode_list(o2['raw_text'], sc(QA[rid[4:-1] + '2'])[0])[0]], gt)
        dd.append(sd['E'] - s1['E']); noise.append(s2['E'] - s1['E']); fp.append(sd['fp'] - s1['fp']); fn.append(sd['fn'] - s1['fn'])
        u = r1['usage']; th.append(u.get('thinking_tokens') or u.get('reasoning_tokens') or 0)
    out[m] = dict(default_minus_min_E=boot(dd), same_setting_D2_minus_D1_E=boot(noise), dFP=boot(fp), dFN=boot(fn),
                  thinking_share=round(float(np.mean(np.array(th) > 0)), 2), thinking_mean=int(np.mean(th)))
json.dump(out, open('/home/claude/stageB/offline/reasoning_gui.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
