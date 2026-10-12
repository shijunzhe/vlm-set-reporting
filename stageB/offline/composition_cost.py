"""Mechanism of the composition cost (four crops in one call vs separate calls), GUI parsing, GroundCUA + ScreenParse.
For the one-call tiles response T1: for each element box that is a false positive under its stated crop, re-map it into
each of the other three crops; if it then matches a ground-truth element (IoU >= 0.5) not matched by any other record,
the error is a crop-index (output contract) error, not a perception error. Also per-crop record counts (report budget).
No model calls. Usage: python -I composition_cost.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
os.chdir(G.ROOT)
MODELS = ['gpt61', 'sonnet55', 'gem38']
P = {i['id']: i for i in json.load(open('data/parse_items.json'))}
QA, RA = {}, {}
for l in open('runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in MODELS:
    for l in open(f'runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
out = {}
for m in MODELS:
    tot = collections.Counter()
    for iid, it in P.items():
        k = f'{iid}|{m}|T1'
        if k not in RA: continue
        gt = it['targets']; scales = sc(QA[k]); scales = scales if scales[0] else None
        v = G.first_json(RA[k]['raw_text'])
        if not (isinstance(v, dict) and isinstance(v.get('crops'), list)): continue
        recs = []
        for row in v['crops']:
            if not isinstance(row, dict) or row.get('crop') not in (1, 2, 3, 4): continue
            ci = row['crop'] - 1
            for o in row.get('elements') or []:
                if not isinstance(o, dict): continue
                b = G.norm_box(o, [0], scales[ci] if scales else None)
                if b is not None: recs.append((ci, b))
        full = [G.crop_to_full(b, G.CROPS[ci]) for ci, b in recs]
        matched = {a: j for a, j in G.lex_match(full, gt, 0.5)}; used = set(matched.values())
        per_crop = collections.Counter(ci for ci, _ in recs)
        tot['records'] += len(recs); tot['tp'] += len(matched); tot['screens'] += 1
        for a, (ci, b) in enumerate(recs):
            if a in matched: continue
            tot['fp'] += 1
            for cj in range(4):
                if cj == ci: continue
                fb = G.crop_to_full(b, G.CROPS[cj])
                hits = [j for j, g in enumerate(gt) if j not in used and G.iou(fb, g) >= 0.5]
                if hits: tot['fp_fixed_by_other_crop'] += 1; used.add(hits[0]); break
        tot['crop_count_spread'] += float(np.std([per_crop[c] for c in range(4)]))
    if tot['screens']:
        out[m] = dict(screens=tot['screens'], records_per_screen=round(tot['records'] / tot['screens'], 1), tp_per_screen=round(tot['tp'] / tot['screens'], 1),
                      fp_per_screen=round(tot['fp'] / tot['screens'], 1), share_fp_explained_by_crop_index=round(tot['fp_fixed_by_other_crop'] / max(1, tot['fp']), 3))
json.dump(out, open('/home/claude/stageB/offline/composition_cost.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
