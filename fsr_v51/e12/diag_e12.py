"""Descriptive diagnostics for E12 (post hoc, labelled as such in the report). Output: runs_e12/diag_e12.json."""
import json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.')
import analyze_e12 as A

reqs = {json.loads(l)['request_id']: json.loads(l) for l in open('runs_e12/requests.jsonl')}
ann = {json.loads(l)['sample_id']: json.loads(l) for l in open('data_e12/annotations_DEV.jsonl')}
PRICE = {'gpt56': (4, 20), 'gpt61': (2, 10), 'opus55': (4, 20), 'sonnet55': (2, 10)}
out = {}
for m in ['gpt56', 'gpt61', 'opus55', 'sonnet55']:
    R = [json.loads(l) for l in open(f'runs_e12/responses_{m}.jsonl')]
    o = {}
    # cost and tokens
    ti = sum((r.get('usage') or {}).get('input_tokens') or 0 for r in R); to = sum((r.get('usage') or {}).get('output_tokens') or 0 for r in R)
    th = [((r.get('usage') or {}).get('thinking_tokens', (r.get('usage') or {}).get('reasoning_tokens')) or 0) for r in R]
    o['tokens'] = {'input': ti, 'output': to, 'calls_with_thinking_or_reasoning': sum(1 for x in th if x > 0), 'thinking_or_reasoning_tokens': sum(th),
                   'cost_usd_list_price': round(ti / 1e6 * PRICE[m][0] + to / 1e6 * PRICE[m][1], 2)}
    per = defaultdict(list)
    for r in R:
        q = reqs[r['request_id']]
        if q['focal']: continue
        recs, au = A.decode(q, r['raw_text'] if r['status'] == 'completed' else '')
        Y = [tuple(t['point']) for t in ann[q['sample_id']]['targets']]; P = [(z['x'], z['y']) for z in recs]
        mt = {j for _, j in A.match(P, Y, A.TAU)}
        top = [j for j, t in enumerate(Y) if t[1] < 512]; bot = [j for j, t in enumerate(Y) if t[1] >= 512]
        acc = A.regional(recs, Y, 'geometry')
        # regions where every target is missed (whole-region omission) and regions with >50% placement error
        reg_miss = reg_bad = 0
        for g in {A.region_of(*t) for t in Y}:
            yy = [t for t in Y if A.region_of(*t) == g]; pp = [p for p, z in zip(P, recs) if z['geo'] == g]
            reg_miss += int(len(pp) == 0)
            reg_bad += int(len(pp) > 0 and A.TP(pp, yy) < .5 * min(len(pp), len(yy)))
        per[q['condition']].append(dict(dose=q['dose'], n=len(Y), r=len(P), E=A.E(P, Y), count=acc['count_part'], place=acc['placement_part'],
                                       recall_top=len([j for j in top if j in mt]) / max(1, len(top)), recall_bottom=len([j for j in bot if j in mt]) / max(1, len(bot)),
                                       short=int(len(P) < .8 * len(Y)), regions_all_missed=reg_miss, regions_placement_bad=reg_bad,
                                       status=au['status'], out_tokens=(r.get('usage') or {}).get('output_tokens')))
    o['full_image'] = {}
    for c, v in per.items():
        o['full_image'][c] = {k: round(float(np.mean([x[k] for x in v])), 3) for k in ('n', 'r', 'E', 'count', 'place', 'recall_top', 'recall_bottom', 'short', 'regions_all_missed', 'regions_placement_bad', 'out_tokens')}
        o['full_image'][c]['reports'] = len(v)
        o['full_image'][c]['E_by_dose'] = {dz: round(float(np.mean([x['E'] for x in v if x['dose'] == dz])), 2) for dz in (96, 192)}
    out[m] = o
json.dump(out, open('runs_e12/diag_e12.json', 'w'), indent=1)
for m, o in out.items():
    print('=====', m, o['tokens'])
    for c, x in o['full_image'].items(): print(f'  {c:13s}', x)
