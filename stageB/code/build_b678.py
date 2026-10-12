"""Manifests for B6 (extra calls for the end-to-end comparison), B7 (default reasoning) and B8 (Qwen3-VL). No model calls.
B6 new calls: SCP-dense arms a and i, reps 1-2 (50 images); FSC count-first reps 1-2 (100 images); ScreenSpot-Pro third click C3
(200 items, same prompt as C1/C2). SKU-110K and GroundCUA reuse existing E3 / GUI / B2 outputs.
B7 (setting 'default' = API default reasoning): SCP-dense 30 images x arms a, c, d, i, k1; FSC 30 images x cf, lo, tl (4 crops);
GroundCUA 20 screens x D1, D2.
B8 (Qwen3-VL-235B-A22B instruct, grid contract): SCP arms a, c, d, k1, k2 (100 images); SCP-dense arms a, d, i (50 images).
Usage: python -I build_b678.py"""
import json, os, sys, copy, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
R = C.ROOT; GUI = '/home/claude/gui'
b1 = [json.loads(l) for l in open(f'{R}/runs/b1/requests_main.jsonl')]
b1p = [json.loads(l) for l in open(f'{R}/runs/b1p/requests_main.jsonl')]
b6 = []
for q in b1:
    if q['meta']['task'] == 'scpd' and q['meta']['arm'] in ('a', 'i'):
        for rep in (1, 2):
            x = copy.deepcopy(q); x['request_id'] = q['request_id'].replace('|r0', f'|r{rep}'); x['meta']['rep'] = rep; b6.append(x)
for q in b1p:
    if q['meta']['arm'] == 'cf':
        for rep in (1, 2):
            x = copy.deepcopy(q); x['request_id'] = q['request_id'].replace('|c0', f'|c0r{rep}'); x['meta']['rep'] = rep; b6.append(x)
for l in open(f'{GUI}/runs/main/requests_A.jsonl'):
    q = json.loads(l)
    if q['meta'].get('arm') == 'click' and q['meta']['rep'] == 1 and q['meta']['item'].startswith('sspro'):
        x = copy.deepcopy(q); x['request_id'] = q['request_id'].replace('|C1', '|C3'); x['meta']['rep'] = 3
        x['images'] = [f'{GUI}/{p}' for p in q['images']]; x['setting'] = 'min'; x['meta']['task'] = 'click3'; b6.append(x)
C.write_manifest(f'{R}/runs/b6/requests_main.jsonl', b6)
# ---------------- B7
b7 = []
for q in b1:
    if q['meta']['task'] == 'scpd' and q['meta']['item'] < 30 and q['meta']['arm'] in ('a', 'c', 'd', 'i', 'k1'):
        x = copy.deepcopy(q); x['setting'] = 'default'; x['request_id'] = 'def|' + q['request_id']; b7.append(x)
fsc_items = sorted({q['meta']['item'] for q in b1p})
rng = random.Random(77); f30 = set(rng.sample(fsc_items, 30))
for q in b1p:
    if q['meta']['item'] in f30:
        x = copy.deepcopy(q); x['setting'] = 'default'; x['request_id'] = 'def|' + q['request_id']; b7.append(x)
P = {i['id']: i for i in json.load(open(f'{GUI}/data/parse_items.json'))}
g20 = sorted(i for i, it in P.items() if it['source'] == 'groundcua')[::3][:20]
for l in open(f'{GUI}/runs/main/requests_A.jsonl'):
    q = json.loads(l)
    if q['meta'].get('item') in g20 and q['meta'].get('arm') == 'draft' and q['request_id'].split('|')[-1] in ('D1', 'D2'):
        x = copy.deepcopy(q); x['setting'] = 'default'; x['request_id'] = 'def|' + q['request_id']; x['images'] = [f'{GUI}/{p}' for p in q['images']]
        x['meta']['task'] = 'gui_draft'; b7.append(x)
C.write_manifest(f'{R}/runs/b7/requests_main.jsonl', b7)
# ---------------- B8 (Qwen)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
b8 = []
for q in b1:
    if q['model_key'] != 'gpt61': continue                     # GPT uses the grid contract; reuse its prompts for Qwen
    t, arm = q['meta']['task'], q['meta']['arm']
    if (t == 'scp' and arm in ('a', 'c', 'd', 'k1', 'k2') and q['meta']['rep'] == 0) or (t == 'scpd' and arm in ('a', 'd', 'i')):
        x = copy.deepcopy(q); x['model_key'] = 'qwen3vl'; x['request_id'] = q['request_id'].replace('|gpt61|', '|qwen3vl|'); b8.append(x)
C.write_manifest(f'{R}/runs/b8/requests_main.jsonl', b8)
from collections import Counter
print('b6', len(b6), Counter(q['meta'].get('task') for q in b6)); print('b7', len(b7), Counter(q['meta'].get('task') for q in b7)); print('b8', len(b8))
