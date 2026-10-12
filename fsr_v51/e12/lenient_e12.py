"""Post hoc sensitivity: lenient readout that recovers coordinate pairs from syntactically broken JSON.
Rows: the text is split at each "region_id" key; every u/v (or x/y) numeric pair in a segment belongs to that row.
Free list: every x/y pair. Values outside [0,1) are dropped. Labelled post hoc in the report."""
import json, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, '.')
import analyze_e12 as A
reqs = {json.loads(l)['request_id']: json.loads(l) for l in open('runs_e12/requests.jsonl')}
ann = {json.loads(l)['sample_id']: json.loads(l) for l in open('data_e12/annotations_DEV.jsonl')}
RID = re.compile(r'"region_id"\s*:\s*"(R0\d\d)"')
def pairs(seg, frame):
    k1, k2 = ('u', 'v') if frame == 'local' else ('x', 'y')
    return [(float(a), float(b)) for a, b in re.findall(rf'"{k1}"\s*:\s*(-?[0-9.]+)\s*,\s*"{k2}"\s*:\s*(-?[0-9.]+)', seg)]
def lenient(q, raw):
    out = []
    if q['kind'] == 'free':
        return [(1024 * x, 1024 * y) for x, y in pairs(raw, 'global') if 0 <= x < 1 and 0 <= y < 1]
    ms = list(RID.finditer(raw))
    for i, m in enumerate(ms):
        rid = m.group(1)
        if rid not in q['requested']: continue
        seg = raw[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(raw)]
        for a, b in pairs(seg, q['frame']):
            p = A.to_xy({'u': a, 'v': b} if q['frame'] == 'local' else {'x': a, 'y': b}, q['frame'], rid)
            if p: out.append(p)
    return out
res = {}
for m in ['gpt56', 'gpt61', 'opus55', 'sonnet55']:
    E = defaultdict(lambda: defaultdict(list))
    for l in open(f'runs_e12/responses_{m}.jsonl'):
        r = json.loads(l); q = reqs[r['request_id']]
        if q['focal']: continue
        Y = [tuple(t['point']) for t in ann[q['sample_id']]['targets']]
        strict = [(z['x'], z['y']) for z in A.decode(q, r['raw_text'])[0]]
        E[q['condition']]['strict'].append(A.E(strict, Y)); E[q['condition']]['lenient'].append(A.E(lenient(q, r['raw_text']), Y))
    res[m] = {c: {k: round(float(np.mean(v)), 2) for k, v in d.items()} for c, d in E.items()}
    print(m, res[m])
json.dump(res, open('runs_e12/lenient_sensitivity.json', 'w'), indent=1)
