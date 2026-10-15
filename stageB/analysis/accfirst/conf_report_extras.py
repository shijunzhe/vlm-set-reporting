"""Report items of protocol v3 section 5 that are not tests (descriptive only; does not change H1 to H3).
Comparator accuracy on the drawn representatives, both-order agreement, chain clusters and cluster diameters, items with a
single cluster and their errors, per-flow deployment cost, total collection cost and attempts, results by ui_type and
platform. Latency was not recorded per request by the runner and is not reported.
Usage: python3 accfirst/conf_report_extras.py --subset conf_v1 | main -> accfirst/conf_extras_<subset>.json"""
import argparse, collections, glob, itertools, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
BASE = os.environ.get('FSR_BASE', '/home/claude'); GUI = BASE + '/gui'
sys.path.insert(0, GUI + '/code'); sys.path.insert(0, BASE + '/fsr_v51/e3')
cwd = os.getcwd(); import fusion_stages as F; import gl_stages as GLS; import decision as DC; os.chdir(cwd)
from score_e3 import PRICE
G, B = F.G, F.B
ap = argparse.ArgumentParser(); ap.add_argument('--subset', required=True); a = ap.parse_args(); SUB = a.subset
os.chdir(GUI)
C = F.items_of('main' if SUB == 'main' else 'conf')
if SUB == 'main':
    keep = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}; C = {k: v for k, v in C.items() if k in keep}
items = sorted(C); run = DC.Run(GUI, SUB, G, B)
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))
cost = lambda m, r: (r['usage']['input_tokens'] * PRICE[m][0] + r['usage']['output_tokens'] * PRICE[m][1]) / 1e6
res = {}
# comparator accuracy on drawn representatives
for cfg in ('M', 'D'):
    Q, R, _ = run.stage(f'CMPA_{cfg}'); per = collections.defaultdict(dict)
    for rid, r in R.items():
        q = Q[rid]; v = G.first_json(r['raw_text']); lab = v.get('preferred_id') if isinstance(v, dict) else None
        per[(q['meta']['item'], tuple(q['meta']['pair']))][q['meta']['order']] = (q['meta']['labels'].get(lab), q['meta']['points'])
    single, agree, agree_ok, n = [], 0, 0, 0
    for (c, pair), d in per.items():
        pts = next(iter(d.values()))[1]; h = {i: hit(pts[i], c) for i in pair}
        if sum(h.values()) != 1: continue
        n += 1; right = [i for i in pair if h[i]][0]; single += [int(v[0] == right) for v in d.values()]
        if d.get('ab', (None,))[0] is not None and d.get('ab')[0] == d.get('ba', (None,))[0]: agree += 1; agree_ok += int(d['ab'][0] == right)
    res[f'comparator_{cfg}'] = dict(pairs=len(per), one_right_one_wrong=n, single_call_accuracy=round(float(np.mean(single)), 3) if single else None,
                                   both_orders_agree=agree, agree_correct=agree_ok)
# clusters
for cfg, fl in (('M', 'M-C'), ('D', 'D-C')):
    chain, diam, single_items, single_err = 0, [], 0, 0
    for c in items:
        A, sc, cls, _ = DC.structure(run, cfg, c, C[c]); pts = {x['id']: x['point'] for x in A}
        for cl in cls:
            P = [pts[i] for i in cl['members']]
            d = max([DC.dist(p, q, C[c]) for p, q in itertools.combinations(P, 2)] or [0]); diam.append(d)
            chain += d > DC.DELTA
        if len(cls) == 1:
            single_items += 1; single_err += 1 - hit(cls[0]['rep']['point'], c)
    res[f'clusters_{cfg}'] = dict(chain_clusters=chain, max_diameter_pct=round(100 * max(diam), 2) if diam else None,
                                  single_cluster_items=single_items, single_cluster_errors=single_err)
# costs and attempts
stage_files = collections.defaultdict(dict); att = 0; usd_all = 0.0
LATER = {'SG', 'CMPA_D4'}   # the four-source measurement, run after the protocol run (accfirst/d4_eval.py reports it)
for f in glob.glob(f'runs/{SUB}/responses_*_*.jsonl'):
    st = os.path.basename(f)[len('responses_'):].rsplit('_', 1)[0]
    if st in LATER: continue
    for l in open(f):
        r = json.loads(l); att += r.get('attempts', 1) or 1
        if r['status'] == 'completed':
            m = r['request_id'].split('|')[1]; cst = cost(m, r); usd_all += cst; stage_files[st][r['request_id']] = (cst, r['usage'])
res['collection'] = dict(usd=round(usd_all, 2), physical_attempts=att, completed=sum(len(v) for v in stage_files.values()))
DEPLOY = {'GPT-default-zoom': [('DA', 'gpt61', 'C1d'), ('DB', 'gpt61', 'B4d')],
          'GLF': [('GLF1', 'gpt61', None), ('GLF2', 'gpt61', None), ('GLF3', 'gpt61', None)],
          'GPTx3': [('DA', 'gpt61', None), ('DB', 'gpt61', None)],
          'D-C': [('A', 'gem38', 'C1'), ('B', 'gem38', 'B4'), ('DA', 'gpt61', 'C1d'), ('DB', 'gpt61', 'B4d'), ('DA', 'sonnet55', 'C1d'), ('DB', 'sonnet55', 'B4d'),
                  ('S', 'gem38', 'S|gem38'), ('SD', 'gem38', 'SD'), ('CMPA_D', 'gem38', None)]}
for fl, parts in DEPLOY.items():
    per_item = collections.defaultdict(float); calls = collections.defaultdict(int)
    for st, m, tag in parts:
        for rid, (cst, u) in stage_files.get(st, {}).items():
            p = rid.split('|')
            if p[1] != m: continue
            if tag and '|'.join(p[2:]) != tag and p[2] != tag: continue
            per_item[p[0]] += cst; calls[p[0]] += 1
    res[f'deploy_{fl}'] = dict(usd_per_item=round(float(np.mean([per_item[c] for c in items])), 5), calls_per_item=round(float(np.mean([calls[c] for c in items])), 2))
json.dump(res, open(HERE / f'conf_extras_{SUB}.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
