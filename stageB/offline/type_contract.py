"""Label-vocabulary ('type contract') check across all B1 SCP / SCP-dense arms. For every arm: share of records whose
type is outside the five requested names (by name), typed error as scored, and typed error after mapping 'diamond' to
'square' (a rotated square). If arm differences vanish after the mapping, they came from the label vocabulary, not
from finding shapes. No model calls. Usage: python -I type_contract.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
Q = C.load_requests('b1', 'main'); RS = C.load_responses('b1', 'main')
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']; TOL = {'scp': 30, 'scpd': 20}
GT = {('scp', i): json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(100)}
GT.update({('scpd', i): json.load(open(f'/home/claude/stageB/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes'] for i in range(50)})
rng = np.random.default_rng(0)
per = collections.defaultdict(lambda: collections.defaultdict(list)); voc = collections.defaultdict(collections.Counter)
for rid, q in Q.items():
    mt = q['meta']
    if mt['task'] not in ('scp', 'scpd'): continue
    r = RS.get(rid)
    if not r or r['status'] != 'completed': continue
    allrecs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent'])
    G = GT[(mt['task'], mt['item'])]; m = q['model_key']; key = (mt['task'], m, mt['arm'])
    for x in allrecs: voc[key][x['type'] if x['type'] in T5 else str(x['type'])[:12]] += 1
    def E(recs):
        rr = [x for x in recs if x['type'] in T5]; mm = C.match_typed(rr, G, TOL[mt['task']]); return len(rr) + len(G) - 2 * len(mm)
    mapped = [dict(x, type='square') if x['type'] == 'diamond' else x for x in allrecs]
    per[key][mt['item']].append(dict(E=E(allrecs), E_map=E(mapped), diamond=sum(x['type'] == 'diamond' for x in allrecs),
                                     offvocab=sum(x['type'] not in T5 for x in allrecs), k=len(allrecs)))


def pd(task, m, a1, a2, k):
    A = {i: np.mean([x[k] for x in L]) for i, L in per[(task, m, a1)].items()}; B = {i: np.mean([x[k] for x in L]) for i, L in per[(task, m, a2)].items()}
    ids = sorted(set(A) & set(B)); d = np.array([B[i] - A[i] for i in ids]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return dict(a=round(float(np.mean([A[i] for i in ids])), 3), b=round(float(np.mean([B[i] for i in ids])), 3), diff=round(float(d.mean()), 3),
                ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])


out = {'levels': {}, 'contrasts': {}}
for (task, m, arm), D in sorted(per.items()):
    out['levels'][f'{task}|{m}|{arm}'] = {k: round(float(np.mean([np.mean([x[k] for x in L]) for L in D.values()])), 3) for k in ('E', 'E_map', 'diamond', 'offvocab', 'k')}
for task in ('scp', 'scpd'):
    for m in C.MODELS:
        for a1, a2 in (('d', 'j'), ('d', 'i'), ('i', 'j'), ('a', 'd'), ('c', 'd'), ('f', 'g'), ('d', 'g')):
            if per[(task, m, a1)] and per[(task, m, a2)]:
                out['contrasts'][f'{task}|{m}|{a2}_vs_{a1}'] = dict(E=pd(task, m, a1, a2, 'E'), E_map=pd(task, m, a1, a2, 'E_map'))
json.dump(out, open('/home/claude/stageB/offline/type_contract.json', 'w'), indent=1)
for k, v in out['levels'].items():
    if 'sonnet' in k: print(k, v)
for k, v in out['contrasts'].items():
    print(k, 'E', v['E']['diff'], v['E']['ci'], '| E_map', v['E_map']['diff'], v['E_map']['ci'])
