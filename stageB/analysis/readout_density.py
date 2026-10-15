"""Same-response readout gap across object density (Figure 4, Table 15). No model calls. For every response that contains both a list
and a stated count (count-first prompts: B1 arm a on PixMo, SCP, SCP-dense; B1' arm cf on FSC-147), error of the stated count vs
error of the code count of the list in the same response. SCP / SCP-dense: per-type count error with the declared alias map;
PixMo / FSC: total count error. code = alias-mapped code count; code_raw = unmapped (pure readout, as in Table 7). Paired image bootstrap. Output: out/readout_density.json"""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
OUT = Path(__file__).parent / 'out'; rng = np.random.default_rng(20261012)
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']; SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
GT = {('scp', i): collections.Counter(s['type'] for s in json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json'))['shapes']) for i in range(100)}
GT.update({('scpd', i): collections.Counter(s['type'] for s in json.load(open(f'{C.ROOT}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes']) for i in range(50)})
items_p = json.load(open(f'{C.ROOT}/runs/b1p/items.json'))
res = {}
Q1 = C.load_requests('b1', 'main'); R1 = C.load_responses('b1', 'main'); Qp = C.load_requests('b1p', 'main'); Rp = C.load_responses('b1p', 'main')
def boot(a, b, c=None):
    extra = {}
    if c is not None:
        cc = np.array(c, float); aa = np.array(a, float); bsr = []
        for _ in range(4000):
            s = rng.integers(0, len(aa), len(aa)); bsr.append((cc[s].mean() - aa[s].mean()) / aa[s].mean() if aa[s].mean() else 0)
        extra = dict(code_raw=round(float(cc.mean()), 3), rel_raw=round(float((cc.mean() - aa.mean()) / aa.mean()) * 100, 1) if aa.mean() else None,
                     ci_raw=[round(float(np.percentile(bsr, 2.5)) * 100, 1), round(float(np.percentile(bsr, 97.5)) * 100, 1)])
    a = np.array(a, float); b = np.array(b, float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / a[s].mean() if a[s].mean() else 0)
    return dict(n=len(a), stated=round(float(a.mean()), 3), code=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / a.mean()) * 100, 1) if a.mean() else None,
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], **extra)
for m in C.MODELS:
    for task in ('px', 'scp', 'scpd'):
        A, B, N, Craw = [], [], [], []
        for rid, q in Q1.items():
            mt = q['meta']
            if q['model_key'] != m or mt['arm'] != 'a' or mt['rep'] != 0 or mt['task'] != {'px': 'pixmo', 'scp': 'scp', 'scpd': 'scpd'}[task]: continue
            r = R1.get(rid)
            if not r or r['status'] != 'completed': continue
            if task == 'px':
                recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], mt['w'], mt['h'], *mt['sent'], list_key='points', count_key='count')
                try: dec = int(dec)
                except Exception: continue
                A.append(abs(dec - mt['n'])); B.append(abs(len(recs) - mt['n'])); Craw.append(abs(len(recs) - mt['n'])); N.append(mt['n'])
            else:
                recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, 1024, 1024)
                if not isinstance(dec, dict): continue
                g = GT[(task, mt['item'])]; lc = collections.Counter(SYN.get(x['type'], x['type']) for x in recs)
                A.append(sum(abs(int(dec.get(t, 0) or 0) - g[t]) for t in T5)); B.append(sum(abs(lc[t] - g[t]) for t in T5)); N.append(sum(g.values()))
                lr = collections.Counter(x['type'] for x in recs); Craw.append(sum(abs(lr[t] - g[t]) for t in T5))
        res[f'{task}|{m}'] = dict(mean_objects=round(float(np.mean(N)), 1), **boot(A, B, Craw))
    A, B, N = [], [], []
    for iid, it in items_p.items():
        rid = f'fsc|{iid}|{m}|cf|c0'; r = Rp.get(rid)
        if not r or r['status'] != 'completed': continue
        recs, dec, _ = C.decode_shapes(r['raw_text'], Qp[rid]['meta']['mode'], it['W'], it['H'], *Qp[rid]['meta']['sent'], list_key='points', count_key='count')
        try: dec = int(dec)
        except Exception: continue
        A.append(abs(dec - it['n'])); B.append(abs(len(recs) - it['n'])); N.append(it['n'])
    res[f'fsc|{m}'] = dict(mean_objects=round(float(np.mean(N)), 1), **boot(A, B, B))
json.dump(res, open(OUT / 'readout_density.json', 'w'), indent=1)
for k, v in res.items(): print(k, v)
