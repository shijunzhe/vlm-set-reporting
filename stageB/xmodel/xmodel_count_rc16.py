"""Cross-model fusion for a counting target under the v5.17 readout (EXP-RC16, SCP-dense 50, min). No model calls.
Per image and type: each model's readout (degeneracy-check readout); fused = per-type median of the three models; also mean of
GPT and Sonnet. Per-type count error, paired bootstrap vs each single model.
usage: python -I xmodel_count_rc16.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from analyze_diag import degenerate, T5
SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
GT = {i: collections.Counter(s['type'] for s in json.load(open(f'{C.ROOT}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes']) for i in range(50)}
Q = C.load_requests('rc16', 'main'); RS = C.load_responses('rc16', 'main'); rng = np.random.default_rng(0)
read = collections.defaultdict(dict)
for rid, q in Q.items():
    r = RS.get(rid); m = q['model_key']; i = q['meta']['item']
    recs, dec, _ = C.decode_shapes(r['raw_text'], q['meta']['mode'], 1024, 1024, 1024, 1024)
    recs = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs]
    lc = collections.Counter(x['type'] for x in recs if x['type'] in T5)
    ok = isinstance(dec, dict); w = {t: int(dec.get(t, 0) or 0) for t in T5} if ok else None
    fl = degenerate(recs, sum(w.values()) if ok else None, r, True)
    read[m][i] = w if (fl and ok) else {t: lc[t] for t in T5}
def err(c, i): return sum(abs(c[t] - GT[i][t]) for t in T5)
ids = sorted(set.intersection(*[set(v) for v in read.values()]))
E = {m: [err(read[m][i], i) for i in ids] for m in read}
E['median3'] = [err({t: float(np.median([read[m][i][t] for m in C.MODELS])) for t in T5}, i) for i in ids]
E['mean_gpt_sonnet'] = [err({t: (read['gpt61'][i][t] + read['sonnet55'][i][t]) / 2 for t in T5}, i) for i in ids]
out = {k: round(float(np.mean(v)), 3) for k, v in E.items()}
def b(a, c):
    a = np.array(E[a], float); c = np.array(E[c], float); d = c - a; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return dict(diff=round(float(d.mean()), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)])
out['median3_vs_sonnet'] = b('sonnet55', 'median3'); out['median3_vs_gpt'] = b('gpt61', 'median3'); out['meanGS_vs_sonnet'] = b('sonnet55', 'mean_gpt_sonnet')
print(out); json.dump(out, open('/home/claude/stageB/xmodel/xmodel_count_rc16.json', 'w'), indent=1)
