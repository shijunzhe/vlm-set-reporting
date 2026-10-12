"""Paired bootstrap CIs for the cross-model summary (E3 per-image errors from xmodel_e3.json). No model calls.
usage: python -I xmodel_ci.py"""
import json, numpy as np
d = json.load(open('/home/claude/stageB/xmodel/xmodel_e3.json')); rng = np.random.default_rng(1); M = ['gpt61', 'sonnet55', 'gem38']
def ci(E, a, b):
    ids = sorted(E[a]); x = np.array([E[a][i] for i in ids], float); y = np.array([E[b][i] for i in ids], float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(x), len(x)); bs.append((y[s].mean() - x[s].mean()) / x[s].mean())
    return dict(a=a, b=b, va=round(x.mean(), 2), vb=round(y.mean(), 2), rel=round((y.mean() - x.mean()) / x.mean() * 100, 1), ci=[round(np.percentile(bs, 2.5) * 100, 1), round(np.percentile(bs, 97.5) * 100, 1)])
out = {}
for ds, E in d['per_image'].items():
    mean = {k: np.mean(list(v.values())) for k, v in E.items()}
    same = [f'{k}|{m}' for m in M for k in ('single', 'same2_inter', 'same2_union', 'same3_unan', 'same3_maj', 'same3_union')]
    cross = [k for k in E if k.startswith('cross2_')] + ['cross3_unan', 'cross3_maj', 'cross3_union']
    bs_, bc = min(same, key=mean.get), min(cross, key=mean.get)
    avg_unan = {str(i): np.mean([E[f'same3_unan|{m}'][i] for m in M]) for i in E['cross3_unan']}
    E2 = dict(E); E2['avg_same3_unan'] = avg_unan
    out[ds] = dict(best_cross_vs_best_same=ci(E, bs_, bc), cross3_unan_vs_best_same=ci(E, bs_, 'cross3_unan'), cross3_unan_vs_avg_model_same3_unan=ci(E2, 'avg_same3_unan', 'cross3_unan'))
    print(ds, out[ds])
json.dump(out, open('/home/claude/stageB/xmodel/xmodel_ci.json', 'w'), indent=1, default=float)
