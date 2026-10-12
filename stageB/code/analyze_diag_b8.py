"""Label-free readout choice (same rule and thresholds as analyze_diag.py) applied to Qwen3-VL in B8.
SCP: naive = written count of arm a; diagnosed = arm c (list first, count after), code count unless degenerate.
SCP-dense (no arm c in B8): naive = written count of arm a; diagnosed = same response, code count unless degenerate.
Usage: python -I analyze_diag_b8.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
R = C.ROOT; T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']; rng = np.random.default_rng(0); m = 'qwen3vl'
RS = C.load_responses('b8', 'main')
GT = {('scp', i): json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json')) for i in range(100)}
GT.update({('scpd', i): json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json')) for i in range(50)})


def degenerate(recs, written_total, r):
    n = len(recs); flags = []
    if n and sum(1 for x in recs if x['type'] not in T5) / n > 0.1: flags.append('schema')
    if written_total and (n > 1.5 * written_total or n < 0.5 * written_total): flags.append('length')
    seen = set(); rep = 0
    for x in recs:
        k = (round(x['cx']), round(x['cy'])); rep += k in seen; seen.add(k)
    if n and rep / n > 0.1: flags.append('repeat')
    if str(r.get('finish_reason') or r.get('usage', {}).get('stop_reason')) in ('length', 'max_tokens', 'error'): flags.append('trunc')
    return flags


def boot(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / max(1e-9, a[s].mean()))
    return dict(a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / max(1e-9, a.mean())) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(a))


out = {}
for task, darm in (('scp', 'c'), ('scpd', 'a')):
    naive, code, diag, flagged, fl_kinds = [], [], [], 0, collections.Counter()
    for i in range(100 if task == 'scp' else 50):
        ra = RS.get(f'{task}|{i:03d}|{m}|a|r0'); rc = RS.get(f'{task}|{i:03d}|{m}|{darm}|r0')
        if not ra or not rc: continue
        g = collections.Counter(s['type'] for s in GT[(task, i)]['shapes'])
        _, da, _ = C.decode_shapes(ra['raw_text'], 'grid', 1024, 1024, 1024, 1024)
        recs, dc, _ = C.decode_shapes(rc['raw_text'], 'grid', 1024, 1024, 1024, 1024)
        if not isinstance(da, dict) or not isinstance(dc, dict): continue
        w = {t: int(dc.get(t, 0) or 0) for t in T5}; lc = collections.Counter(x['type'] for x in recs if x['type'] in T5)
        fl = degenerate(recs, sum(w.values()), rc); flagged += bool(fl); fl_kinds.update(fl)
        e_code = sum(abs(lc[t] - g[t]) for t in T5); e_w = sum(abs(w[t] - g[t]) for t in T5)
        naive.append(sum(abs(int(da.get(t, 0) or 0) - g[t]) for t in T5)); code.append(e_code); diag.append(e_w if fl else e_code)
    out[f'{task}|{m}'] = dict(n=len(naive), diag_arm=darm, flagged_share=round(flagged / max(1, len(naive)), 3), flag_kinds=dict(fl_kinds),
                              naive_written=round(float(np.mean(naive)), 2), code_only=round(float(np.mean(code)), 2), diagnosed=round(float(np.mean(diag)), 2),
                              diagnosed_vs_naive=boot(naive, diag), diagnosed_vs_code=boot(code, diag))
json.dump(out, open(f'{R}/runs/b8/diag_b8_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
