"""Label-free readout choice (FSR step 1 diagnosis): one call that returns both a list and a written count; use the code count
of the list unless the list is degenerate, in which case use the written count. A list is degenerate if any of:
  schema decay: > 10% of records lack the required fields (type for SCP),
  length mismatch: list length > 1.5x or < 0.5x the written count,
  repetition: > 10% of records repeat an earlier coordinate (to 1 px),
  truncation: the call ended on the output limit.
Thresholds fixed before looking at accuracy. Compared with the naive written count and with code-count-only.
Usage: python -I analyze_diag.py"""
import json, os, sys, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
R = C.ROOT; T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']; rng = np.random.default_rng(0)
Q1 = C.load_requests('b1', 'main'); R1 = C.load_responses('b1', 'main'); Qp = C.load_requests('b1p', 'main'); Rp = C.load_responses('b1p', 'main')
items = json.load(open(f'{R}/runs/b1p/items.json'))
GT = {('scp', i): json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json')) for i in range(100)}
GT.update({('scpd', i): json.load(open(f'{R}/data/scpd_gen/ground_truth/gt_{i:03d}.json')) for i in range(50)})


def degenerate(recs, written_total, r, typed):
    n = len(recs); flags = []
    if typed and n and sum(1 for x in recs if x['type'] not in T5) / n > 0.1: flags.append('schema')
    if written_total and (n > 1.5 * written_total or n < 0.5 * written_total): flags.append('length')
    seen = set(); rep = 0
    for x in recs:
        k = (round(x['cx']), round(x['cy']))
        rep += k in seen; seen.add(k)
    if n and rep / n > 0.1: flags.append('repeat')
    if str(r.get('finish_reason')) in ('length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS'): flags.append('trunc')
    return flags


def boot(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / max(1e-9, a[s].mean()))
    return dict(a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float((b.mean() - a.mean()) / max(1e-9, a.mean())) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(a))


out = {}
for task in ('scp', 'scpd'):
    for m in C.MODELS:
        naive, code, diag, flagged = [], [], [], 0
        for i in range(100 if task == 'scp' else 50):
            ra = R1.get(f'{task}|{i:03d}|{m}|a|r0'); rc = R1.get(f'{task}|{i:03d}|{m}|c|r0')
            if not ra or not rc: continue
            g = collections.Counter(s['type'] for s in GT[(task, i)]['shapes'])
            _, da, _ = C.decode_shapes(ra['raw_text'], C.MODE[m], 1024, 1024, 1024, 1024)
            recs, dc, _ = C.decode_shapes(rc['raw_text'], C.MODE[m], 1024, 1024, 1024, 1024)
            if not isinstance(da, dict) or not isinstance(dc, dict): continue
            w = {t: int(dc.get(t, 0) or 0) for t in T5}; lc = collections.Counter(x['type'] for x in recs if x['type'] in T5)
            fl = degenerate(recs, sum(w.values()), rc, True); flagged += bool(fl)
            e_code = sum(abs(lc[t] - g[t]) for t in T5); e_w = sum(abs(w[t] - g[t]) for t in T5)
            naive.append(sum(abs(int(da.get(t, 0) or 0) - g[t]) for t in T5)); code.append(e_code); diag.append(e_w if fl else e_code)
        if naive:
            out[f'{task}|{m}'] = dict(n=len(naive), flagged_share=round(flagged / len(naive), 3), naive_written=round(float(np.mean(naive)), 2),
                                      code_only=round(float(np.mean(code)), 2), diagnosed=round(float(np.mean(diag)), 2),
                                      diagnosed_vs_naive=boot(naive, diag), diagnosed_vs_code=boot(code, diag))
for m in C.MODELS:
    naive, code, diag, flagged = [], [], [], 0
    for iid, it in items.items():
        rid = f'fsc|{iid}|{m}|cf|c0'; r = Rp.get(rid)
        if not r or r['status'] != 'completed': continue
        recs, dec, _ = C.decode_shapes(r['raw_text'], Qp[rid]['meta']['mode'], it['W'], it['H'], *Qp[rid]['meta']['sent'], list_key='points', count_key='count')
        try: dec = int(dec)
        except Exception: continue
        fl = degenerate(recs, dec, r, False); flagged += bool(fl)
        naive.append(abs(dec - it['n'])); code.append(abs(len(recs) - it['n'])); diag.append(abs(dec - it['n']) if fl else abs(len(recs) - it['n']))
    out[f'fsc|{m}'] = dict(n=len(naive), flagged_share=round(flagged / max(1, len(naive)), 3), naive_written=round(float(np.mean(naive)), 2),
                           code_only=round(float(np.mean(code)), 2), diagnosed=round(float(np.mean(diag)), 2),
                           diagnosed_vs_naive=boot(naive, diag), diagnosed_vs_code=boot(code, diag))
json.dump(out, open(f'{R}/runs/b6/diag_analysis.json', 'w'), indent=1)
for k, v in out.items(): print(k, v['n'], 'flagged', v['flagged_share'], 'naive', v['naive_written'], 'code', v['code_only'], 'diag', v['diagnosed'], 'vs naive', v['diagnosed_vs_naive']['rel'], v['diagnosed_vs_naive']['ci'], 'vs code', v['diagnosed_vs_code']['rel'], v['diagnosed_vs_code']['ci'])
