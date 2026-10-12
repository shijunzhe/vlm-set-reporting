"""Small checks cited in the experiment-G document, saved for traceability. No model calls.
1 Gemini default-thinking coordinate offset on SCP (experiment G, d_enum) vs min.
2 GPT SCP-dense grouping (B1 arm j vs d, i): position misses and type-agnostic error.
3 Gemini SCP-dense wording (B1 arm a one-sided vs b two-sided): untyped share, extra records.
4 Sonnet B7 SCP-dense count-first lists: thinking vs not, short lists, written count vs truth, length flag.
5 Gemini SCP: min vs default on the same 50 images (experiment G, d_enum).
Usage: python -I quick_checks.py"""
import json, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code'); import bcommon as C
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
def gt(task, i): return json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json' if task == 'scp' else f'/home/claude/stageB/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes']
out = {}
Qg = C.load_requests('g1', 'main'); Rg = C.load_responses('g1', 'main')
off = collections.defaultdict(list); E = collections.defaultdict(dict)
for rid, q in Qg.items():
    mt = q['meta']
    if q['model_key'] != 'gem38' or mt['task'] != 'scp' or mt['arm'] != 'd_enum': continue
    recs, _, _ = C.decode_shapes(Rg[rid]['raw_text'], mt['mode'], 1024, 1024, 1024, 1024); G = gt('scp', mt['item'])
    for a, b in C.match_typed(recs, G, 30, typed=False): off[mt['setting']].append((recs[a]['cx'] - G[b]['cx'], recs[a]['cy'] - G[b]['cy']))
    rr = [x for x in recs if x['type'] in T5]; mm = C.match_typed(rr, G, 30); E[mt['setting']][mt['item']] = len(rr) + len(G) - 2 * len(mm)
out['1_gemini_offset'] = {s: dict(n=len(v), mean_dx_dy=[round(float(x), 2) for x in np.mean(v, 0)]) for s, v in off.items()}
ids = sorted(set(E['min']) & set(E['default']))
out['5_gemini_min_vs_default_same_images'] = dict(n=len(ids), min=round(float(np.mean([E['min'][i] for i in ids])), 2), default=round(float(np.mean([E['default'][i] for i in ids])), 2))
Q1 = C.load_requests('b1', 'main'); R1 = C.load_responses('b1', 'main'); st = collections.defaultdict(list)
for rid, q in Q1.items():
    mt = q['meta']
    if mt['task'] != 'scpd': continue
    if not ((q['model_key'] == 'gpt61' and mt['arm'] in ('d', 'j', 'i')) or (q['model_key'] == 'gem38' and mt['arm'] in ('a', 'b'))): continue
    recs, _, _ = C.decode_shapes(R1[rid]['raw_text'], mt['mode'], 1024, 1024, *mt['sent']); G = gt('scpd', mt['item'])
    rr = [x for x in recs if x['type'] in T5]; mm = C.match_typed(rr, G, 20); mag = C.match_typed(recs, G, 20, typed=False)
    st[f"{q['model_key']}|{mt['arm']}"].append(dict(E=len(rr) + len(G) - 2 * len(mm), E_agn=len(recs) + len(G) - 2 * len(mag), pos_miss=len(G) - len(mag),
                                                   extras_agn=len(recs) - len(mag), untyped=float(np.mean([x['type'] not in T5 for x in recs])) if recs else 0.0))
out['2_3_b1_scpd_decomposition'] = {k: {kk: round(float(np.mean([x[kk] for x in v])), 3) for kk in v[0]} | {'n': len(v)} for k, v in st.items()}
Q7 = C.load_requests('b7', 'main'); R7 = C.load_responses('b7', 'main'); rows = []
for rid, q in Q7.items():
    mt = q['meta']
    if q['model_key'] != 'sonnet55' or mt.get('task') != 'scpd' or mt['arm'] != 'a': continue
    r = R7[rid]; recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, *mt['sent']); n = len(gt('scpd', mt['item']))
    w = sum(int(dec.get(t, 0) or 0) for t in T5) if isinstance(dec, dict) else None
    rows.append(dict(think=bool(r['usage'].get('thinking_tokens')), k=len(recs), n=n, w=w, flag=bool(w and (len(recs) > 1.5 * w or len(recs) < 0.5 * w))))
short = [x for x in rows if x['k'] < x['n'] - 10]
out['4_sonnet_b7_countfirst'] = dict(calls=len(rows), thinking_calls=sum(x['think'] for x in rows), short_lists=len(short), short_in_thinking=sum(x['think'] for x in short),
                                     short_shortfall=[x['n'] - x['k'] for x in short], short_written_over_true=[round(x['w'] / x['n'], 2) for x in short if x['w']],
                                     short_flagged_by_length=sum(x['flag'] for x in short),
                                     written_over_true_nonthinking_complete=round(float(np.mean([x['w'] / x['n'] for x in rows if not x['think'] and x['w'] and x['k'] >= x['n'] - 10])), 3))
json.dump(out, open('/home/claude/stageB/offline/quick_checks.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
