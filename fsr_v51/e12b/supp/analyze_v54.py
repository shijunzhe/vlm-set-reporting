"""v5.4 analysis (written before the supplementary calls; E12b main data are already known, so every analysis on them
is a post-freeze revision and is labelled as such).

Changes implemented here:
 * floor screening uses an ABSOLUTE per-image threshold A fixed by design information (E12b: A = 0.05 x mean design
   count = 7.2 errors per image), never the true count of the screened image;
 * when every candidate is rejected the procedure returns NONE (no fallback to the least bad candidate);
 * rejection (label-free), floor ranking (heuristic) and confirmation (labels) are separate rules, evaluated separately;
 * reference-anchored rejection: R_a >= mean U_2t(P_a, P_ref) - u_ref, u_ref from the labelled sources;
 * the 200 random 12/12 splits of the same 24 scenes are a sensitivity analysis, not independent data sets;
 * C-P1 and repair contrasts use a cluster bootstrap over source scenes;
 * repair is compared with two baselines on the same reports, with every branch observed:
     API     retry the identical request only on failed / empty / unparseable answers (strict decoding),
     STRONG  the prespecified observable triggers (unknown, missing rows, truncation, empty, contract violation) with
             lenient readout and an identical whole-request retry,
     OURS    the prespecified triggers with the targeted re-query (rows) or follow-up with the previous answer (lists);
   API -> STRONG isolates the triggers, STRONG -> OURS isolates the repair action;
 * label-free quantities (absolute floor, flags) are separated from the post hoc audit that needs labels;
 * supplementary blocks: CAP16 (default reasoning at the 16,384 cap) vs Adef (32,768); B1v2 (registry-consistent
   examples) vs B1 (inconsistent examples) vs 4x4.
Output: runs_e12b_supp/analysis_v54.json
"""
import json, os, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

os.environ['E12B_RUNS'] = 'runs_e12b'
ROOT = Path(__file__).resolve().parent.parent; E12 = ROOT.parent / 'e12'
for p in (ROOT, E12, Path(__file__).parent): sys.path.insert(0, str(p))
import analyze_e12 as A
import analyze_e12b as Z
import build_e12b_repair as BR
TAU = A.TAU; A_ABS = 7.2
PRICE = Z.PRICE
SUPP = ROOT / os.environ.get('E12B_SUPP', 'runs_e12b_supp')


def boot(v, B=4000, seed=1):
    return A.boot(v, B=B, seed=seed)


def cluster_boot(clusters, stat, B=4000, seed=2):
    """clusters: list of lists of items; stat: function(list of items) -> float. Percentile 95% interval."""
    rng = random.Random(seed); est = stat([x for c in clusters for x in c]); bs = []
    for _ in range(B):
        s = [x for _ in clusters for x in clusters[rng.randrange(len(clusters))]]
        try: bs.append(stat(s))
        except (ZeroDivisionError, np.linalg.LinAlgError, ValueError): pass
    bs = np.array(bs)
    return [round(float(np.ravel(est)[0]) if np.ndim(est) else float(est), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(clusters)]


rows, _ = Z.load()
out = {'A_abs_per_image': A_ABS}

# ------------------------------------------------------------------ (1) absolute floor screening, NONE outcome
cand = defaultdict(lambda: defaultdict(dict))
for r in rows:
    q = r['q']
    if q['block'] == 'A': cand[(q['model_key'], q['arm'], q['grid'])][q['sample_id']][q['rep']] = r
scr = {}
for m in Z.MODELS:
    res = {}
    for arm in ('A1', 'A2', 'A3'):
        for g in ('on', 'off'):
            by = cand[(m, arm, g)]
            fl = [A.E(d[0]['P'], d[1]['P'], 2 * TAU) / 2 for d in by.values()]
            ac = [(d[0]['E'] + d[1]['E']) / 2 for d in by.values()]
            res[f'{arm}-{g}'] = dict(abs_floor=round(float(np.mean(fl)), 2), mean_E_posthoc=round(float(np.mean(ac)), 2),
                                    rejected=bool(np.mean(fl) > A_ABS), meets_requirement_posthoc=bool(np.mean(ac) <= A_ABS))
    res['NONE'] = all(v['rejected'] for v in res.values())
    scr[m] = res
out['1_absolute_floor_screening'] = scr

# ------------------------------------------------------------------ (2) selection sensitivity analysis
SEL = {}
for m in Z.MODELS:
    keys = [(m, arm, g) for arm in ('A1', 'A2', 'A3') for g in ('on', 'off')]
    scenes = sorted(cand[keys[0]]); cache = {}
    def st(key, S, what, ref=None):
        v = []
        for s in S:
            ck = (key, s, what, ref)
            if ck not in cache:
                d = cand[key][s]
                if what == 'E': cache[ck] = (d[0]['E'] + d[1]['E']) / 2
                elif what == 'floor': cache[ck] = A.E(d[0]['P'], d[1]['P'], 2 * TAU) / 2
                elif what == 'flags': cache[ck] = (d[0]['flags']['any'] + d[1]['flags']['any']) / 2
                elif what == 'U_ref': cache[ck] = A.E(d[0]['P'], cand[ref][s][0]['P'], 2 * TAU)
            v.append(cache[ck])
        return float(np.mean(v))
    rng = random.Random(7); rec = defaultdict(list)
    def evaluate(name, pick, tE, best):
        if pick is None:
            rec[name].append(dict(none=True, correct_none=best > A_ABS))
        else:
            e = float(np.mean([tE[k] for k in pick]))                       # ties averaged
            rec[name].append(dict(none=False, regret=e - best, meets=e <= A_ABS))
    def argmins(vals):
        mn = min(vals.values()); return [k for k, v in vals.items() if abs(v - mn) < 1e-12]
    for _ in range(200):
        S = scenes[:]; rng.shuffle(S); cal, test = S[:12], S[12:]; lab5 = cal[:5]
        tE = {k: st(k, test, 'E') for k in keys}; best = min(tE.values())
        surv = [k for k in keys if st(k, cal, 'floor') <= A_ABS]
        evaluate('FIXED (rows+grid, E12 recipe)', [(m, 'A2', 'on')], tE, best)
        evaluate('FLAGS (fewest observable flags, 0 labels)', argmins({k: st(k, cal, 'flags') for k in keys}), tE, best)
        evaluate('FLOOR-RANK (heuristic, 0 labels)', argmins({k: st(k, cal, 'floor') for k in keys}), tE, best)
        evaluate('LAB5', argmins({k: st(k, lab5, 'E') for k in keys}), tE, best)
        evaluate('LAB12', argmins({k: st(k, cal, 'E') for k in keys}), tE, best)
        evaluate('REJ -> LAB5', argmins({k: st(k, lab5, 'E') for k in surv}) if surv else None, tE, best)
        evaluate('REJ -> LAB12', argmins({k: st(k, cal, 'E') for k in surv}) if surv else None, tE, best)
        if surv:
            ref = min(surv, key=lambda k: (st(k, lab5, 'E'), keys.index(k))); u_ref = st(ref, lab5, 'E')
            surv2 = [k for k in surv if k == ref or st(k, cal, 'U_ref', ref) - u_ref <= A_ABS]
            evaluate('REJ -> ANCHOR -> LAB5', argmins({k: st(k, lab5, 'E') for k in surv2}), tE, best)
        else:
            evaluate('REJ -> ANCHOR -> LAB5', None, tE, best)
    summ = {}
    for name, L in rec.items():
        sel = [x for x in L if not x['none']]; non = [x for x in L if x['none']]
        summ[name] = dict(none_rate=round(len(non) / len(L), 3),
                          correct_none_rate=(round(sum(x['correct_none'] for x in non) / len(non), 3) if non else None),
                          mean_regret_when_selecting=(round(float(np.mean([x['regret'] for x in sel])), 2) if sel else None),
                          p_regret_above_A=(round(float(np.mean([x['regret'] > A_ABS for x in sel])), 3) if sel else None),
                          p_selected_meets_requirement=(round(float(np.mean([x['meets'] for x in sel])), 3) if sel else None))
    SEL[m] = summ
out['2_selection_sensitivity_200_splits_same_24_scenes'] = SEL

# ------------------------------------------------------------------ (3) label-free certification vs post hoc audit
cert = {}
for m in Z.MODELS:
    tot = fl_err = cert_unflag = 0.0; floor_abs = []
    for key, by in cand.items():
        if key[0] != m: continue
        for s, d in by.items():
            a, b = d[0], d[1]; U = A.E(a['P'], b['P'], 2 * TAU); floor_abs.append(U / 2)
            tot += a['E'] + b['E']
            if a['flags']['any'] or b['flags']['any']: fl_err += a['E'] * a['flags']['any'] + b['E'] * b['flags']['any']
            else: cert_unflag += min(U, a['E'] + b['E'])
    cert[m] = dict(label_free_mean_abs_floor_per_report=round(float(np.mean(floor_abs)), 2),
                   label_free_flag_rate=round(float(np.mean([r['flags']['any'] for r in rows if r['q']['model_key'] == m and r['q']['block'] == 'A'])), 3),
                   posthoc_share_of_true_error_in_flagged_reports=round(fl_err / tot, 3) if tot else None,
                   posthoc_share_certified_by_floor_in_unflagged_pairs=round(cert_unflag / tot, 3) if tot else None)
out['3_label_free_vs_posthoc_audit'] = cert

# ------------------------------------------------------------------ (4) repair: API vs STRONG vs OURS
bases = [json.loads(l) for l in (ROOT / 'runs_e12b/repair_bases.jsonl').read_text().splitlines() if l.strip()]
def load_resp(d):
    o = {}
    for f in d.glob('responses_*.jsonl'):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r['status'] == 'completed' or r['request_id'] not in o: o[r['request_id']] = r
    return o
rep = load_resp(ROOT / 'runs_e12b_repair'); sup = load_resp(SUPP)
truth = {}
for f in (ROOT / 'data_e12b/annotations_DEV.jsonl', E12 / 'data_e12/annotations_DEV.jsonl'):
    for l in f.read_text().splitlines():
        if l.strip(): a_ = json.loads(l); truth[a_['sample_id']] = [tuple(t['point']) for t in a_['targets']]
man, raws = {}, {}
for f in (ROOT / 'runs_e12b/requests.jsonl', E12 / 'runs_e12/requests.jsonl'):
    for l in f.read_text().splitlines():
        if l.strip(): q = json.loads(l); man[q['request_id']] = q
for f in list((ROOT / 'runs_e12b').glob('responses_*.jsonl')) + list((E12 / 'runs_e12').glob('responses_*.jsonl')):
    for l in f.read_text().splitlines():
        if l.strip(): r = json.loads(l); raws[r['request_id']] = r
def cost(r, m):
    u = (r or {}).get('usage') or {}; return (u.get('input_tokens', 0) * PRICE[m][0] + u.get('output_tokens', 0) * PRICE[m][1]) / 1e6
def lenient_pts(kind, raw, requested):
    if kind == 'rows': return [p for v in BR.lenient_rows(raw, requested).values() for p in v['pts']]
    return BR.lenient_list(raw)['pts']
items, missing = [], []
for b in bases:
    y = truth[b['sample_id']]; q = man[b['base_id']]; rb = raws[b['base_id']]; raw = rb.get('raw_text', '') if rb['status'] == 'completed' else ''
    m = b['model_key']
    recs, au = A.decode(q, raw); strict = [(z['x'], z['y']) for z in recs]
    P0 = lenient_pts(b['kind'], raw, b['requested'])
    # OURS
    ours, c_ours = P0, 0.0
    if b['flagged']:
        rr = rep['E12bC:' + b['base_id']]; rraw = rr.get('raw_text', '') if rr['status'] == 'completed' else ''; c_ours = cost(rr, m)
        if b['kind'] == 'rows':
            new = BR.lenient_rows(rraw, b['flagged'])
            if new:
                s1 = {k: v['pts'] for k, v in b['step0'].items()}
                for g in b['flagged']: s1[g] = new[g]['pts'] if g in new else s1.get(g, [])
                ours = [p for v in s1.values() for p in v]
        else:
            new = BR.lenient_list(rraw)
            if rraw.strip() and (new['pts'] or new['status'] in ('found', 'empty')): ours = new['pts']
    api_trig = (not raw.strip()) or au['salvaged'] > 0
    retry = sup.get('E12bS:RETRY:' + b['base_id'])
    if (api_trig or b['flagged']) and retry is None: missing.append(b['base_id'])
    rraw = (retry or {}).get('raw_text', '') if (retry or {}).get('status') == 'completed' else ''
    # API baseline: strict decode, identical retry on API triggers, strict decode of the retry (keep base if retry empty)
    api, c_api = strict, 0.0
    if api_trig:
        c_api = cost(retry, m)
        if rraw.strip(): api = [(z['x'], z['y']) for z in A.decode(q, rraw)[0]]
    # STRONG baseline: lenient readout, identical whole retry on the prespecified triggers
    strong, c_strong = P0, 0.0
    if b['flagged']:
        c_strong = cost(retry, m)
        if rraw.strip():
            pts = lenient_pts(b['kind'], rraw, b['requested'])
            if pts or b['kind'] == 'list': strong = pts
    scope = set(b['flagged']) if (b['kind'] == 'rows' and b['flagged']) else None
    ys, ps = defaultdict(list), defaultdict(list)
    for t_ in y: ys[A.region_of(*t_)].append(t_)
    for p_ in P0: ps[A.region_of(*p_)].append(p_)
    cnt = plc = 0
    for g in set(ys) | set(ps):
        if scope is not None and g not in scope: continue
        tp = A.TP(ps[g], ys[g]); cnt += abs(len(ys[g]) - len(ps[g])); plc += 2 * (min(len(ys[g]), len(ps[g])) - tp)
    items.append(dict(src=b['source'], arm=b['src_arm'], m=m, sid=b['sample_id'], flagged=bool(b['flagged']), api_trig=api_trig,
                      E_strict=A.E(strict, y), E0=A.E(P0, y), E_api=A.E(api, y), E_strong=A.E(strong, y), E_ours=A.E(ours, y),
                      c_api=c_api, c_strong=c_strong, c_ours=c_ours, cnt=cnt, plc=plc,
                      scope=len(b['flagged']) if (b['kind'] == 'rows' and b['flagged']) else (16 if b['flagged'] else 0)))
out['4_missing_retry_branches'] = missing
grp = defaultdict(list)
for it in items: grp[(it['src'], it['arm'], it['m'])].append(it)
R4 = {}
for k, L in sorted(grp.items()):
    if not any(x['flagged'] or x['api_trig'] for x in L): continue
    R4['|'.join(k)] = dict(reports=len(L), api_retries=sum(x['api_trig'] for x in L), triggered=sum(x['flagged'] for x in L),
                           **{f'mean_{e}': round(float(np.mean([x[e] for x in L])), 3) for e in ('E_strict', 'E0', 'E_api', 'E_strong', 'E_ours')},
                           cost_api=round(sum(x['c_api'] for x in L), 3), cost_strong=round(sum(x['c_strong'] for x in L), 3), cost_ours=round(sum(x['c_ours'] for x in L), 3))
out['4_repair_by_source'] = R4
cl = defaultdict(list)
for it in items: cl[(it['src'], it['sid'])].append(it)
clusters = list(cl.values())
def diff(a, b): return lambda S: float(np.mean([x[a] - x[b] for x in S]))
out['4_repair_contrasts_cluster_bootstrap'] = {
    'STRONG - API (trigger effect), errors per report': cluster_boot(clusters, diff('E_strong', 'E_api')),
    'OURS - STRONG (action effect), errors per report': cluster_boot(clusters, diff('E_ours', 'E_strong')),
    'OURS - API, errors per report': cluster_boot(clusters, diff('E_ours', 'E_api'))}
trig = [it for it in items if it['flagged']]
tcl = defaultdict(list)
for it in trig: tcl[(it['src'], it['sid'])].append(it)
def coef(i):
    def f(S):
        X = np.array([[x['cnt'], x['plc']] for x in S], float); g = np.array([x['E0'] - x['E_ours'] for x in S], float)
        return float(np.linalg.lstsq(X, g, rcond=None)[0][i])
    return f
out['4_C_P1_cluster_bootstrap'] = dict(n_reports=len(trig), n_source_scenes=len(tcl),
                                       count_coef=cluster_boot(list(tcl.values()), coef(0)), placement_coef=cluster_boot(list(tcl.values()), coef(1)),
                                       scope_sizes=sorted(x['scope'] for x in trig))

# ------------------------------------------------------------------ (5) matched cap and (6) B1v2
sreq = {json.loads(l)['request_id']: json.loads(l) for l in (SUPP / 'requests.jsonl').read_text().splitlines() if l.strip()}
ann = {json.loads(l)['sample_id']: [tuple(t['point']) for t in json.loads(l)['targets']] for l in (ROOT / 'data_e12b/annotations_DEV.jsonl').read_text().splitlines() if l.strip()}
srows = []
for rid, q in sreq.items():
    if q['block'] == 'RETRY': continue
    r = sup.get(rid); raw = r.get('raw_text', '') if r and r['status'] == 'completed' else ''
    P = [(z['x'], z['y']) for z in A.decode(q, raw)[0]]; Y = ann[q['sample_id']]
    u = (r or {}).get('usage') or {}
    srows.append(dict(q=q, E=A.E(P, Y), out_tok=u.get('output_tokens') or 0, trunc=str((r or {}).get('finish_reason')) in Z.TRUNC, done=r is not None))
out['5_6_supp_missing'] = sum(not x['done'] for x in srows)
cap = {}
for m in ('gpt56', 'gpt61'):
    for arm in ('A2', 'A3'):
        a = {x['q']['sample_id']: x for x in srows if x['q']['block'] == 'CAP16' and x['q']['model_key'] == m and x['q']['arm'] == arm}
        b = {r['q']['sample_id']: r for r in rows if r['q']['block'] == 'Adef' and r['q']['model_key'] == m and r['q']['arm'] == arm}
        com = sorted(set(a) & set(b))
        cap[f'{m}|{arm}'] = dict(E_cap16=round(float(np.mean([a[s]['E'] for s in com])), 3), E_cap32=round(float(np.mean([b[s]['E'] for s in com])), 3),
                                 cap16_minus_cap32=boot([a[s]['E'] - b[s]['E'] for s in com]), truncated_at_cap16=sum(a[s]['trunc'] for s in com),
                                 max_output_tokens_cap16=max(a[s]['out_tok'] for s in com), max_output_tokens_cap32=max(b[s]['out_tok'] for s in com))
out['5_matched_cap'] = cap
b1 = {}
b1ids = sorted({r['q']['sample_id'] for r in rows if r['q']['block'] == 'B1'})
for m in Z.MODELS:
    def sm(src, k):
        d = defaultdict(list)
        if src == 'k4':
            for r in rows:
                q = r['q']
                if q['block'] == 'A' and q['model_key'] == m and q['arm'] == 'A2' and q['grid'] == 'off' and q['sample_id'] in b1ids: d[q['sample_id']].append(r['E'])
        elif src == 'B1':
            for r in rows:
                q = r['q']
                if q['block'] == 'B1' and q['model_key'] == m and q['granularity'] == k: d[q['sample_id']].append(r['E'])
        else:
            for x in srows:
                q = x['q']
                if q['block'] == 'B1v2' and q['model_key'] == m and q['granularity'] == k: d[q['sample_id']].append(x['E'])
        return {s: float(np.mean(v)) for s, v in d.items()}
    k4 = sm('k4', 4); res = {'k4': round(float(np.mean(list(k4.values()))), 3)}
    for k in (2, 8):
        v2, v1 = sm('B1v2', k), sm('B1', k)
        res[f'k{k}_v2'] = round(float(np.mean(list(v2.values()))), 3) if v2 else None
        res[f'k{k}_original'] = round(float(np.mean(list(v1.values()))), 3)
        if v2:
            res[f'k{k}_v2 - k4'] = boot([v2[s] - k4[s] for s in b1ids]); res[f'k{k}_v2 - k{k}_original'] = boot([v2[s] - v1[s] for s in b1ids])
    b1[m] = res
out['6_B1v2'] = b1

(SUPP / 'analysis_v54.json').write_text(json.dumps(out, indent=1, default=str))
if __name__ == '__main__':
    for k in ('1_absolute_floor_screening',):
        for m, v in out[k].items(): print(k, m, 'NONE' if v['NONE'] else '', {a: (x['abs_floor'], x['rejected']) for a, x in v.items() if a != 'NONE'})
    for m, v in SEL.items():
        print('SEL', m)
        for n_, s in v.items(): print('   ', n_, s)
    print('certification', cert)
    print('missing retry branches', missing)
    for k, v in R4.items(): print('repair', k, v)
    print(out['4_repair_contrasts_cluster_bootstrap']); print(out['4_C_P1_cluster_bootstrap'])
    print('cap', cap); print('B1', b1); print('supp missing', out['5_6_supp_missing'])
