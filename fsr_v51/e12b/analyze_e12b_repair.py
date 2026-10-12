"""E12b-C analysis (prespecified with build_e12b_repair.py, before any repair call).

For every base report: E_strict (frozen E12 decoder), E_step0 (lenient readout, 0 calls), E_final (after the at most one
repair call; fallback to step 0 when the repair call fails or returns nothing readable). Reported per source arm and
model: mean E at each stage, extra calls, repair cost, paired bootstrap over scenes (repetitions averaged).

C-P1 (prediction): the gain of the repair call is carried by count error, not placement error, in the flagged scope.
  For every repaired report: gain = E_step0 - E_final; cnt / plc = count / placement part of the step-0 report inside
  the flagged regions (rows) or the whole image (lists). Fit gain = a*cnt + b*plc without intercept; bootstrap over
  reports (4000). Supported when the 95% CI of a is above 0 and the point estimate of b is below a.
  D4 of the v5.2 plan is triggered when C-P1 is not supported.
H3 preview: final E of rows+repair against list+follow-up at the same cap (E12b A2 vs A3; E12 J-global vs F-down).
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent; E12 = ROOT.parent / 'e12'
sys.path.insert(0, str(E12)); sys.path.insert(0, str(ROOT))
import analyze_e12 as A
import build_e12b_repair as BR

PRICE = {'gpt56': (4, 20), 'gpt61': (2, 10), 'opus55': (4, 20), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}
RUNS = ROOT / 'runs_e12b'


def truth():
    Y = {}
    for f in (ROOT / 'data_e12b/annotations_DEV.jsonl', E12 / 'data_e12/annotations_DEV.jsonl'):
        for l in f.read_text().splitlines():
            if l.strip(): a = json.loads(l); Y[a['sample_id']] = [tuple(t['point']) for t in a['targets']]
    return Y


def split(P, Y, scope=None):
    ys, ps = defaultdict(list), defaultdict(list)
    for t in Y: ys[A.region_of(*t)].append(t)
    for p in P: ps[A.region_of(*p)].append(p)
    c = k = 0
    for g in set(ys) | set(ps):
        if scope is not None and g not in scope: continue
        t = A.TP(ps[g], ys[g]); c += abs(len(ys[g]) - len(ps[g])); k += min(len(ys[g]), len(ps[g])) - t
    return c, 2 * k


def main():
    Y = truth()
    bases = [json.loads(l) for l in (RUNS / 'repair_bases.jsonl').read_text().splitlines() if l.strip()]
    reqs = {json.loads(l)['request_id']: json.loads(l) for l in (RUNS / 'repair_requests.jsonl').read_text().splitlines() if l.strip()}
    resp = {}
    for f in (ROOT / 'runs_e12b_repair').glob('responses_*.jsonl'):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r['status'] == 'completed' or r['request_id'] not in resp: resp[r['request_id']] = r
    missing = [k for k in reqs if k not in resp]
    if missing: raise RuntimeError(f'{len(missing)} repair requests without a terminal record')
    # strict E of the base (frozen decoder)
    man = {}
    for f in (RUNS / 'requests.jsonl', E12 / 'runs_e12/requests.jsonl'):
        for l in f.read_text().splitlines():
            if l.strip(): q = json.loads(l); man[q['request_id']] = q
    raws = {}
    for f in list(RUNS.glob('responses_*.jsonl')) + list((E12 / 'runs_e12').glob('responses_*.jsonl')):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r['status'] == 'completed' or r['request_id'] not in raws: raws[r['request_id']] = r
    rows = []
    for b in bases:
        y = Y[b['sample_id']]; q = man[b['base_id']]; rb = raws[b['base_id']]
        strict = [(z['x'], z['y']) for z in A.decode(q, rb.get('raw_text', '') if rb['status'] == 'completed' else '')[0]]
        if b['kind'] == 'rows':
            s0 = {k: v['pts'] for k, v in b['step0'].items()}
        else:
            s0 = {'*': b['step0']['pts']}
        P0 = [p for v in s0.values() for p in v]
        final, call, cost, fallback = P0, 0, 0.0, False
        if b['flagged']:
            rr = resp['E12bC:' + b['base_id']]; call = 1; u = rr.get('usage') or {}
            pr = PRICE[b['model_key']]; cost = (u.get('input_tokens', 0) * pr[0] + u.get('output_tokens', 0) * pr[1]) / 1e6
            raw = rr.get('raw_text', '') if rr['status'] == 'completed' else ''
            if b['kind'] == 'rows':
                new = BR.lenient_rows(raw, b['flagged'])
                if new:
                    s1 = dict(s0)
                    for g in b['flagged']: s1[g] = new[g]['pts'] if g in new else s0.get(g, [])
                    final = [p for v in s1.values() for p in v]
                else: fallback = True
            else:
                new = BR.lenient_list(raw)
                if raw.strip() and (new['pts'] or new['status'] in ('found', 'empty')): final = new['pts']
                else: fallback = True
        scope = set(b['flagged']) if (b['kind'] == 'rows' and b['flagged']) else None
        cnt, plc = split(P0, y, scope)
        rows.append(dict(b=b, E_strict=A.E(strict, y), E0=A.E(P0, y), E1=A.E(final, y), call=call, cost=cost, fallback=fallback,
                         cnt=cnt if b['flagged'] else 0, plc=plc if b['flagged'] else 0))
    out = {'groups': {}}
    grp = defaultdict(list)
    for r in rows: grp[(r['b']['source'], r['b']['src_arm'], r['b']['model_key'])].append(r)
    for (src, arm, m), L in sorted(grp.items()):
        def scene_mean(k):
            d = defaultdict(list)
            for r in L: d[r['b']['sample_id']].append(r[k])
            return {s: float(np.mean(v)) for s, v in d.items()}
        e0, e1, es = scene_mean('E0'), scene_mean('E1'), scene_mean('E_strict')
        out['groups'][f'{src}|{arm}|{m}'] = dict(
            reports=len(L), flagged=sum(1 for r in L if r['b']['flagged']), fallbacks=sum(r['fallback'] for r in L),
            E_strict=round(float(np.mean(list(es.values()))), 3), E_step0=round(float(np.mean(list(e0.values()))), 3),
            E_final=round(float(np.mean(list(e1.values()))), 3),
            step0_minus_strict=A.boot([e0[s] - es[s] for s in e0]), final_minus_step0=A.boot([e1[s] - e0[s] for s in e0]),
            repair_cost_usd=round(sum(r['cost'] for r in L), 3), extra_calls_per_report=round(sum(r['call'] for r in L) / len(L), 3))
    # C-P1
    R = [r for r in rows if r['call']]
    def fit(S):
        X = np.array([[r['cnt'], r['plc']] for r in S], float); g = np.array([r['E0'] - r['E1'] for r in S], float)
        if len(S) < 3 or not X.any(): return None
        return np.linalg.lstsq(X, g, rcond=None)[0]
    est = fit(R); rng = random.Random(3); bs = []
    for _ in range(4000):
        f = fit([R[rng.randrange(len(R))] for _ in range(len(R))])
        if f is not None: bs.append(f)
    bs = np.array(bs) if bs else None
    cp1 = None
    if est is not None and bs is not None:
        lo, hi = np.percentile(bs[:, 0], [2.5, 97.5]); blo, bhi = np.percentile(bs[:, 1], [2.5, 97.5])
        sup = bool(lo > 0 and est[1] < est[0])
        cp1 = dict(a_count=[round(float(est[0]), 3), round(float(lo), 3), round(float(hi), 3)],
                   b_placement=[round(float(est[1]), 3), round(float(blo), 3), round(float(bhi), 3)], n_reports=len(R),
                   verdict='supported' if sup else 'not supported')
    out['C_P1'] = cp1
    out['per_model_gain'] = {m: dict(repaired=len(L), mean_gain=round(float(np.mean([r['E0'] - r['E1'] for r in L])), 3),
                                     mean_cnt=round(float(np.mean([r['cnt'] for r in L])), 3), mean_plc=round(float(np.mean([r['plc'] for r in L])), 3))
                             for m in PRICE for L in [[r for r in R if r['b']['model_key'] == m]] if L}
    out['D4 C-P1 fails -> drop repair routing from S-route'] = dict(triggered=(cp1 is None or cp1['verdict'] != 'supported'))
    # H3 preview
    h3 = {}
    for m in PRICE:
        for src, a1, a2 in (('E12b', 'A2', 'A3'), ('E12', 'J-global', 'F-down')):
            d1 = {}; d2 = {}
            for r in rows:
                if r['b']['model_key'] != m or r['b']['source'] != src: continue
                tgt = d1 if r['b']['src_arm'] == a1 else d2 if r['b']['src_arm'] == a2 else None
                if tgt is not None: tgt.setdefault(r['b']['sample_id'], []).append(r['E1'])
            com = sorted(set(d1) & set(d2))
            if com: h3[f'{src}|{m}|{a1}+repair - {a2}+followup'] = A.boot([np.mean(d1[s]) - np.mean(d2[s]) for s in com])
    out['H3_preview'] = h3
    (RUNS / 'analysis_e12b_repair.json').write_text(json.dumps(out, indent=1))
    for k, v in out['groups'].items(): print(k, v)
    print('C-P1', cp1); print('per model', out['per_model_gain']); print('H3', h3)


if __name__ == '__main__':
    main()
