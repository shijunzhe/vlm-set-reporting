"""E3 repair and split re-analysis (no model calls).

Repair
  * Unit: source (item) clusters; bootstrap over the 124 distinct sources (some sources contribute an L and an R report).
  * Outcome: dE/n (gain in set errors per true object; NOT a share of the original error) and also raw gain dE.
  * Policies: always retry, always change, route by recurrence (change if recurring else retry); differences
    route - always_retry and route - always_change, with source-bootstrap CIs. Policy values are estimated by stratum
    reweighting (recurrence shares in the flagged population), as in analyze_e3.py.
  * Equalised recurrence: recurrence measured with exactly one other call (rep 1) for both L-min and R-min.
  * Cost accounting: number of calls per flagged report: retry/change 1; routing needs one diagnostic call unless a second
    call already exists (here it existed by design; in deployment it is an extra call).
Split (SKU-110K, 'full-image quadrant requests'): the full image is sent in every call and each call reports one quadrant.
  * Error decomposition by location: ground-truth objects and predictions are assigned to the boundary band (centre within
    one median predicted box short side of the quadrant lines) or the interior; FN and FP in each zone, single vs split.
  * Duplicates across quadrants: pairs of predictions from different quadrant calls with IoU >= 0.5 (label-free).
"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
import analyze_e3 as A0
from score_e3 import load_items

ROOT = Path(__file__).parent
items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl'); base = {x['request_id']: x for x in rows}
OUT = {}

# ------------------------------------------------------------------ repair
rq = {json.loads(l)['request_id']: json.loads(l) for l in (ROOT / 'runs/requests_repair.jsonl').read_text().splitlines() if l.strip()}
rep = A0.load_rows('runs_repair/scored.jsonl')
byk = {(x['item'], x['model'], x['cand'], x['rep']): x for x in rows if x['block'] in ('MAIN', 'AGG')}
recs = []
for x in rep:
    q = rq[x['request_id']]; b = base[q['base_id']]; it = items[q['item_id']]; pm = A0.primary(it); post = x[f'E_{pm}']
    if q['action'] == 'CHANGE' and q['arm'] == 'R':
        fl = set(q['requested']); reg = {r['region_id']: r for r in L.registry(it['width'], it['height'])}
        inr = lambda r, g: reg[g]['x0'] <= r['cx'] < reg[g]['x0'] + reg[g]['width'] and reg[g]['y0'] <= r['cy'] < reg[g]['y0'] + reg[g]['height']
        keep = [r for r in A0.recs_of(b) if not any(inr(r, g) for g in fl)]; new = [r for r in A0.recs_of(x) if any(inr(r, g) for g in fl)]
        post = L.error(it, keep + new, A0.metric_of(it, pm))[0] if x['n_pred'] or not x['flags']['empty'] else b[f'E_{pm}']
    elif x['flags']['empty'] or x['flags']['failed']:
        post = b[f'E_{pm}']
    other = byk.get((q['item_id'], q['model_key'], q['cand'], 1))
    recur1 = bool(other and other['flags']['any'])
    recs.append(dict(src=q['item_id'], recur=q['recur'], recur1=recur1, action=q['action'], arm=q['arm'], model=q['model_key'],
                     rel=(b[f'E_{pm}'] - post) / max(1, it['n']), gain=b[f'E_{pm}'] - post, cleared=int(not x['flags']['any']),
                     regions=len(q['requested']) if q['arm'] == 'R' else None))
srcs = sorted({r['src'] for r in recs})

def policy_values(R, key='recur', f='rel'):
    cellv = {}
    for rr in (False, True):
        for a in ('RETRY', 'CHANGE'):
            v = [r[f] for r in R if r[key] == rr and r['action'] == a]; cellv[(rr, a)] = float(np.mean(v)) if v else float('nan')
    sh = {rr: sum(1 for r in R if r[key] == rr) for rr in (False, True)}; tot = sum(sh.values())
    val = lambda pol: sum(sh[rr] / tot * cellv[(rr, pol(rr))] for rr in (False, True))
    route = val(lambda rr: 'CHANGE' if rr else 'RETRY'); retry = val(lambda rr: 'RETRY'); change = val(lambda rr: 'CHANGE')
    return dict(route=route, always_retry=retry, always_change=change, route_minus_retry=route - retry, route_minus_change=route - change), cellv

def boot_policies(key, f, B=2000):
    point, cellv = policy_values(recs, key, f); bs = defaultdict(list); rng = random.Random(f'rep:{key}:{f}')
    grp = defaultdict(list)
    for r in recs: grp[r['src']].append(r)
    for _ in range(B):
        R = [r for s in (rng.choice(srcs) for _ in srcs) for r in grp[s]]
        try:
            v, _ = policy_values(R, key, f)
            if all(math.isfinite(x) for x in v.values()):
                for k, x in v.items(): bs[k].append(x)
        except ZeroDivisionError: pass
    return {k: [round(point[k], 3), round(float(np.percentile(bs[k], 2.5)), 3), round(float(np.percentile(bs[k], 97.5)), 3)] for k in point}, \
           {f'{k[0]}|{k[1]}': round(v, 3) for k, v in cellv.items()}

OUT['repair'] = dict(
    n_reports=len(recs), n_sources=len(srcs),
    cells={f'{rr}|{a}': dict(n=sum(1 for r in recs if r['recur'] == rr and r['action'] == a),
                            cleared=round(float(np.mean([r['cleared'] for r in recs if r['recur'] == rr and r['action'] == a])), 3))
           for rr in (False, True) for a in ('RETRY', 'CHANGE')},
    policies_rel=boot_policies('recur', 'rel')[0], policies_gain=boot_policies('recur', 'gain')[0],
    equalised_recurrence=dict(shares={str(rr): sum(1 for r in recs if r['recur1'] == rr) for rr in (False, True)},
                              policies_rel=boot_policies('recur1', 'rel')[0], cells=boot_policies('recur1', 'rel')[1]),
    agreement_recur_vs_recur1=round(float(np.mean([r['recur'] == r['recur1'] for r in recs])), 3),
    R_change_regions=dict(sorted(((str(k), v) for k, v in __import__('collections').Counter(r['regions'] for r in recs if r['arm'] == 'R' and r['action'] == 'CHANGE').items()))),
    calls_per_flagged_report=dict(always_retry=1, always_change=1, route_with_extra_diagnostic=2, route_with_existing_second_call=1))
print('repair', json.dumps(OUT['repair'], indent=0)[:2500])

# ------------------------------------------------------------------ split
sp = defaultdict(list)
for x in rows:
    if x['block'] == 'SPLIT': sp[(x['model'], x['item'])].append(x)
res = {}
for m in ('gpt61', 'sonnet55', 'gem38'):
    acc = defaultdict(list)
    for (mm, i), parts in sp.items():
        if mm != m or len(parts) != 4: continue
        it = items[i]; W, H = it['width'], it['height']
        single = A0.recs_of(byk[(i, m, 'L-min', 0)])
        allp = [r for p in parts for r in A0.recs_of(p)]
        sides = [min(r['box'][2] - r['box'][0], r['box'][3] - r['box'][1]) for r in single + allp if r['box']]
        band = float(np.median(sides)) if sides else 20.0
        def zone(cx, cy): return 'boundary' if (abs(cx - W / 2) < band or abs(cy - H / 2) < band) else 'interior'
        for name, P in (('single', single), ('split', allp)):
            m_ = L.match_boxes(P, [(t['box'], t['label']) for t in it['targets']], .75); mp = {a for a, _ in m_}; mt = {b for _, b in m_}
            for z in ('boundary', 'interior'):
                fn = sum(1 for j, t in enumerate(it['targets']) if j not in mt and zone((t['box'][0] + t['box'][2]) / 2, (t['box'][1] + t['box'][3]) / 2) == z)
                fp = sum(1 for j, r in enumerate(P) if j not in mp and zone(r['cx'], r['cy']) == z)
                acc[f'{name}|{z}|FN'].append(fn); acc[f'{name}|{z}|FP'].append(fp)
            acc[f'{name}|n_pred'].append(len(P))
        nb = sum(1 for t in it['targets'] if zone((t['box'][0] + t['box'][2]) / 2, (t['box'][1] + t['box'][3]) / 2) == 'boundary'); acc['n_boundary'].append(nb); acc['n'].append(it['n'])
        dup = 0
        for a in range(4):
            for b in range(a + 1, 4):
                Pa, Pb = A0.recs_of(parts[a]), A0.recs_of(parts[b])
                if Pa and Pb:
                    M = L.iou_matrix([r['box'] for r in Pa], [r['box'] for r in Pb]); dup += int((M >= .5).any(axis=1).sum())
        acc['cross_quadrant_duplicates'].append(dup)
        # predictions whose centre lies outside the requested quadrant
        outq = 0
        for p in parts:
            q = p['quadrant']; x0, y0, x1, y1 = q[0] * W, q[1] * H, q[2] * W, q[3] * H
            outq += sum(1 for r in A0.recs_of(p) if not (x0 <= r['cx'] < x1 and y0 <= r['cy'] < y1))
        acc['outside_requested_quadrant'].append(outq)
    res[m] = {k: round(float(np.mean(v)), 2) for k, v in acc.items()}
    print('split', m, res[m])
OUT['split'] = res
(ROOT / 'runs/repair_split_v2.json').write_text(json.dumps(OUT, indent=1))
