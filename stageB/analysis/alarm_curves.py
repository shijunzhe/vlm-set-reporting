"""Repair E. Click alarms as empirical risk-coverage curves (draft 2.2, Prop. 5 note; 3.7). ScreenSpot-Pro / OSWorld-G items of
EXP-GUI (260 with C1, C2 and the zoom refinement for every model). No model calls.
Partner set by a fixed rule (no label use): the cheapest other model (Gemini 3.8 for GPT-6.1 and Sonnet 5.5;
Sonnet 5.5 for Gemini 3.8). Agreement: distance <= tau x longer side, tau in a grid. Alarms:
  same      A's C1 vs A's C2 (unzoomed)
  cross     A's C1 vs partner's C1 (unzoomed)
  cross_z   A's zoomed point vs partner's zoomed point (after two-stage zoom)
  same_z    A's zoomed point vs A's second zoomed point (zoom around C2, stage Z2: the same zoom request around the second click)
Per tau: coverage (act share), risk when acting (miss rate among acted), errors caught (share of A's errors alarmed), extra calls.
Matched-coverage comparison: risk of same vs cross interpolated at equal coverage. Output: out/alarm_curves.json"""
import json, math, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/claude/gui/code')
import fsrgui as G
OUT = Path(__file__).parent / 'out'; OUT.mkdir(exist_ok=True)
os.chdir(G.ROOT)
M = ['gpt61', 'sonnet55', 'gem38']; PARTNER = {'gpt61': 'gem38', 'sonnet55': 'gem38', 'gem38': 'sonnet55'}
TAUS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]
C = {c['id']: c for c in json.load(open('data/click_items_prepped.json'))}
def load(stage):
    Q, R = {}, {}
    for l in open(f'runs/main/requests_{stage}.jsonl'): q = json.loads(l); Q[q['request_id']] = q
    for m in M:
        for l in open(f'runs/main/responses_{stage}_{m}.jsonl'):
            r = json.loads(l)
            if r['status'] == 'completed': R[r['request_id']] = r
    return Q, R
QA, RA = load('A'); QB, RB = load('B'); QZ, RZ = load('Z2')
sc = lambda q: None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
def pt(cid, m, s):
    k = f'{cid}|{m}|{s}'; return G.decode_click(RA[k]['raw_text'], sc(QA[k]))[0] if k in RA else 'missing'
def zpt(cid, m):
    p1 = pt(cid, m, 'C1'); k = f'{cid}|{m}|B4'; r4 = RB.get(k)
    if r4 is None or p1 in (None, 'missing'): return p1
    q4 = QB[k]; p, _ = G.decode_click(r4['raw_text'], sc(q4)); w = q4['meta']['window']
    return [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else p1
def zpt2(cid, m):
    p2 = pt(cid, m, 'C2'); k = f'{cid}|{m}|B4c2'; r4 = RZ.get(k)
    if r4 is None or p2 in (None, 'missing'): return p2
    q4 = QZ[k]; p, _ = G.decode_click(r4['raw_text'], sc(q4)); w = q4['meta']['window']
    return [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else p2
items = [cid for cid in C if all(pt(cid, m, s) != 'missing' for m in M for s in ('C1', 'C2'))]
P = {(c, m, s): pt(c, m, s) for c in items for m in M for s in ('C1', 'C2')}; Z = {(c, m): zpt(c, m) for c in items for m in M}; Z2 = {(c, m): zpt2(c, m) for c in items for m in M}
def dist(p, q, c): return math.hypot((p[0] - q[0]) * C[c]['width'], (p[1] - q[1]) * C[c]['height']) / max(C[c]['width'], C[c]['height'])
def curve(act_pt, other_pt):
    out = []
    hit = np.array([G.hit(act_pt(c), C[c]['bbox_norm']) for c in items], float)
    D = np.array([dist(act_pt(c), other_pt(c), c) if act_pt(c) is not None and other_pt(c) is not None else np.inf for c in items])
    for t in TAUS:
        act = D <= t; err = 1 - hit
        out.append(dict(tau=t, coverage=round(float(act.mean()), 4), risk_when_acting=round(float(err[act].mean()) if act.any() else 0.0, 4),
                        errors_caught=round(float((err * ~act).sum() / max(1, err.sum())), 4)))
    return dict(base_accuracy=round(float(hit.mean()), 4), errors=int((1 - hit).sum()), points=out)
def interp_risk(cv, cov):
    pts = sorted((p['coverage'], p['risk_when_acting']) for p in cv['points'])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if cov < xs[0] or cov > xs[-1]: return None
    return float(np.interp(cov, xs, ys))
res = dict(n=len(items), partner_rule=PARTNER, taus=TAUS, models={})
for m in M:
    p = PARTNER[m]
    same = curve(lambda c: P[(c, m, 'C1')], lambda c: P[(c, m, 'C2')])
    cross = curve(lambda c: P[(c, m, 'C1')], lambda c: P[(c, p, 'C1')])
    cross_z = curve(lambda c: Z[(c, m)], lambda c: Z[(c, p)])
    same_z = curve(lambda c: Z[(c, m)], lambda c: Z2[(c, m)])
    matched_z = []
    lo_z = max(min(q['coverage'] for q in same_z['points']), min(q['coverage'] for q in cross_z['points']))
    hi_z = min(max(q['coverage'] for q in same_z['points']), max(q['coverage'] for q in cross_z['points']))
    for cov in (np.linspace(lo_z, hi_z, 3) if hi_z >= lo_z else []):
        rs, rc = interp_risk(same_z, cov), interp_risk(cross_z, cov)
        matched_z.append(dict(coverage=round(float(cov), 4), risk_same=None if rs is None else round(rs, 4), risk_cross=None if rc is None else round(rc, 4)))
    matched = []
    lo = max(min(q['coverage'] for q in same['points']), min(q['coverage'] for q in cross['points']))
    hi = min(max(q['coverage'] for q in same['points']), max(q['coverage'] for q in cross['points']))
    for cov in (np.linspace(lo, hi, 3) if hi >= lo else []):
        rs, rc = interp_risk(same, cov), interp_risk(cross, cov)
        matched.append(dict(coverage=round(float(cov), 4), risk_same=None if rs is None else round(rs, 4), risk_cross=None if rc is None else round(rc, 4)))
    res['models'][m] = dict(partner=p, same=same, cross=cross, cross_after_zoom=cross_z, same_after_zoom=same_z, matched_coverage=matched, matched_coverage_after_zoom=matched_z,
                            extra_calls=dict(same=1, cross=1, cross_after_zoom='1 partner call + 1 partner zoom', same_after_zoom='1 call + 1 zoom'))
    print(m, 'partner', p, 'base', same['base_accuracy'], 'zoom base', cross_z['base_accuracy'])
    for t, a, b, z in zip(TAUS, same['points'], cross['points'], cross_z['points']):
        print(f"  tau {t}: same cov {a['coverage']} risk {a['risk_when_acting']} caught {a['errors_caught']} | cross cov {b['coverage']} risk {b['risk_when_acting']} caught {b['errors_caught']} | zoomed cross cov {z['coverage']} risk {z['risk_when_acting']} caught {z['errors_caught']}")
    print('  matched', matched)
    print('  after zoom: same caught', [q['errors_caught'] for q in same_z['points']], 'cross caught', [q['errors_caught'] for q in cross_z['points']], 'matched', matched_z)
json.dump(res, open(OUT / 'alarm_curves.json', 'w'), indent=1)
