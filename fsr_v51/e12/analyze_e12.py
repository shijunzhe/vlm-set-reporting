"""Prespecified analysis for v5-E12 (development/calibration stage for E3; not confirmatory).

Inputs (directory RUNS, default runs_e12): requests.jsonl (the frozen manifest), responses_<model>.jsonl
(request_id, status, raw_text, finish_reason, usage); data_e12/annotations_DEV.jsonl (ground truth, used only
for scoring). Output: RUNS/analysis_<tag>.json.

Rules fixed before any model call
---------------------------------
* Reconciliation. Every manifest request must have a terminal record (completed or failed). A request with no
  record makes the analysis refuse to run, unless allow_incomplete=True, in which case it is scored under the
  failure rule and the output is flagged incomplete. Nothing is skipped silently.
* Failure rule. A request without a completed response is an empty report: every target in its scope is missed.
* Decoding never reads ground truth and never raises. Strict JSON first; if that fails, complete JSON objects are
  recovered by a left-to-right scan (counted as salvaged). Rows whose objects field is not a list, objects that are
  not {x,y}/{u,v} with numeric values in [0,1), contradictory statuses, duplicate and unrequested rows are all
  counted. Every object occurrence keeps its raw occurrence index and character offset.
* Projection. Primary: geometric, Pi_A(P) = records whose physical position lies in A (the theory's projection).
  Secondary: row ownership (records reported in row A), with o (out-of-row records), B and chi_match saved.
* tau = 11 px. delta grid {2,4,6,8} px, primary 6. Residual tolerance 18 px (half the minimum spacing).
  Statistical unit: scene (12). Bootstrap: 4000 resamples of scenes, percentile intervals. Models never pooled.

Primary results
  P1  Delta E = E(direct request, Y_A) - E(Pi_A joint report, Y_A), per frame. Category: direct worse / direct
      better (95% CI excludes 0), equivalent (90% CI inside +-1 error per region), or inconclusive.
  P2  Representation 2x2 on J-local, J-global, A-local, A-global: frame, format, interaction. Equivalence margin
      +-2 errors per image.
  P3  Theorem 1 on (direct, projected joint): actual gap, bound U+Omega, slack; unmatched mass split into
      harmful (P's errors) and helpful (R's errors) parts; the bound is attained when the helpful part is zero.
Secondary
  S1 compatibility excess CE (a signed diagnostic; never used for method choice); S2 row-ownership version with
  o, B, chi; S3 evidence control M-local - D-local; S4 delta rule with a 'no usable bound' outcome;
  S5 mechanism: order adherence, inclusion fraction, flip test of position slopes (spatial / sequential /
  mixed / indeterminate / none), covariance-kernel R^2 as a descriptive diagnostic; S6 failure events and
  thinking/reasoning tokens.
"""
import json, math, os, random, re, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).parent
RUNS = ROOT / os.environ.get('E12_RUNS', 'runs_e12')
TAU, DELTA, GRID, RTOL = 11.0, 6.0, (2.0, 4.0, 6.0, 8.0), 18.0
M_REGION, M_IMAGE = 1.0, 2.0                       # equivalence margins (errors per region / per image)
SLOPE_MIN = 2.0                                    # px per image height: smallest slope treated as practically nonzero
W = H = 1024; CELL = 256
DEC = json.JSONDecoder()


class ReconciliationError(RuntimeError):
    pass


# ---------------------------------------------------------------- decoding (label-free, never raises)
def region_of(x, y):
    if not (0 <= x < W and 0 <= y < H): return None
    return f'R{int(y // CELL) * 4 + int(x // CELL) + 1:03d}'


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def to_xy(obj, frame, rid):
    """Physical coordinates or None."""
    if not isinstance(obj, dict): return None
    if frame == 'local':
        u, v = obj.get('u'), obj.get('v')
        if not (num(u) and num(v) and 0 <= u < 1 and 0 <= v < 1) or not (isinstance(rid, str) and re.fullmatch(r'R0(0[1-9]|1[0-6])', rid)):
            return None
        k = int(rid[1:]) - 1
        return (k % 4) * CELL + CELL * u, (k // 4) * CELL + CELL * v
    x, y = obj.get('x'), obj.get('y')
    if not (num(x) and num(y) and 0 <= x < 1 and 0 <= y < 1): return None
    return W * x, H * y


def scan_objects(text):
    """Complete top-level JSON values found left to right, with their character offsets."""
    out, i = [], 0
    while i < len(text):
        j = min([p for p in (text.find('{', i), text.find('[', i)) if p >= 0], default=-1)
        if j < 0: break
        try:
            v, end = DEC.raw_decode(text, j); out.append((v, j)); i = end
        except ValueError:
            i = j + 1
    return out


def flatten_rows(vals):
    rows = []
    for v, off in vals:
        if isinstance(v, list):
            rows += [(o, off) for o in v]
        else:
            rows.append((v, off))
    return rows


def decode(q, raw):
    """Return (records, audit). record: x, y, row (reported row or None), geo (physical region), occ, off."""
    au = dict(salvaged=0, malformed_rows=0, invalid_objects=0, unrequested_rows=0, missing_rows=0, unknown_rows=0,
              empty_rows=0, found_rows=0, contradictory_rows=0, dup_rows=0, repeats=0, status={}, n_occurrences=0)
    raw = raw if isinstance(raw, str) else ''
    recs, occ = [], 0
    if q['kind'] == 'free':
        items, base = None, 0
        try:
            s = raw.index('{'); v, _ = DEC.raw_decode(raw, s)
            if isinstance(v, dict) and isinstance(v.get('objects'), list): items, base = v['objects'], s
        except ValueError:
            pass
        if items is None:                          # truncated or malformed: recover complete point objects
            if raw.strip(): au['salvaged'] = 1
            body = raw.lstrip()
            items = [v for v, _ in flatten_rows(scan_objects(body[1:] if body.startswith('{') else body)) if isinstance(v, dict) and ('x' in v or 'y' in v)]
        for it in items:
            p = to_xy(it, 'global', None); au['n_occurrences'] += 1
            if p is None: au['invalid_objects'] += 1; occ += 1; continue
            recs.append(dict(x=p[0], y=p[1], row=None, geo=region_of(*p), occ=occ, off=None)); occ += 1
        pts = [(round(r['x'], 3), round(r['y'], 3)) for r in recs]
        au['repeats'] = len(pts) - len(set(pts))
        return recs, au
    strict = None
    t = re.sub(r'^```[a-zA-Z]*\s*', '', raw.strip()); t = re.sub(r'\s*```$', '', t)   # a standalone code fence is not an error
    try:
        if q['format'] == 'array':
            v = json.loads(t[t.index('['): t.rindex(']') + 1]); strict = [(o, 0) for o in v] if isinstance(v, list) else None
        else:
            strict = [(json.loads(l), 0) for l in t.splitlines() if l.strip()]
    except (ValueError, TypeError):
        strict = None
    rows = strict if strict is not None else flatten_rows(scan_objects(raw))
    if strict is None and raw.strip(): au['salvaged'] = 1
    seen = set()
    for o, off in rows:
        if not isinstance(o, dict) or 'region_id' not in o: au['malformed_rows'] += 1; continue
        rid = o.get('region_id')
        if rid not in q['requested']: au['unrequested_rows'] += 1; continue
        if rid in seen: au['dup_rows'] += 1
        seen.add(rid); st = o.get('status'); au['status'][rid] = st
        objs = o.get('objects')
        if not isinstance(objs, list): au['malformed_rows'] += 1; continue
        if st in ('unknown', 'empty', 'found'): au[f'{st}_rows'] += 1
        else: au['malformed_rows'] += 1
        if (st == 'empty' and objs) or (st == 'found' and not objs): au['contradictory_rows'] += 1
        pts = []
        for it in objs:
            p = to_xy(it, q['frame'], rid); au['n_occurrences'] += 1
            if p is None: au['invalid_objects'] += 1; occ += 1; continue
            pts.append((round(p[0], 3), round(p[1], 3)))
            recs.append(dict(x=p[0], y=p[1], row=rid, geo=region_of(*p), occ=occ, off=off)); occ += 1
        au['repeats'] += len(pts) - len(set(pts))
    au['missing_rows'] = len([r for r in q['requested'] if r not in seen])
    return recs, au


# ---------------------------------------------------------------- matching and set errors
def match(a, b, tol):
    if not a or not b: return []
    A = np.array(a, float); B = np.array(b, float)
    ok = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1]) <= tol
    r, c = linear_sum_assignment((~ok).astype(np.int8))
    return [(int(i), int(j)) for i, j in zip(r, c) if ok[i, j]]


def TP(p, y, tol=TAU): return len(match(p, y, tol))
def E(p, y, tol=TAU): return len(p) + len(y) - 2 * TP(p, y, tol)
def U(p, r, d=DELTA): return len(p) + len(r) - 2 * TP(p, r, d)


def unmatched_mass(P, R, Y, d=DELTA):
    """Split U_d(P,R) into harmful mass (P worse: P's unmatched false records + R's unmatched true records) and helpful
    mass (P better: P's unmatched true records + R's unmatched false records), using each report's own tau-matching."""
    m = match(P, R, d); mp = {i for i, _ in m}; mr = {j for _, j in m}
    tp_p = {i for i, _ in match(P, Y, TAU)}; tp_r = {i for i, _ in match(R, Y, TAU)}
    harm = sum(1 for i in range(len(P)) if i not in mp and i not in tp_p) + sum(1 for j in range(len(R)) if j not in mr and j in tp_r)
    help_ = sum(1 for i in range(len(P)) if i not in mp and i in tp_p) + sum(1 for j in range(len(R)) if j not in mr and j not in tp_r)
    return harm, help_


def regional(recs, Yt, owner):
    """Partition accounting for a full-image report: B, chi_match, o, and the count/placement split of B."""
    ys, ps = defaultdict(list), defaultdict(list)
    for t in Yt: ys[region_of(*t)].append(t)
    o = 0
    for r in recs:
        key = r['row'] if owner == 'row' and r['row'] is not None else r['geo']
        o += int(owner == 'row' and r['row'] is not None and r['row'] != r['geo'])
        ps[key].append((r['x'], r['y']))
    B = C = K = tl = 0
    for g in set(ys) | set(ps):
        t = TP(ps[g], ys[g]); tl += t
        c = abs(len(ys[g]) - len(ps[g])); k = min(len(ys[g]), len(ps[g])) - t
        B += c + 2 * k; C += c; K += k
    P = [(r['x'], r['y']) for r in recs]
    chi = TP(P, Yt) - tl
    assert chi >= 0 and B == E(P, Yt) + 2 * chi
    return dict(B=B, count_part=C, placement_part=2 * K, chi_match=chi, o=o)


# ---------------------------------------------------------------- statistics
def boot(vals, B=4000, seed=1, level=.95):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(v) < 2: return None
    rng = random.Random(seed); n = len(v); m = sum(v) / n
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    a = (1 - level) / 2
    return [round(m, 3), round(bs[int(a * B)], 3), round(bs[int((1 - a) * B) - 1], 3), n]


def category(vals, margin):
    c95, c90 = boot(vals), boot(vals, level=.90)
    if c95 is None: return {'estimate': None, 'category': 'no data'}
    if c95[1] > 0: cat = 'positive (first worse)'
    elif c95[2] < 0: cat = 'negative (first better)'
    elif -margin <= c90[1] and c90[2] <= margin: cat = f'equivalent within +-{margin}'
    else: cat = 'inconclusive'
    return {'estimate_ci95': c95, 'ci90': c90, 'margin': margin, 'category': cat}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def spearman(a, b):
    if len(a) < 3: return None
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return None if math.isnan(c) else float(c)


# ---------------------------------------------------------------- analysis
def load(models, allow_incomplete):
    reqs = [json.loads(l) for l in (RUNS / 'requests.jsonl').read_text().splitlines() if l.strip()]
    models = models or sorted({q['model_key'] for q in reqs})
    resp, recon = {}, {}
    for m in models:
        f = RUNS / f'responses_{m}.jsonl'
        R = {}
        if f.exists():
            for l in f.read_text().splitlines():
                if not l.strip(): continue
                x = json.loads(l)
                if x.get('request_id') not in R or x.get('status') == 'completed': R[x.get('request_id')] = x
        planned = [q['request_id'] for q in reqs if q['model_key'] == m]
        missing = [r for r in planned if r not in R]
        extra = [r for r in R if r not in set(planned)]
        recon[m] = {'planned': len(planned), 'completed': sum(1 for r in planned if R.get(r, {}).get('status') == 'completed'),
                    'failed': sum(1 for r in planned if r in R and R[r].get('status') != 'completed'),
                    'missing_response': len(missing), 'unplanned_records': len(extra)}
        if missing and not allow_incomplete:
            raise ReconciliationError(f'{m}: {len(missing)} of {len(planned)} planned requests have no terminal record')
        resp[m] = R
    return reqs, models, resp, recon


def analyze(tag, models=None, allow_incomplete=False):
    reqs, models, resp, recon = load(models, allow_incomplete)
    ann = {a['sample_id']: a for a in map(json.loads, (ROOT / 'data_e12/annotations_DEV.jsonl').read_text().splitlines())}
    out = {'tag': tag, 'tau': TAU, 'delta': DELTA, 'grid': GRID, 'reconciliation': recon,
           'complete': all(r['missing_response'] == 0 for r in recon.values()), 'models': {}}
    for m in models:
        R = resp[m]; D = {}
        for q in reqs:
            if q['model_key'] != m: continue
            r = R.get(q['request_id'])
            ok = r is not None and r.get('status') == 'completed'
            recs, au = decode(q, r.get('raw_text', '') if ok else '')
            au['failed_or_missing'] = int(not ok); au['missing_response'] = int(r is None)
            au['truncated'] = int(ok and (r.get('finish_reason') or '') in ('length', 'max_tokens'))
            u = (r or {}).get('usage') or {}
            au['thinking_tokens'] = u.get('thinking_tokens', u.get('reasoning_tokens'))
            au['output_tokens'] = u.get('output_tokens')
            D[(q['sample_id'], q['condition'], q['focal'], q['rep'])] = (recs, au, q)
        sids = sorted(ann); reps = sorted({k[3] for k in D})
        Y = {s: [tuple(t['point']) for t in ann[s]['targets']] for s in sids}
        YA = lambda s, A: [t for t in Y[s] if region_of(*t) == A]
        P_ = lambda recs: [(r['x'], r['y']) for r in recs]
        res = {}

        # A: full-image error and regional accounting
        full = ['F-down', 'F-up', 'J-local', 'J-global', 'J-global-rev', 'A-local', 'A-global']
        sceneE = {c: {} for c in full}; acc = {c: defaultdict(list) for c in full}
        for c in full:
            for s in sids:
                v = []
                for r in reps:
                    if (s, c, None, r) not in D: continue
                    recs = D[(s, c, None, r)][0]; v.append(E(P_(recs), Y[s]))
                    if c != 'F-down' and c != 'F-up':
                        for k, x in regional(recs, Y[s], 'row').items(): acc[c][k].append(x)
                if v: sceneE[c][s] = mean(v)
        res['A_full_error'] = {c: boot(list(sceneE[c].values())) for c in full}
        res['A_full_error_by_dose'] = {c: {dz: mean([v for s, v in sceneE[c].items() if len(ann[s]['targets']) == dz]) for dz in (96, 192)} for c in full}
        res['A_row_ownership_accounting_mean_per_report'] = {c: {k: round(mean(v), 3) for k, v in acc[c].items()} for c in full if acc[c]}

        # P2: representation 2x2
        cells = ('J-local', 'J-global', 'A-local', 'A-global')
        ok = [s for s in sids if all(s in sceneE[c] for c in cells)]
        g = lambda c, s: sceneE[c][s]
        res['P2_frame_global_minus_local'] = category([(g('J-global', s) + g('A-global', s) - g('J-local', s) - g('A-local', s)) / 2 for s in ok], M_IMAGE)
        res['P2_format_array_minus_ndjson'] = category([(g('A-local', s) + g('A-global', s) - g('J-local', s) - g('J-global', s)) / 2 for s in ok], M_IMAGE)
        res['P2_interaction'] = category([(g('A-global', s) - g('A-local', s)) - (g('J-global', s) - g('J-local', s)) for s in ok], M_IMAGE)
        res['P2_order_Jglobal_rev_minus_nat'] = category([g('J-global-rev', s) - g('J-global', s) for s in sids if s in sceneE['J-global-rev'] and s in sceneE['J-global']], M_IMAGE)
        res['P2_order_F_up_minus_down'] = category([g('F-up', s) - g('F-down', s) for s in sids if s in sceneE['F-up'] and s in sceneE['F-down']], M_IMAGE)

        # P1, P3, S1-S4 on focal regions
        P1, P3, S = {}, {}, {}
        disp_cr, disp_ry = [], []
        for frame in ('local', 'global'):
            dE_geo, dE_row, ce, gap, bnd6, slack6, slackmin, harm, helpm = ([] for _ in range(9))
            for s in sids:
                per = defaultdict(list)
                for A in sorted({k[2] for k in D if k[0] == s and k[1] == f'D-{frame}'}):
                    y = YA(s, A)
                    Dr = {r: P_([z for z in D[(s, f'D-{frame}', A, r)][0] if z['row'] == A]) for r in reps if (s, f'D-{frame}', A, r) in D}
                    Jg = {r: P_([z for z in D[(s, f'J-{frame}', None, r)][0] if z['geo'] == A]) for r in reps if (s, f'J-{frame}', None, r) in D}
                    Jr = {r: P_([z for z in D[(s, f'J-{frame}', None, r)][0] if z['row'] == A]) for r in reps if (s, f'J-{frame}', None, r) in D}
                    com = sorted(set(Dr) & set(Jg))
                    if not com: continue
                    per['dE_geo'].append(mean([E(Dr[r], y) - E(Jg[r], y) for r in com]))
                    per['dE_row'].append(mean([E(Dr[r], y) - E(Jr[r], y) for r in com]))
                    btw = [U(Dr[a], Jg[b]) for a in com for b in com]
                    win_d = [U(Dr[a], Dr[b]) for a, b in combinations(com, 2)]; win_j = [U(Jg[a], Jg[b]) for a, b in combinations(com, 2)]
                    if win_d and win_j: per['ce'].append(mean(btw) - .5 * mean(win_d) - .5 * mean(win_j))
                    for r in com:
                        P, Rf = Dr[r], Jg[r]; e = E(P, y)
                        up = {d: E(Rf, y, TAU - d) + U(P, Rf, d) for d in GRID}
                        lo = {d: E(Rf, y, TAU + d) - U(P, Rf, d) for d in GRID}
                        assert all(lo[d] <= e <= up[d] for d in GRID), 'Theorem 1 violated: implementation error'
                        om = 2 * (TP(Rf, y, TAU + DELTA) - TP(Rf, y, TAU - DELTA))
                        gp = abs(e - E(Rf, y)); hm, hp = unmatched_mass(P, Rf, y)
                        per['gap'].append(gp); per['bnd6'].append(U(P, Rf) + om); per['slack6'].append(U(P, Rf) + om - gp)
                        per['slackmin'].append(min(up.values()) - e); per['harm'].append(hm); per['help'].append(hp)
                        disp_cr += [math.hypot(P[i][0] - Rf[j][0], P[i][1] - Rf[j][1]) for i, j in match(P, Rf, RTOL)]
                        disp_ry += [math.hypot(Rf[i][0] - y[j][0], Rf[i][1] - y[j][1]) for i, j in match(Rf, y, RTOL)]
                for k, lst in (('dE_geo', dE_geo), ('dE_row', dE_row), ('ce', ce), ('gap', gap), ('bnd6', bnd6), ('slack6', slack6),
                               ('slackmin', slackmin), ('harm', harm), ('help', helpm)):
                    if per[k]: lst.append(mean(per[k]))
            P1[frame] = {'dE_direct_minus_geometric_projection': category(dE_geo, M_REGION)}
            P3[frame] = {'actual_gap': boot(gap), 'bound_U_plus_Omega_delta6': boot(bnd6), 'slack_delta6': boot(slack6),
                         'upper_slack_min_over_grid': boot(slackmin), 'harmful_mass': boot(harm), 'helpful_mass': boot(helpm)}
            S[frame] = {'S1_compatibility_excess_signed_diagnostic': boot(ce),
                        'S2_dE_direct_minus_row_projection': category(dE_row, M_REGION)}
        ev = []
        for s in sids:
            v = []
            for A in {k[2] for k in D if k[0] == s and k[1] == 'M-local'}:
                for r in reps:
                    if (s, 'M-local', A, r) in D and (s, 'D-local', A, r) in D:
                        v.append(E(P_([z for z in D[(s, 'M-local', A, r)][0] if z['row'] == A]), YA(s, A))
                                 - E(P_([z for z in D[(s, 'D-local', A, r)][0] if z['row'] == A]), YA(s, A)))
            if v: ev.append(mean(v))
        S['S3_evidence_control_M_minus_D_local'] = category(ev, M_REGION)
        p90 = lambda v: float(np.percentile(v, 90)) if v else None
        dstar = next((d for d in GRID if p90(disp_cr) is not None and d >= p90(disp_cr)), None)
        usable = dstar is not None and p90(disp_ry) is not None and TAU - dstar >= p90(disp_ry)
        S['S4_delta_rule'] = {'p90_candidate_reference_displacement': p90(disp_cr), 'p90_reference_truth_displacement': p90(disp_ry),
                              'delta_star': dstar, 'usable_bound': bool(usable),
                              'outcome': f'delta = {dstar}' if usable else 'no usable bound at tau = 11'}
        res['P1_subset_execution'] = P1; res['P3_theorem1_on_direct_vs_projection'] = P3; res['secondary'] = S

        # S5: mechanism
        mech = {}
        for pair, expect in ((('F-down', 'F-up'), (+1, -1)), (('J-global', 'J-global-rev'), (+1, -1))):
            info = {}
            for c, sign in zip(pair, expect):
                rho, frac_t, rows_by_scene = [], [], defaultdict(list)
                for s in sids:
                    for r in reps:
                        if (s, c, None, r) not in D: continue
                        recs, au, _ = D[(s, c, None, r)]; P = P_(recs); N = max(1, au['n_occurrences'] - 1)
                        mm = match(P, Y[s], RTOL); frac_t.append(len(mm) / max(1, len(Y[s])))
                        if len(mm) >= 3: rho.append(spearman([recs[i]['occ'] for i, _ in mm], [Y[s][j][1] for _, j in mm]))
                        rows_by_scene[s] += [(Y[s][j][1] / H, recs[i]['occ'] / N, P[i][1] - Y[s][j][1]) for i, j in mm]
                rho = [x for x in rho if x is not None]
                sc = sorted(s for s in rows_by_scene if len(rows_by_scene[s]) >= 10)

                def slope(scenes):
                    Z = np.vstack([np.array(rows_by_scene[s]) for s in scenes]); X = np.c_[np.ones(len(Z)), Z[:, 0]]
                    return float(np.linalg.lstsq(X, Z[:, 2], rcond=None)[0][1])
                if len(sc) >= 2:
                    rng = random.Random(11); bs = sorted(slope([sc[rng.randrange(len(sc))] for _ in sc]) for _ in range(1000))
                    a = [round(slope(sc), 2), round(bs[25], 2), round(bs[974], 2)]
                else:
                    a = None
                info[c] = {'order_adherence_fraction': round(sum(1 for x in rho if x * sign > .5) / len(rho), 3) if rho else None,
                           'median_spearman_index_vs_y': round(float(np.median(rho)), 3) if rho else None,
                           'mean_fraction_targets_in_residual_analysis': round(mean(frac_t), 3) if frac_t else None,
                           'position_slope_dy_px_per_image_ci95': a}
            a1, a2 = info[pair[0]]['position_slope_dy_px_per_image_ci95'], info[pair[1]]['position_slope_dy_px_per_image_ci95']
            adh = min(info[c]['order_adherence_fraction'] or 0 for c in pair)
            if a1 is None or a2 is None: verdict = 'no data'
            elif adh < .8: verdict = 'not applicable: order instruction followed in under 80% of reports'
            else:
                nz = lambda a: (a[1] > 0 or a[2] < 0) and abs(a[0]) >= SLOPE_MIN
                if not nz(a1) and not nz(a2): verdict = 'none (no practically nonzero position slope)'
                elif nz(a1) and nz(a2) and np.sign(a1[0]) == np.sign(a2[0]):
                    verdict = 'spatial (same-sign slopes under both orders)' if min(abs(a1[0]), abs(a2[0])) >= .5 * max(abs(a1[0]), abs(a2[0])) else 'mixed (same sign, unequal magnitude)'
                elif nz(a1) and nz(a2): verdict = 'sequential (slope sign flips with order)'
                else: verdict = 'indeterminate'
            info['flip_test_verdict'] = verdict
            mech['/'.join(pair)] = info
        kern = {}
        for c in ('F-down', 'J-global'):
            prod, brown = [], []
            for s in sids:
                M = []
                for r in reps:
                    if (s, c, None, r) not in D: break
                    recs, au, _ = D[(s, c, None, r)]; P = P_(recs); N = max(1, au['n_occurrences'] - 1)
                    M.append({j: (P[i][1] - Y[s][j][1], recs[i]['occ'] / N) for i, j in match(P, Y[s], RTOL)})
                if len(M) < 3: continue
                common = sorted(set.intersection(*[set(x) for x in M]))
                if len(common) < 8: continue
                Em = np.array([[M[r][j][0] for j in common] for r in range(len(M))]); Cv = np.cov(Em.T)
                yv = np.array([Y[s][j][1] / H for j in common]); iv = np.array([mean([M[r][j][1] for r in range(len(M))]) for j in common])
                iu = np.triu_indices(len(common), 1); cv = Cv[iu]
                for X, store in ((np.c_[np.ones(len(cv)), (yv[:, None] * yv[None, :])[iu], (yv[:, None] + yv[None, :])[iu]], prod),
                                 (np.c_[np.ones(len(cv)), np.minimum(iv[:, None], iv[None, :])[iu]], brown)):
                    b = np.linalg.lstsq(X, cv, rcond=None)[0]; res_ = cv - X @ b
                    store.append(1 - res_.var() / cv.var() if cv.var() > 0 else None)
            kern[c] = {'product_kernel_R2': boot(prod), 'brownian_kernel_R2': boot(brown),
                       'note': f'descriptive only; with {len(reps)} repetitions the sample covariance has rank at most {max(0, len(reps) - 1)}'}
        mech['covariance_kernel_descriptive'] = kern
        res['S5_mechanism'] = mech

        # S6: failure events and thinking tokens
        Ev = defaultdict(lambda: defaultdict(int))
        for (s, c, A, r), (recs, au, q) in D.items():
            e = Ev[c]; e['calls'] += 1
            for k in ('failed_or_missing', 'missing_response', 'truncated', 'salvaged', 'malformed_rows', 'invalid_objects', 'missing_rows',
                      'unknown_rows', 'contradictory_rows', 'dup_rows', 'unrequested_rows', 'repeats'):
                e[k] += au.get(k, 0)
            e['calls_with_thinking'] += int((au.get('thinking_tokens') or 0) > 0)
            regs = [A] if A else [f'R{i:03d}' for i in range(1, 17)]
            for g_ in regs:
                k = len(YA(s, g_)); rr = sum(1 for z in recs if (z['row'] if z['row'] else z['geo']) == g_)
                if k >= 4:
                    e['regions_k_ge_4'] += 1; e['catastrophic_underreport'] += int(rr <= k / 2)
                    e['catastrophic_marked_found'] += int(rr <= k / 2 and au['status'].get(g_) == 'found')
        res['S6_failure_events'] = {c: dict(v) for c, v in Ev.items()}
        out['models'][m] = res
    (RUNS / f'analysis_{tag}.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('tag', nargs='?', default='main'); ap.add_argument('--models')
    ap.add_argument('--allow-incomplete', action='store_true')
    a = ap.parse_args()
    o = analyze(a.tag, a.models.split(',') if a.models else None, a.allow_incomplete)
    print(json.dumps({'complete': o['complete'], 'reconciliation': o['reconciliation']}, indent=1))
