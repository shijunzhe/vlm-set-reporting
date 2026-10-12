"""E3 analysis: does acting on the FSR model change decisions for the better, compared with a reasonable engineering
procedure that uses the same candidates, calls and labels?

Units: dataset x model cells (4 x 3). Primary metric: points (FSC-147) with tolerance tau_point; boxes with IoU >= 0.75
(report-to-report threshold for the floor IoU >= 0.5). Secondary: IoU >= 0.5 and box centres.
Dataset-level constants (from the dataset's mean count, no per-image truth): equivalence margin M = max(1, 0.05 mean n);
requirement A = 0.2 mean n per image (sensitivity 0.1 and 0.3).

A  interface effects (rep 0): mean error per candidate; R - L and default - min, paired bootstrap over sources
B  selection (200 random splits per cell): screening set S (10 sources from the screening pool, 2 calls, no labels),
   labelled set (10 other sources), evaluation set (the rest). Rules:
     ENG      lowest labelled mean error among the 4 candidates
     FSR      (1) reject candidates whose mean floor on S exceeds A; NONE if all are rejected
              (2) label-free equivalence: mean_S U_delta(a,b) + band <= M (band from the labelled sources of the reference)
              (3) among survivors choose the lowest labelled mean; replace it by a cheaper survivor when the two are certified
                  equivalent or |labelled difference| + 2 SE < M; if only one survivor, or all survivors are mutually
                  certified, no labels are needed (cheapest survivor)
     ablations FSR without rejection, FSR without equivalence
   Outcomes: NONE rate and correctness (best evaluation mean > A), regret, requirement met, cost of the chosen interface,
   labels used.
C  aggregation (L-min, 3 calls on every source): single, majority, union, and the rule (union iff nested >= 0.9 and
   max - median count > M)
D  repair 2 x 2 (recurrence x action): gain per repaired report; value of FSR routing (retry if not recurring, else change)
   vs always retry
E  load: completion-flag and catastrophe rates by count tertile; SKU split (union of 4 quadrant calls) vs single call
F  certificate validity and theorem checks
"""
import itertools, json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
from score_e3 import load_items, PRICE, TAG

ROOT = Path(__file__).parent
CANDS = ['L-min', 'R-min', 'L-default', 'R-default']
import os
MODELS = os.environ.get('E3_MODELS', 'gpt61,sonnet55,gem38').split(','); DATASETS = ['fsc147', 'sku110k', 'dota', 'doclaynet']
COMP = ('empty', 'truncated', 'unknown', 'missing_rows', 'failed')


def boot(v, B=2000, seed=1):
    v = [x for x in v if x is not None]
    if len(v) < 2: return None
    rng = random.Random(seed); n = len(v); bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(sum(v) / n, 3), round(bs[int(.025 * B)], 3), round(bs[int(.975 * B) - 1], 3), n]


def recs_of(row):
    return [dict(cx=r[0], cy=r[1], label=r[2], box=r[3]) for r in row['recs']]


def metric_of(item, name):
    return 'point' if (item['task'] == 'point' or name == 'point') else ('box', .75 if name == 'iou75' else .5)


def primary(item): return 'point' if item['task'] == 'point' else 'iou75'


def load_rows(path):
    return [json.loads(l) for l in (ROOT / path).read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ aggregation helpers
def cluster_match(item, a, b):
    return L.match_points(L._pts(a, item['labels']), L._pts(b, item['labels']), item['tau_point'])


def avg_rec(rs):
    out = dict(cx=float(np.mean([r['cx'] for r in rs])), cy=float(np.mean([r['cy'] for r in rs])), label=rs[0]['label'], box=None)
    if rs[0]['box'] is not None: out['box'] = [float(np.mean([r['box'][k] for r in rs])) for k in range(4)]
    return out


def majority(item, reps):
    a, b, c = reps; m12 = dict(cluster_match(item, a, b)); m13 = dict(cluster_match(item, a, c)); m23 = dict(cluster_match(item, b, c))
    out = []; used_b, used_c = set(m12.values()), set(m13.values())
    for i, p in enumerate(a):
        grp = [p] + ([b[m12[i]]] if i in m12 else []) + ([c[m13[i]]] if i in m13 else [])
        if len(grp) >= 2: out.append(avg_rec(grp))
    for j, q in enumerate(b):
        if j in used_b: continue
        if j in m23 and m23[j] not in used_c: out.append(avg_rec([q, c[m23[j]]]))
    return out


def union(item, reps):
    base = list(reps[0])
    for P in reps[1:]:
        mt = {j for _, j in cluster_match(item, base, P)}; base += [p for j, p in enumerate(P) if j not in mt]
    return base


def nestedness(item, reps):
    v = []
    for i, j in itertools.combinations(range(len(reps)), 2):
        a, b = sorted([reps[i], reps[j]], key=len)
        if a: v.append(len(cluster_match(item, a, b)) / len(a))
    return float(np.mean(v)) if v else 1.0


def main():
    items = load_items(); rows = load_rows(f'runs_main/scored{TAG}.jsonl')
    by = defaultdict(lambda: defaultdict(dict))          # (ds, model) -> (cand, item, rep) -> row ; plus SPLIT
    split = defaultdict(lambda: defaultdict(list))
    for x in rows:
        if x['block'] in ('MAIN', 'AGG'): by[(x['dataset'], x['model'])][(x['cand'], x['item'], x['rep'])] = x
        elif x['block'] == 'SPLIT': split[(x['dataset'], x['model'])][x['item']].append(x)
    pool = json.loads((ROOT / 'runs/screening_pool.json').read_text())
    out = {'constants': {}}
    R = {k: {} for k in 'ABCDEF'}
    for ds in DATASETS:
        its = [i for i in items.values() if i['dataset'] == ds]; mean_n = float(np.mean([i['n'] for i in its]))
        M = max(1.0, 0.05 * mean_n); A = 0.2 * mean_n; out['constants'][ds] = dict(mean_n=round(mean_n, 1), M=round(M, 2), A=round(A, 1))
        ids = sorted(i['id'] for i in its); scr = sorted(pool[ds])
        for m in MODELS:
            cell = by[(ds, m)]; key = f'{ds}|{m}'
            if not cell: continue
            pm = primary(its[0])
            def E(c, i, r=0, name=None):
                x = cell.get((c, i, r)); return None if x is None else x[f'E_{name or pm}']
            def cost(c, i, r=0):
                x = cell.get((c, i, r)); return 0.0 if x is None else (x['in_tok'] * PRICE[m][0] + x['out_tok'] * PRICE[m][1]) / 1e6
            # ---------- A
            a = {c: round(float(np.mean([E(c, i) for i in ids])), 2) for c in CANDS}
            sec = {c: {nm: round(float(np.mean([E(c, i, 0, nm) for i in ids])), 2) for nm in (('point', 'iou50') if its[0]['task'] == 'box' else ('point',))} for c in CANDS}
            con = {}
            for c1, c2 in (('R-min', 'L-min'), ('R-default', 'L-default'), ('L-default', 'L-min'), ('R-default', 'R-min')):
                d = [E(c1, i) - E(c2, i) for i in ids]; b = boot(d)
                cat = 'first worse' if b[1] > 0 else 'first better' if b[2] < 0 else ('equivalent' if -M <= b[1] and b[2] <= M else 'inconclusive')
                con[f'{c1} - {c2}'] = dict(ci=b, category=cat)
            R['A'][key] = dict(mean_E=a, secondary=sec, contrasts=con, cost_per_call={c: round(float(np.mean([cost(c, i) for i in ids])), 4) for c in CANDS})
            # ---------- precompute floors, cross disagreements, bands
            it = {i: items[i] for i in ids}
            recs = {(c, i, r): recs_of(cell[(c, i, r)]) for (c, i, r) in cell}
            def selection(pmx, A):
                TJ = .75 if pmx == 'iou75' else .5
                def floor(c, i):
                    if (c, i, 1) not in recs: return None
                    return L.disagreement(it[i], recs[(c, i, 0)], recs[(c, i, 1)], metric_of(it[i], pmx)) / 2
                fl = {(c, i): floor(c, i) for c in CANDS for i in scr}
                # theorem check (without ignore-region removal)
                viol = tot = 0
                for c in CANDS:
                    for i in scr:
                        if (c, i, 1) not in recs: continue
                        i2 = dict(it[i], ignore=[]); mt = metric_of(i2, pmx)
                        tot += 1; viol += int(2 * fl[(c, i)] > L.error(i2, recs[(c, i, 0)], mt)[0] + L.error(i2, recs[(c, i, 1)], mt)[0] + 1e-9)
                # delta disagreement between candidates (rep 0) and bands
                if pmx == 'point':
                    tau = None
                    def Ud(i, c1, c2): return L.disagreement(dict(it[i], tau_point=it[i]['tau_point'] / 4), recs[(c1, i, 0)], recs[(c2, i, 0)], 'point')   # 2*(tau/4) = tau/2 = delta
                    def band(c, i):
                        lo = L.error(dict(it[i], tau_point=it[i]['tau_point'] * .5), recs[(c, i, 0)], 'point')[0]; hi = L.error(dict(it[i], tau_point=it[i]['tau_point'] * 1.5), recs[(c, i, 0)], 'point')[0]
                        return lo - hi
                else:
                    def Ud(i, c1, c2):
                        m_ = L.match_boxes(recs[(c1, i, 0)], [(r['box'], r['label']) for r in recs[(c2, i, 0)]], 1 - (1 - TJ) / 2)
                        return len(recs[(c1, i, 0)]) + len(recs[(c2, i, 0)]) - 2 * len(m_)
                    def band(c, i):
                        return L.error(it[i], recs[(c, i, 0)], ('box', 1 - (1 - TJ) / 2))[0] - L.error(it[i], recs[(c, i, 0)], ('box', 1 - 1.5 * (1 - TJ)))[0]
                UD = {(i, c1, c2): Ud(i, c1, c2) for i in ids for c1, c2 in itertools.combinations(CANDS, 2)}
                BD = {(c, i): band(c, i) for c in CANDS for i in ids}
                # ---------- B selection
                rng = random.Random(int.from_bytes(key.encode(), 'little') % (2 ** 32)); res = defaultdict(list); cert_checks = []
                for _ in range(200):
                    S = rng.sample(scr, 10); rest = [i for i in ids if i not in S]; lab = rng.sample(rest, 10); ev = [i for i in rest if i not in lab]
                    tE = {c: float(np.mean([E(c, i, 0, pmx) for i in ev])) for c in CANDS}; best = min(tE.values())
                    lE = {c: [E(c, i, 0, pmx) for i in lab] for c in CANDS}; ccost = {c: float(np.mean([cost(c, i) for i in S])) for c in CANDS}
                    eng_pick = min(CANDS, key=lambda c: (np.mean(lE[c]), ccost[c]))
                    def record(name, pick, labels_used):
                        if pick is None: res[name].append(dict(none=True, correct=best > A, labels=labels_used)); return
                        res[name].append(dict(none=False, regret=tE[pick] - best, meets=tE[pick] <= A, cost=ccost[pick], labels=labels_used,
                                              cost_ratio=ccost[pick] / max(1e-9, min(ccost.values())), d_eng=tE[pick] - tE[eng_pick]))
                    def confirm(pick, labels_used):
                        # confirmation needs labels: NONE iff the labelled mean of the chosen candidate exceeds A by more than 2 SE
                        if pick is None: return None, labels_used
                        v = np.array(lE[pick], float); se = v.std(ddof=1) / math.sqrt(len(v))
                        return (None if v.mean() - 2 * se > A else pick), 10
                    # ENG
                    record('ENG', eng_pick, 10)
                    record('ENG+confirm', *confirm(eng_pick, 10))
                    def fsr(reject=True, equiv=True):
                        F = {c: float(np.mean([fl[(c, i)] for i in S])) for c in CANDS}
                        surv = [c for c in CANDS if (F[c] <= A or not reject)]
                        if not surv: return None, 0
                        def certified(c1, c2):
                            k = (c1, c2) if CANDS.index(c1) < CANDS.index(c2) else (c2, c1)
                            D = float(np.mean([UD[(i, k[0], k[1])] for i in S])); b = max(float(np.mean([BD[(c1, i)] for i in lab])), float(np.mean([BD[(c2, i)] for i in lab])))
                            return D + max(0.0, b) <= M
                        if len(surv) == 1: return surv[0], 0
                        if equiv and all(certified(c1, c2) for c1, c2 in itertools.combinations(surv, 2)):
                            return min(surv, key=lambda c: ccost[c]), 0
                        bestc = min(surv, key=lambda c: (np.mean(lE[c]), ccost[c]))
                        for c in sorted(surv, key=lambda c: ccost[c]):
                            if ccost[c] >= ccost[bestc]: break
                            d = np.array(lE[c]) - np.array(lE[bestc]); se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0
                            if (equiv and certified(c, bestc)) or abs(d.mean()) + 2 * se < M: return c, 10
                        return bestc, 10
                    for name, kw in (('FSR', {}), ('FSR-noReject', dict(reject=False)), ('FSR-noEquiv', dict(equiv=False))):
                        p, nl = fsr(**kw); record(name, p, nl)
                        if name == 'FSR': record('FSR+confirm', *confirm(p, nl))
                    for c1, c2 in itertools.combinations(CANDS, 2):
                        D = float(np.mean([UD[(i, c1, c2)] for i in S])); b = max(float(np.mean([BD[(c1, i)] for i in lab])), float(np.mean([BD[(c2, i)] for i in lab])))
                        if D + max(0, b) <= M: cert_checks.append(abs(tE[c1] - tE[c2]) <= M)
                summ = {}
                for name, lst in res.items():
                    sel = [x for x in lst if not x['none']]; non = [x for x in lst if x['none']]
                    summ[name] = dict(none_rate=round(len(non) / len(lst), 3), none_correct=(round(float(np.mean([x['correct'] for x in non])), 3) if non else None),
                                      regret=(round(float(np.mean([x['regret'] for x in sel])), 2) if sel else None),
                                      meets=(round(float(np.mean([x['meets'] for x in sel])), 3) if sel else None),
                                      cost_ratio_vs_cheapest=(round(float(np.mean([x['cost_ratio'] for x in sel])), 3) if sel else None),
                                      labels=round(float(np.mean([x['labels'] for x in lst])), 2),
                                      d_eng_paired=(round(float(np.mean([x['d_eng'] for x in sel])), 2) if sel else None))
                orac = {c: float(np.mean([E(c, i, 0, pmx) for i in ids])) for c in CANDS}
                return dict(rules=summ, best_overall=min(orac, key=orac.get), any_meets_A=min(orac.values()) <= A,
                                   floors={c: round(float(np.mean([fl[(c, i)] for i in scr])), 2) for c in CANDS},
                                   certificates=dict(n=len(cert_checks), valid_rate=(round(float(np.mean(cert_checks)), 3) if cert_checks else None)),
                                   theorem=dict(pairs=tot, violations=viol))
            R['B'][key] = {'primary_A0.2': selection(pm, A), 'primary_A0.1': selection(pm, 0.1 * mean_n), 'primary_A0.3': selection(pm, 0.3 * mean_n)}
            if its[0]['task'] == 'box':
                R['B'][key]['point_A0.2'] = selection('point', A); R['B'][key]['iou50_A0.2'] = selection('iou50', A)
            # ---------- C aggregation (L-min, 3 calls)
            agg = defaultdict(list)
            for i in ids:
                if not all((('L-min', i, r) in recs) for r in (0, 1, 2)): continue
                reps = [recs[('L-min', i, r)] for r in (0, 1, 2)]; mt = metric_of(it[i], pm)
                cnt = sorted(len(P) for P in reps); uu = (cnt[2] - cnt[1]) > M and nestedness(it[i], reps) >= 0.9
                e_s = float(np.mean([L.error(it[i], P, mt)[0] for P in reps])); e_m = L.error(it[i], majority(it[i], reps), mt)[0]; e_u = L.error(it[i], union(it[i], reps), mt)[0]
                agg['single'].append(e_s); agg['majority'].append(e_m); agg['union'].append(e_u); agg['rule'].append(e_u if uu else e_m); agg['picked_union'].append(int(uu))
            R['C'][key] = dict(n=len(agg['single']), **{k: round(float(np.mean(v)), 2) for k, v in agg.items()},
                               gain_rule=boot([s - r for s, r in zip(agg['single'], agg['rule'])]), gain_majority=boot([s - r for s, r in zip(agg['single'], agg['majority'])]))
            # ---------- E load and split
            ld = {}
            for k in range(3):
                sub = [i for i in ids if it[i]['stratum'] == k]
                fr = [any(cell[(c, i, 0)]['flags'][f] for f in COMP) for c in ('L-min', 'R-min') for i in sub]
                ca = [E(c, i) > .5 * it[i]['n'] and cell[(c, i, 0)]['n_pred'] < .5 * it[i]['n'] for c in ('L-min', 'R-min') for i in sub]
                ld[f'tertile{k}'] = dict(mean_n=round(float(np.mean([it[i]['n'] for i in sub])), 1), completion_flag_rate=round(float(np.mean(fr)), 3), short_report_rate=round(float(np.mean(ca)), 3))
            sp = None
            if ds == 'sku110k' and split[(ds, m)]:
                d1, d2 = [], []
                for i in ids:
                    parts = split[(ds, m)].get(i, [])
                    if len(parts) != 4: continue
                    u = []
                    for x in parts:
                        q = x['quadrant']; W, H = it[i]['width'], it[i]['height']
                        u += [r for r in recs_of(x) if q[0] * W <= r['cx'] < q[2] * W and q[1] * H <= r['cy'] < q[3] * H]
                    mt = metric_of(it[i], pm); d1.append(L.error(it[i], u, mt)[0]); d2.append(E('L-min', i))
                sp = dict(n=len(d1), split_E=round(float(np.mean(d1)), 2), single_E=round(float(np.mean(d2)), 2), split_minus_single=boot([a_ - b_ for a_, b_ in zip(d1, d2)]))
            R['E'][key] = dict(load=ld, split=sp)
    # ---------- D repair
    rep_path = ROOT / f'runs_repair/scored{TAG}.jsonl'
    if rep_path.exists():
        reprow = load_rows(f'runs_repair/scored{TAG}.jsonl'); rq = {json.loads(l)['request_id']: json.loads(l) for l in (ROOT / 'runs/requests_repair.jsonl').read_text().splitlines() if l.strip()}
        base = {x['request_id']: x for x in rows}
        D = defaultdict(list)
        for x in reprow:
            q = rq[x['request_id']]; b = base[q['base_id']]; it = items[q['item_id']]; pm = primary(it)
            post = x[f'E_{pm}']
            if q['action'] == 'CHANGE' and q['arm'] == 'R':
                fl_reg = set(q['requested']); reg = {r['region_id']: r for r in L.registry(it['width'], it['height'])}
                def inreg(r, g): return reg[g]['x0'] <= r['cx'] < reg[g]['x0'] + reg[g]['width'] and reg[g]['y0'] <= r['cy'] < reg[g]['y0'] + reg[g]['height']
                keep = [r for r in recs_of(b) if not any(inreg(r, g) for g in fl_reg)]
                new = [r for r in recs_of(x) if any(inreg(r, g) for g in fl_reg)]
                post = L.error(it, keep + new, metric_of(it, pm))[0] if x['n_pred'] or not x['flags']['empty'] else b[f'E_{pm}']
            elif x['flags']['empty'] or x['flags']['failed']:
                post = b[f'E_{pm}']                                   # unreadable repair: keep the base report
            D[(q['recur'], q['action'])].append(dict(gain=b[f'E_{pm}'] - post, rel=(b[f'E_{pm}'] - post) / max(1, it['n']), ds=q['dataset'], m=q['model_key']))
        tab = {f'recur={k[0]}|{k[1]}': dict(n=len(v), mean_gain=round(float(np.mean([y['gain'] for y in v])), 2), mean_rel_gain=round(float(np.mean([y['rel'] for y in v])), 3),
                                            ci=boot([y['rel'] for y in v])) for k, v in sorted(D.items())}
        share = {r: (len(D[(r, 'RETRY')]) + len(D[(r, 'CHANGE')])) for r in (False, True)}; tot = sum(share.values()) or 1
        def val(policy):
            return sum(share[r] / tot * (np.mean([y['rel'] for y in D[(r, policy(r))]]) if D[(r, policy(r))] else 0) for r in (False, True))
        R['D'] = dict(table=tab, policy_value_rel_gain=dict(ENG_always_retry=round(float(val(lambda r: 'RETRY')), 3),
                                                            FSR_route=round(float(val(lambda r: 'CHANGE' if r else 'RETRY')), 3),
                                                            always_change=round(float(val(lambda r: 'CHANGE')), 3)), shares=share)
    out.update(R)
    (ROOT / f'runs/analysis_e3{TAG}.json').write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == '__main__':
    o = main()
    print(json.dumps(o['constants']))
    for k, v in o['A'].items(): print('A', k, v['mean_E'], {c: x['category'] for c, x in v['contrasts'].items()})
    for k, vv in o['B'].items():
        for var, v in vv.items(): print('B', k, var, 'floors', v['floors'], 'cert', v['certificates'], 'thm', v['theorem']); [print('    ', n, s) for n, s in v['rules'].items()]
    for k, v in o['C'].items(): print('C', k, v)
    for k, v in o['E'].items(): print('E', k, v)
    print('D', json.dumps(o.get('D'), default=str))
