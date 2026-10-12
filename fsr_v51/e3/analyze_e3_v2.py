"""E3 decision analysis v2. No model calls.

Changes from analyze_e3.py (which is kept unchanged):
  * One loss everywhere: the ignore-aware task loss E_I used for scoring. The floor is computed for the same loss:
        E_I(P,Y) + E_I(Q,Y) >= U_2tau(P_out, Q_out) - |P_in| - |Q_in|
    where P_in are the predictions whose centre lies in an ignore region (proof in FSR v5.7, Prop. 1'). F_I = max(0, .)/2.
  * Population-level statements use a statistical term. A candidate is rejected without labels only if the lower
    confidence bound of its screening floor exceeds A:  mean_S F_I - 2 SE_S(F_I) > A. (The plain mean rule of E3 is kept
    as 'FSR3-mean' for comparison.)
  * Three-state output: MEETS (a candidate whose labelled upper bound mean + 2SE <= A; the cheapest such candidate),
    FAILS (every candidate rejected, by the floor or by labelled lower bound mean - 2SE > A), INSUFFICIENT (otherwise).
  * Label accounting: any step that reads labels counts them (no zero-label equivalence branch).
  * Full grid: requirement A in {0.05,...,1.0} x mean n, every metric (FSC: points; boxes: IoU 0.75, IoU 0.5, centres).
  * Descriptive full-sample table: which candidates meet A on the 60-source mean, per metric and A.

Rules compared (200 random splits per cell: S = 10 screening sources with 2 calls, 10 labelled sources, 40 evaluation sources):
  ENG-ship   label 10, take the lowest labelled mean, ship (MEETS) iff that mean <= A, else FAILS
  ENG-CI     label-only three-state rule (no floor)
  FSR3       floor LCB rejection (0 labels), then the label three-state rule on the survivors
  FSR3-mean  as FSR3 but rejection by the plain screening mean (the E3 rule)
Truth for scoring a split: candidate c meets iff its mean loss over the 40 evaluation sources is <= A.
Usage: [E3_TAG=_agn|_corr] [E3_SCREEN=20|30] python analyze_e3_v2.py
"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
import analyze_e3 as A0
from score_e3 import load_items, PRICE, TAG

ROOT = Path(__file__).parent
CANDS = A0.CANDS
A_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
SPLITS = 200
import os
NS = int(os.environ.get('E3_SCREEN', '10'))      # screening sources (unlabelled, 2 calls each); 10 = the E3 setting


def split_ignore(it, recs):
    if not it['ignore']: return recs, []
    out = [r for r in recs if not L.in_ignore(r, it['ignore'])]; inn = [r for r in recs if L.in_ignore(r, it['ignore'])]
    return out, inn


def floor_I(it, ra, rb, metric):
    ao, ai = split_ignore(it, ra); bo, bi = split_ignore(it, rb)
    return max(0.0, L.disagreement(it, ao, bo, metric) - len(ai) - len(bi)) / 2


def lcb(v): v = np.asarray(v, float); return v.mean() - 2 * v.std(ddof=1) / math.sqrt(len(v))
def ucb(v): v = np.asarray(v, float); return v.mean() + 2 * v.std(ddof=1) / math.sqrt(len(v))


def main():
    items = load_items(); rows = A0.load_rows(f'runs_main/scored{TAG}.jsonl')
    by = defaultdict(dict)
    for x in rows:
        if x['block'] in ('MAIN', 'AGG'): by[(x['dataset'], x['model'])][(x['cand'], x['item'], x['rep'])] = x
    pool = json.loads((ROOT / 'runs/screening_pool.json').read_text())
    out = dict(theorem={}, full_sample={}, decisions={}, constants={})
    for ds in A0.DATASETS:
        its = [i for i in items.values() if i['dataset'] == ds]; nbar = float(np.mean([i['n'] for i in its]))
        ids = sorted(i['id'] for i in its); scr = sorted(pool[ds]); out['constants'][ds] = dict(mean_n=round(nbar, 2))
        metrics = ['point'] if its[0]['task'] == 'point' else ['iou75', 'iou50', 'point']
        for m in A0.MODELS:
            cell = by[(ds, m)]; key = f'{ds}|{m}'
            if not cell: continue
            recs = {k: A0.recs_of(v) for k, v in cell.items()}
            cost = {(c, i): (cell[(c, i, 0)]['in_tok'] * PRICE[m][0] + cell[(c, i, 0)]['out_tok'] * PRICE[m][1]) / 1e6 for c in CANDS for i in ids}
            for met in metrics:
                mt = A0.metric_of(its[0], met)
                E = {(c, i): cell[(c, i, 0)][f'E_{met}'] for c in CANDS for i in ids}
                F = {}; viol = viol_plain = tot = 0
                for c in CANDS:
                    for i in scr:
                        it = items[i]; a, b = recs[(c, i, 0)], recs[(c, i, 1)]
                        F[(c, i)] = floor_I(it, a, b, mt)
                        lhs = cell[(c, i, 0)][f'E_{met}'] + cell[(c, i, 1)][f'E_{met}']
                        tot += 1; viol += int(2 * F[(c, i)] > lhs + 1e-9)
                        viol_plain += int(L.disagreement(it, a, b, mt) > lhs + 1e-9)     # the uncorrected floor against the ignore-aware loss
                out['theorem'][f'{key}|{met}'] = dict(pairs=tot, violations_ignore_aware_floor=viol, violations_plain_floor_vs_ignore_loss=viol_plain)
                full = {c: float(np.mean([E[(c, i)] for i in ids])) for c in CANDS}
                out['full_sample'][f'{key}|{met}'] = dict(mean_E={c: round(v, 3) for c, v in full.items()},
                                                          meets={str(a): sorted(c for c in CANDS if full[c] <= a * nbar) for a in A_GRID})
                rng = random.Random(f'v2:{key}:{met}')
                splits = []
                for _ in range(SPLITS):
                    S = rng.sample(scr, NS); rest = [i for i in ids if i not in S]; lab = rng.sample(rest, 10); ev = [i for i in rest if i not in lab]
                    splits.append((S, lab, ev))
                for a in A_GRID:
                    Areq = a * nbar; res = defaultdict(lambda: defaultdict(list)); cand_rej = defaultdict(list)
                    for S, lab, ev in splits:
                        tE = {c: float(np.mean([E[(c, i)] for i in ev])) for c in CANDS}; feasible = min(tE.values()) <= Areq
                        cc = {c: float(np.mean([cost[(c, i)] for i in S])) for c in CANDS}
                        lE = {c: [E[(c, i)] for i in lab] for c in CANDS}
                        fl = {c: [F[(c, i)] for i in S] for c in CANDS}
                        rej_lcb = {c: lcb(fl[c]) > Areq for c in CANDS}; rej_mean = {c: float(np.mean(fl[c])) > Areq for c in CANDS}
                        for c in CANDS:
                            if tE[c] <= Areq: cand_rej['lcb'].append(rej_lcb[c]); cand_rej['mean'].append(rej_mean[c])
                        def three_state(surv, labels_used):
                            if not surv: return 'FAILS', None, labels_used
                            labels_used = 10
                            surv = [c for c in surv if lcb(lE[c]) <= Areq]
                            if not surv: return 'FAILS', None, labels_used
                            acc = [c for c in surv if ucb(lE[c]) <= Areq]
                            if acc: return 'MEETS', min(acc, key=lambda c: (cc[c], np.mean(lE[c]))), labels_used
                            return 'INSUFFICIENT', min(surv, key=lambda c: (np.mean(lE[c]), cc[c])), labels_used
                        best = min(CANDS, key=lambda c: (np.mean(lE[c]), cc[c]))
                        dec = {'ENG-ship': ('MEETS', best, 10) if np.mean(lE[best]) <= Areq else ('FAILS', None, 10),
                               'ENG-CI': three_state(list(CANDS), 0),
                               'FSR3': three_state([c for c in CANDS if not rej_lcb[c]], 0),
                               'FSR3-mean': three_state([c for c in CANDS if not rej_mean[c]], 0)}
                        for name, (st, pick, nl) in dec.items():
                            r = res[name]; r['state'].append(st); r['labels'].append(nl); r['feasible'].append(feasible)
                            r['false_accept'].append(st == 'MEETS' and tE[pick] > Areq)
                            r['true_accept'].append(st == 'MEETS' and tE[pick] <= Areq)
                            r['false_fail'].append(st == 'FAILS' and feasible)
                            r['true_fail'].append(st == 'FAILS' and not feasible)
                            if st == 'MEETS' and feasible:
                                cheapest_ok = min(cc[c] for c in CANDS if tE[c] <= Areq); r['cost_ratio'].append(cc[pick] / max(1e-12, cheapest_ok))
                    summ = {}
                    for name, r in res.items():
                        st = r['state']; n = len(st)
                        summ[name] = dict(MEETS=round(st.count('MEETS') / n, 3), FAILS=round(st.count('FAILS') / n, 3), INSUFFICIENT=round(st.count('INSUFFICIENT') / n, 3),
                                          true_accept=round(float(np.mean(r['true_accept'])), 3), false_accept=round(float(np.mean(r['false_accept'])), 3),
                                          true_fail=round(float(np.mean(r['true_fail'])), 3), false_fail=round(float(np.mean(r['false_fail'])), 3),
                                          labels=round(float(np.mean(r['labels'])), 2),
                                          cost_ratio=round(float(np.mean(r['cost_ratio'])), 3) if r['cost_ratio'] else None)
                    out['decisions'][f'{key}|{met}|{a}'] = dict(feasible_rate=round(float(np.mean(res['ENG-ship']['feasible'])), 3), rules=summ,
                        floor_false_reject={k: (round(float(np.mean(v)), 4) if v else None) for k, v in cand_rej.items()},
                        floor_false_reject_n=len(cand_rej['lcb']))
            print('done', key, flush=True)
    (ROOT / (f'runs/analysis_e3_v2{TAG}' + ('' if NS == 10 else f'_S{NS}') + '.json')).write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    main()
