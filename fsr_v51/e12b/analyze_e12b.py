"""Prespecified analysis for v5.2-E12b (written before any E12b model call).

Inputs (RUNS = $E12B_RUNS or runs_e12b): requests.jsonl, responses_<model>.jsonl; data_e12b/annotations_DEV.jsonl
(ground truth, scoring only). Output: RUNS/analysis_e12b.json and a printed summary.

Rules (inherited from E12)
  * Reconciliation: every manifest request needs a terminal record, otherwise ReconciliationError
    (allow_incomplete=True scores missing ones under the failure rule and flags the output).
  * Failure rule: a request without a completed response is an empty report.
  * Decoding: analyze_e12.decode (label-free, never raises). tau = 11 px, maximum matching.
  * Statistical unit: scene. Repetitions are averaged within scene. Paired percentile bootstrap over scenes
    (4000 resamples). Models never pooled. Equivalence margin per image: max(1, 0.05 n), averaged over the scenes.
  * Count/placement split: geometric ownership on the 4x4 registry (count part + placement part = B = E + 2 chi).

Analyses
  A-P1  per model: A2-A3 and A2-A1 (grid pooled and per grid): Delta E with Delta count and Delta placement.
        Verdicts: Opus/Sonnet 'supported' if A2 is better (95% CI below 0) against A3 or A1 and the count part carries
        at least half of the improvement; GPT-5.6 'supported' unless A2 is better with the count part carrying at least
        half. GPT-6.1 and Gemini descriptive.
  A-P2  end-of-scan omission, A3 forward vs A3rev (grid on, rep 0): order effect = mean over scenes of
        [(recall_top - recall_bottom)_forward - (recall_top - recall_bottom)_reverse] / 2 = early-half minus late-half
        recall with image position balanced. Opus/Sonnet: 'replicated' if the 95% CI is above 0.
  A-P3  grid effect E(on) - E(off) per arm and model; no predicted direction.
  A-P4  disagreement floor on the 2 repetitions of every (model, arm, grid) candidate of block A:
        samplewise check E_tau(a) + E_tau(b) >= U_{2 tau}(a, b) (must never fail); candidate floor = mean over scenes
        of U_{2 tau}/2 / n; the screen eliminates a candidate when floor > 0.05. Reported: violations, screen recall
        among candidates whose actual mean E/n > 0.05, mean E/n of eliminated vs retained.
  Adef  default reasoning vs lowest setting (A2, A3, grid on; min uses rep 0): Delta E and whether the A2-A3 contrast
        reverses (opposite signs, both 95% CIs excluding 0).
  B1    granularity 2x2 / 4x4 (= A2 grid off in block A, same scenes) / 8x8: E and Delta vs 4x4.
  B2    cells c024, c048, g14, g32 (A2 and A3, grid on, 6 scenes; descriptive intervals) against the centre
        (A scenes with n = 96, grid on, rep 0); false positives in target-free regions for c024/c048.
  Flags observable without labels: failed, empty answer, truncated, salvaged JSON, missing rows (A2), unknown rows or
        status, invalid objects or malformed rows (coordinate-contract violations). Tokens and list-price cost per model and block.
  Decision points D1-D7 of the v5.2 plan (D4 needs block C and is reported as pending here).
"""
import json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / 'e12'))
import analyze_e12 as A                                   # decode, E, TP, regional, boot, category

RUNS = ROOT / os.environ.get('E12B_RUNS', 'runs_e12b')
TAU = A.TAU
PRICE = {'gpt56': (4, 20), 'gpt61': (2, 10), 'opus55': (4, 20), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}
TRUNC = {'length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS'}
MODELS = ['gpt56', 'gpt61', 'opus55', 'sonnet55', 'gem38']


class ReconciliationError(RuntimeError):
    pass


def margin(n): return max(1.0, 0.05 * n)


def free_status(raw):
    try:
        s = raw.index('{'); v, _ = A.DEC.raw_decode(raw, s)
        return v.get('status') if isinstance(v, dict) else None
    except (ValueError, AttributeError):
        return None


def load(allow_incomplete=False):
    reqs = [json.loads(l) for l in (RUNS / 'requests.jsonl').read_text().splitlines() if l.strip()]
    ann = {json.loads(l)['sample_id']: json.loads(l) for l in (ROOT / 'data_e12b/annotations_DEV.jsonl').read_text().splitlines() if l.strip()}
    resp = {}
    for m in {q['model_key'] for q in reqs}:
        f = RUNS / f'responses_{m}.jsonl'
        if not f.exists(): continue
        for l in f.read_text().splitlines():
            if not l.strip(): continue
            r = json.loads(l)
            if r['status'] == 'completed' or r['request_id'] not in resp: resp[r['request_id']] = r   # completed wins
    missing = [q['request_id'] for q in reqs if q['request_id'] not in resp]
    if missing and not allow_incomplete:
        raise ReconciliationError(f'{len(missing)} manifest requests have no terminal record, e.g. {missing[:3]}')
    rows = []
    for q in reqs:
        r = resp.get(q['request_id'], {'status': 'missing', 'raw_text': ''})
        ok = r['status'] == 'completed'
        raw = r.get('raw_text', '') if ok else ''
        recs, au = A.decode(q, raw)
        Y = [tuple(t['point']) for t in ann[q['sample_id']]['targets']]
        P = [(z['x'], z['y']) for z in recs]
        acc = A.regional(recs, Y, 'geometry')
        u = r.get('usage') or {}
        fr = str(r.get('finish_reason'))
        st = free_status(raw) if q['kind'] == 'free' else None
        flags = dict(failed=int(not ok), empty=int(ok and not raw.strip()), truncated=int(fr in TRUNC), salvaged=int(au['salvaged'] > 0),
                     missing_rows=int(q['arm'] == 'A2' and au['missing_rows'] > 0),
                     unknown=int(au['unknown_rows'] > 0 or st == 'unknown'),
                     invalid=int(au['invalid_objects'] > 0 or au['malformed_rows'] > 0))   # contract violations, label-free
        flags['any'] = int(any(flags.values()))
        flags['broken'] = int(flags['failed'] or flags['empty'] or flags['salvaged'])
        rows.append(dict(q=q, P=P, Y=Y, recs=recs, n=len(Y), E=A.E(P, Y), count=acc['count_part'], place=acc['placement_part'],
                         flags=flags, in_tok=u.get('input_tokens') or 0, out_tok=u.get('output_tokens') or 0,
                         think=u.get('thinking_tokens') or u.get('reasoning_tokens') or 0))
    return rows, missing


def scene_means(rows, key, sel):
    """{sample_id: mean of key over the selected rows (repetitions averaged)}."""
    acc = defaultdict(list)
    for r in rows:
        if sel(r['q']): acc[r['q']['sample_id']].append(r[key])
    return {s: float(np.mean(v)) for s, v in acc.items()}


def paired(rows, sel1, sel2, key='E'):
    a, b = scene_means(rows, key, sel1), scene_means(rows, key, sel2)
    common = sorted(set(a) & set(b))
    return [a[s] - b[s] for s in common], common


def contrast(rows, sel1, sel2, nmap):
    d, common = paired(rows, sel1, sel2)
    if len(d) < 2: return {'category': 'no data'}
    mg = round(float(np.mean([margin(nmap[s]) for s in common])), 2)
    out = A.category(d, mg)
    out['d_count'] = A.boot(paired(rows, sel1, sel2, 'count')[0]); out['d_place'] = A.boot(paired(rows, sel1, sel2, 'place')[0])
    out['n_scenes'] = len(common)
    return out


def better(c):   # first condition better (lower error), 95% CI below zero
    return isinstance(c.get('estimate_ci95'), list) and c['estimate_ci95'][2] < 0


def count_share(c):
    e, k = c['estimate_ci95'][0], (c.get('d_count') or [0])[0]
    return k / e if e else 0.0


def sel(block, m, arm=None, grid=None, k=4, setting='min', rep=None, cell=None):
    def f(q):
        return (q['block'] == block and q['model_key'] == m and (arm is None or q['arm'] == arm) and (grid is None or q['grid'] == grid)
                and q['granularity'] == k and q['setting'] == setting and (rep is None or q['rep'] == rep) and (cell is None or q['cell'] == cell))
    return f


def analyse(allow_incomplete=False):
    rows, missing = load(allow_incomplete)
    nmap = {r['q']['sample_id']: r['n'] for r in rows}
    models = [m for m in MODELS if any(r['q']['model_key'] == m for r in rows)]
    out = {'incomplete': bool(missing), 'n_missing': len(missing), 'models': models}
    # ---------------- A-P1 and A-P3
    P1, P3, desc = {}, {}, {}
    for m in models:
        c = {}
        for g in ('pooled', 'on', 'off'):
            gg = None if g == 'pooled' else g
            c[f'A2-A3 grid {g}'] = contrast(rows, sel('A', m, 'A2', gg), sel('A', m, 'A3', gg), nmap)
            c[f'A2-A1 grid {g}'] = contrast(rows, sel('A', m, 'A2', gg), sel('A', m, 'A1', gg), nmap)
        wins = [k for k in ('A2-A3 grid pooled', 'A2-A1 grid pooled') if better(c[k])]
        shares = {k: round(count_share(c[k]), 2) for k in wins}
        if m in ('opus55', 'sonnet55'):
            v = 'supported' if any(s >= .5 for s in shares.values()) else ('A2 better but not through the count part' if wins else 'no A2 advantage')
        elif m == 'gpt56':
            v = 'not supported (A2 gain carried by the count part)' if any(s >= .5 for s in shares.values()) else 'supported'
        else:
            v = 'descriptive'
        P1[m] = dict(contrasts=c, A2_wins=wins, count_share=shares, verdict=v)
        P3[m] = {arm: contrast(rows, sel('A', m, arm, 'on'), sel('A', m, arm, 'off'), nmap) for arm in ('A1', 'A2', 'A3')}
        desc[m] = {f'{arm} grid {g}': {k: round(float(np.mean(list(scene_means(rows, k, sel('A', m, arm, g)).values()) or [np.nan])), 3)
                                         for k in ('E', 'count', 'place')} for arm in ('A1', 'A2', 'A3') for g in ('on', 'off')}
    out['A_means'] = desc; out['A_P1'] = P1; out['A_P3_grid'] = P3
    # ---------------- A-P2
    P2 = {}
    for m in models:
        diffs = []
        for s in sorted({r['q']['sample_id'] for r in rows if r['q']['block'] == 'Arev' and r['q']['model_key'] == m}):
            v = {}
            for name, f in (('fwd', sel('A', m, 'A3', 'on', rep=0)), ('rev', sel('Arev', m, 'A3rev', 'on', rep=0))):
                rr = [r for r in rows if r['q']['sample_id'] == s and f(r['q'])]
                if not rr: continue
                r = rr[0]; mt = {j for _, j in A.match(r['P'], r['Y'], TAU)}
                top = [j for j, t in enumerate(r['Y']) if t[1] < 512]; bot = [j for j, t in enumerate(r['Y']) if t[1] >= 512]
                v[name] = len([j for j in top if j in mt]) / max(1, len(top)) - len([j for j in bot if j in mt]) / max(1, len(bot))
            if len(v) == 2: diffs.append((v['fwd'] - v['rev']) / 2)
        b = A.boot(diffs)
        verdict = 'descriptive'
        if m in ('opus55', 'sonnet55'):
            verdict = 'replicated' if b and b[1] > 0 else 'not replicated'
        P2[m] = dict(order_effect_early_minus_late=b, verdict=verdict)
    out['A_P2_scan_order'] = P2
    # ---------------- A-P4
    cand, viol, npairs = [], 0, 0
    for m in models:
        for arm in ('A1', 'A2', 'A3'):
            for g in ('on', 'off'):
                by = defaultdict(dict)
                for r in rows:
                    if sel('A', m, arm, g)(r['q']): by[r['q']['sample_id']][r['q']['rep']] = r
                fl, ac = [], []
                for s, d in by.items():
                    if len(d) < 2: continue
                    a, b = d[0], d[1]; U = A.E(a['P'], b['P'], 2 * TAU)
                    npairs += 1; viol += int(a['E'] + b['E'] < U)
                    fl.append(U / 2 / a['n']); ac.append((a['E'] + b['E']) / 2 / a['n'])
                if fl:
                    cand.append(dict(model=m, arm=arm, grid=g, floor=round(float(np.mean(fl)), 4), actual=round(float(np.mean(ac)), 4),
                                     eliminated=bool(np.mean(fl) > .05), bad=bool(np.mean(ac) > .05)))
    el = [c for c in cand if c['eliminated']]; kept = [c for c in cand if not c['eliminated']]; bad = [c for c in cand if c['bad']]
    out['A_P4_floor'] = dict(pairs=npairs, violations=viol, candidates=cand,
                             screen_recall_among_bad=(round(sum(c['eliminated'] for c in bad) / len(bad), 3) if bad else None),
                             mean_E_per_target_eliminated=(round(float(np.mean([c['actual'] for c in el])), 4) if el else None),
                             mean_E_per_target_retained=(round(float(np.mean([c['actual'] for c in kept])), 4) if kept else None),
                             verdict=('violated: check code' if viol else ('eliminated worse than retained' if el and kept and
                                      np.mean([c['actual'] for c in el]) > np.mean([c['actual'] for c in kept]) else
                                      ('nothing eliminated' if not el else 'eliminated not worse'))))
    # ---------------- Adef
    AD = {}
    for m in models:
        d = {arm: contrast(rows, sel('Adef', m, arm, 'on', setting='default'), sel('A', m, arm, 'on', rep=0), nmap) for arm in ('A2', 'A3')}
        cmin = contrast(rows, sel('A', m, 'A2', 'on', rep=0), sel('A', m, 'A3', 'on', rep=0), nmap)
        cdef = contrast(rows, sel('Adef', m, 'A2', 'on', setting='default'), sel('Adef', m, 'A3', 'on', setting='default'), nmap)
        rev = False
        if isinstance(cmin.get('estimate_ci95'), list) and isinstance(cdef.get('estimate_ci95'), list):
            a, b = cmin['estimate_ci95'], cdef['estimate_ci95']
            rev = (a[2] < 0 and b[1] > 0) or (a[1] > 0 and b[2] < 0)
        think = [r['think'] for r in rows if sel('Adef', m, None, 'on', setting='default')(r['q'])]
        AD[m] = dict(default_minus_min=d, A2_minus_A3_min_rep0=cmin, A2_minus_A3_default=cdef, reversed=rev,
                     mean_thinking_tokens_default=round(float(np.mean(think)), 1) if think else None)
    out['Adef'] = AD
    # ---------------- B1
    B1 = {}
    b1ids = {r['q']['sample_id'] for r in rows if r['q']['block'] == 'B1'}
    for m in models:
        s4 = sel('A', m, 'A2', 'off'); res = {'mean_E': {}}
        for k in (2, 4, 8):
            f = s4 if k == 4 else sel('B1', m, 'A2', 'off', k=k)
            sm = scene_means(rows, 'E', f)
            vals = [v for s_, v in sm.items() if s_ in b1ids]
            res['mean_E'][k] = round(float(np.mean(vals)), 3) if vals else None
        for k in (2, 8):
            res[f'k{k}-k4'] = contrast(rows, sel('B1', m, 'A2', 'off', k=k),
                                       lambda q, s4=s4: s4(q) and q['sample_id'] in b1ids, nmap)
        B1[m] = res
    out['B1'] = B1
    # ---------------- B2
    B2 = {}
    for m in models:
        res = {}
        for cell in ('n96', 'c024', 'c048', 'g14', 'g32'):
            blk, rep = ('A', 0) if cell == 'n96' else ('B2', 0)
            ent = {}
            for arm in ('A2', 'A3'):
                f = sel(blk, m, arm, 'on', rep=rep, cell=cell)
                sm = scene_means(rows, 'E', f); ent[f'{arm}_E'] = round(float(np.mean(list(sm.values()))), 3) if sm else None
                if cell in ('c024', 'c048'):
                    fp = []
                    for r in rows:
                        if f(r['q']):
                            occ = {A.region_of(*t) for t in r['Y']}
                            fp.append(sum(1 for z in r['recs'] if z['geo'] not in occ))
                    ent[f'{arm}_records_in_target_free_regions'] = round(float(np.mean(fp)), 3) if fp else None
            d, _ = paired(rows, sel(blk, m, 'A2', 'on', rep=rep, cell=cell), sel(blk, m, 'A3', 'on', rep=rep, cell=cell))
            ent['A2-A3'] = A.boot(d); res[cell] = ent
        B2[m] = res
    out['B2'] = B2
    # ---------------- flags, tokens, cost
    FL, COST = {}, {}
    for m in models:
        FL[m] = {}
        for blk in ('A', 'Adef', 'Arev', 'B1', 'B2'):
            for arm in ('A1', 'A2', 'A3', 'A3rev'):
                rr = [r for r in rows if r['q']['model_key'] == m and r['q']['block'] == blk and r['q']['arm'] == arm]
                if rr: FL[m][f'{blk}:{arm}'] = {k: round(float(np.mean([r['flags'][k] for r in rr])), 3) for k in rr[0]['flags']}
        ti = sum(r['in_tok'] for r in rows if r['q']['model_key'] == m); to = sum(r['out_tok'] for r in rows if r['q']['model_key'] == m)
        COST[m] = dict(input_tokens=ti, output_tokens=to, cost_usd=round(ti / 1e6 * PRICE[m][0] + to / 1e6 * PRICE[m][1], 2))
    out['flags'] = FL; out['cost'] = COST
    # error captured by observable flags (label-free), block A
    cap = {}
    for m in models:
        rr = [r for r in rows if r['q']['model_key'] == m and r['q']['block'] == 'A']
        tot = sum(r['E'] for r in rr); cap[m] = round(sum(r['E'] for r in rr if r['flags']['any']) / tot, 3) if tot else None
    out['share_of_A_error_in_flagged_reports'] = cap
    # ---------------- decision points
    eq = [m for m in models if 'equivalent' in P1[m]['contrasts']['A2-A3 grid pooled'].get('category', '')]
    on_only = [m for m in models if better(P1[m]['contrasts']['A2-A3 grid on']) and not better(P1[m]['contrasts']['A2-A3 grid off'])]
    off_too = [m for m in models if better(P1[m]['contrasts']['A2-A3 grid off'])]
    bad_models = [m for m in models if all(FL[m][k]['broken'] > .2 for k in FL[m] if k.startswith('A:'))]
    out['decisions'] = {
        'D1 A2~A3 equivalent in >=3 models -> H1 becomes an equivalence test': dict(models=eq, triggered=len(eq) >= 3),
        'D2 A2>A3 only with grid -> E3 uses a drawn grid in every core arm': dict(models=on_only, triggered=len(on_only) > len(off_too)),
        'D3 A2>A3 also without grid -> E3 without drawn grid': dict(models=off_too, triggered=len(off_too) >= len(on_only) and len(off_too) > 0),
        'D4 C-P1 fails -> drop repair routing from S-route': 'pending (block C)',
        'D5 floor screen does not separate -> drop S-route step 1': dict(verdict=out['A_P4_floor']['verdict'],
                                                                        triggered=out['A_P4_floor']['verdict'] not in ('eliminated worse than retained',)),
        'D6 default reasoning reverses A2-A3 -> E3 main setting = default': dict(models=[m for m in models if AD[m]['reversed']],
                                                                                triggered=any(AD[m]['reversed'] for m in models)),
        'D7 parse failure / empty > 20% on every A arm -> model leaves E3 main analysis': dict(models=bad_models, triggered=bool(bad_models)),
    }
    return out


def summary(o):
    print('incomplete' if o['incomplete'] else 'complete', '| models', o['models'])
    for m in o['models']:
        print(f'== {m}  cost {o["cost"][m]}  flagged-share of A error {o["share_of_A_error_in_flagged_reports"][m]}')
        for k, v in o['A_means'][m].items(): print(f'   {k:12s} {v}')
        for k, c in o['A_P1'][m]['contrasts'].items():
            if 'pooled' in k: print(f'   {k}: {c.get("estimate_ci95")} {c.get("category")} dcount {c.get("d_count")} dplace {c.get("d_place")}')
        print('   A-P1:', o['A_P1'][m]['verdict'], '| A-P2:', o['A_P2_scan_order'][m])
        print('   grid on-off:', {a: (c.get('estimate_ci95'), c.get('category')) for a, c in o['A_P3_grid'][m].items()})
        print('   Adef:', {a: c.get('estimate_ci95') for a, c in o['Adef'][m]['default_minus_min'].items()}, 'reversed', o['Adef'][m]['reversed'],
              'thinking', o['Adef'][m]['mean_thinking_tokens_default'])
        print('   B1 mean E by k:', o['B1'][m]['mean_E'])
        print('   B2:', {c: (v['A2_E'], v['A3_E']) for c, v in o['B2'][m].items()})
    p4 = o['A_P4_floor']; print('A-P4: pairs', p4['pairs'], 'violations', p4['violations'], 'recall', p4['screen_recall_among_bad'],
                                'elim/kept', p4['mean_E_per_target_eliminated'], p4['mean_E_per_target_retained'], p4['verdict'])
    for k, v in o['decisions'].items(): print(' ', k, '->', v)


if __name__ == '__main__':
    inc = '--allow-incomplete' in sys.argv
    o = analyse(allow_incomplete=inc)
    (RUNS / 'analysis_e12b.json').write_text(json.dumps(o, indent=1, default=str))
    summary(o)
