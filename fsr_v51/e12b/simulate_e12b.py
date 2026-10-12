"""Simulation gate for E12b: scripted synthetic responses for the full manifest, then the prespecified analysis.
No model calls. Every check must pass before any E12b request is sent. Output: checks/e12b_simulation.json.

Scripted behaviour (per model)
  gpt56    A1/A3: 20% of points displaced by 30 px (placement error); A2 exact          -> A-P1 'supported'
  opus55   A1/A3: deterministic omission of all targets in the last 4 regions of the scan order (A3rev: first 4 in
           image order); A2 exact. Default setting: A2 loses half its targets, A3 exact  -> A-P1 'supported',
           A-P2 'replicated', Adef reversed
  sonnet55 A1: each target dropped with p = 0.3 independently per call; A2, A3 exact    -> A-P2 'not replicated',
           floor screen eliminates sonnet A1
  gpt61    all arms: 1 px jitter; c024 A2 adds 2 points in target-free regions; Adef A3 truncated empty answers
  gem38    every grid-OFF call returns an empty answer                                 -> D7 triggered for gem38 only
"""
import json, math, os, random, shutil, sys
from pathlib import Path

ROOT = Path(__file__).parent
SIM = ROOT / 'runs_e12b_sim'


def fmt_rows(q, pts, arm):
    reg = q['registry']; out = []
    for r in reg:
        inside = [(x, y) for x, y in pts if r['x0'] <= x < r['x0'] + r['width'] and r['y0'] <= y < r['y0'] + r['height']]
        if arm == 'A1' and not inside: continue
        st = 'found' if inside else 'empty'
        out.append(json.dumps({'region_id': r['region_id'], 'status': st, 'objects': [{'x': round(x / 1024, 4), 'y': round(y / 1024, 4)} for x, y in inside]}))
    return '\n'.join(out)


def fmt_list(pts):
    return json.dumps({'status': 'found', 'objects': [{'x': round(x / 1024, 4), 'y': round(y / 1024, 4)} for x, y in pts]})


def behave(q, Y, rng):
    m, arm, setting = q['model_key'], q['arm'], q['setting']
    pts = [tuple(p) for p in Y]; finish = 'stop'
    reg4 = [(r % 4 * 256, r // 4 * 256) for r in range(16)]
    def in_regions(p, idx): return any(x0 <= p[0] < x0 + 256 and y0 <= p[1] < y0 + 256 for x0, y0 in (reg4[i] for i in idx))
    if m == 'gpt56' and arm in ('A1', 'A3', 'A3rev'):
        pts = [(min(1023, x + 30), y) if rng.random() < .2 else (x, y) for x, y in pts]
    if m == 'opus55':
        if setting == 'default':
            if arm == 'A2': pts = pts[::2]
        elif arm in ('A1', 'A3'):
            pts = [p for p in pts if not in_regions(p, range(12, 16))]
        elif arm == 'A3rev':
            pts = [p for p in pts if not in_regions(p, range(0, 4))]
    if m == 'sonnet55' and arm == 'A1':
        pts = [p for p in pts if rng.random() > .3]
    if m == 'gpt61':
        pts = [(x + rng.uniform(-1, 1), y + rng.uniform(-1, 1)) for x, y in pts]
        if q['cell'] == 'c024' and arm == 'A2':
            occ = {(int(x // 256), int(y // 256)) for x, y in Y}
            free = [(c, r) for c in range(4) for r in range(4) if (c, r) not in occ]
            pts += [(free[0][0] * 256 + 128, free[0][1] * 256 + 128), (free[1][0] * 256 + 128, free[1][1] * 256 + 128)]
        if q['block'] == 'Adef' and arm == 'A3':
            return '', 'length'
    if m == 'gem38' and q['grid'] == 'off':
        return '', 'stop'
    raw = fmt_list(pts) if arm in ('A3', 'A3rev') else fmt_rows(q, pts, arm)
    return raw, finish


def run_sim(drop=None):
    if SIM.exists(): shutil.rmtree(SIM)
    SIM.mkdir()
    reqs = [json.loads(l) for l in (ROOT / 'runs_e12b/requests.jsonl').read_text().splitlines() if l.strip()]
    ann = {json.loads(l)['sample_id']: json.loads(l) for l in (ROOT / 'data_e12b/annotations_DEV.jsonl').read_text().splitlines()}
    shutil.copy(ROOT / 'runs_e12b/requests.jsonl', SIM / 'requests.jsonl')
    rng = random.Random(5); files = {}
    for q in reqs:
        if drop and q['request_id'] == drop: continue
        raw, fin = behave(q, [t['point'] for t in ann[q['sample_id']]['targets']], rng)
        files.setdefault(q['model_key'], []).append(json.dumps({'request_id': q['request_id'], 'status': 'completed', 'raw_text': raw,
                                                                 'finish_reason': fin, 'usage': {'input_tokens': 1000, 'output_tokens': 100}}))
    for m, L in files.items(): (SIM / f'responses_{m}.jsonl').write_text('\n'.join(L) + '\n')
    return reqs, ann


def main():
    os.environ['E12B_RUNS'] = 'runs_e12b_sim'
    sys.path.insert(0, str(ROOT))
    import importlib, analyze_e12b as Z
    importlib.reload(Z)
    checks = {}
    reqs, ann = run_sim(drop=None)
    o = Z.analyse()
    checks['floor_never_violated'] = o['A_P4_floor']['violations'] == 0 and o['A_P4_floor']['pairs'] == 5 * 3 * 2 * 24
    checks['P1_gpt56_supported'] = o['A_P1']['gpt56']['verdict'] == 'supported'
    checks['P1_opus_supported'] = o['A_P1']['opus55']['verdict'] == 'supported'
    c = o['A_P1']['opus55']['contrasts']['A2-A3 grid pooled']
    checks['opus_A2-A3_is_pure_count'] = abs(c['estimate_ci95'][0] - c['d_count'][0]) < 1e-6 and c['d_place'][0] == 0
    g = o['A_P1']['gpt56']['contrasts']['A2-A3 grid pooled']
    checks['gpt56_A2-A3_is_placement'] = g['estimate_ci95'][2] < 0 and abs(g['d_count'][0]) < abs(g['d_place'][0])
    checks['gpt61_equivalent'] = 'equivalent' in o['A_P1']['gpt61']['contrasts']['A2-A3 grid pooled']['category']
    checks['P2_opus_replicated'] = o['A_P2_scan_order']['opus55']['verdict'] == 'replicated'
    checks['P2_sonnet_not_replicated'] = o['A_P2_scan_order']['sonnet55']['verdict'] == 'not replicated'
    # exact order effect for opus: forward drops bottom row (regions 13-16) -> top-bottom = +r; reverse drops top row -> -r'
    checks['P2_opus_effect_positive'] = o['A_P2_scan_order']['opus55']['order_effect_early_minus_late'][0] > .2
    sc = {(c_['model'], c_['arm'], c_['grid']): c_ for c_ in o['A_P4_floor']['candidates']}
    checks['floor_eliminates_random_drops'] = sc[('sonnet55', 'A1', 'on')]['eliminated'] and sc[('sonnet55', 'A1', 'off')]['eliminated']
    checks['floor_misses_deterministic_drops'] = (not sc[('opus55', 'A3', 'on')]['eliminated']) and sc[('opus55', 'A3', 'on')]['bad']
    checks['floor_le_actual_everywhere'] = all(c_['floor'] <= c_['actual'] + 1e-9 for c_ in o['A_P4_floor']['candidates'])
    checks['Adef_opus_reversed_only'] = o['Adef']['opus55']['reversed'] and not any(o['Adef'][m]['reversed'] for m in ('gpt56', 'gpt61', 'sonnet55'))
    checks['D7_gem38_only'] = o['decisions']['D7 parse failure / empty > 20% on every A arm -> model leaves E3 main analysis']['models'] == ['gem38']
    checks['B2_fp_target_free'] = o['B2']['gpt61']['c024']['A2_records_in_target_free_regions'] == 2.0 and o['B2']['sonnet55']['c024']['A2_records_in_target_free_regions'] == 0.0
    checks['B1_perfect_sonnet'] = o['B1']['sonnet55']['mean_E'] == {2: 0.0, 4: 0.0, 8: 0.0}
    checks['truncation_flagged'] = o['flags']['gpt61']['Adef:A3']['truncated'] == 1.0 and o['flags']['gpt61']['Adef:A3']['empty'] == 1.0
    checks['gem_gridoff_E_equals_n'] = o['A_means']['gem38']['A2 grid off']['E'] == float(sum(len(ann[s]['targets']) for s in ann if s.startswith('e12b_A_')) / 24)
    # reconciliation
    drop = next(q['request_id'] for q in reqs if q['model_key'] == 'sonnet55' and q['arm'] == 'A2' and q['block'] == 'A')
    run_sim(drop=drop)
    try:
        Z.analyse(); checks['reconciliation_raises'] = False
    except Z.ReconciliationError:
        checks['reconciliation_raises'] = True
    o2 = Z.analyse(allow_incomplete=True)
    checks['incomplete_flagged_and_scored'] = o2['incomplete'] and o2['n_missing'] == 1 and o2['A_means']['sonnet55']['A2 grid on']['E'] + o2['A_means']['sonnet55']['A2 grid off']['E'] > 0
    shutil.rmtree(SIM)
    ok = all(checks.values())
    (ROOT / 'checks').mkdir(exist_ok=True)
    (ROOT / 'checks/e12b_simulation.json').write_text(json.dumps({'all_pass': ok, 'checks': checks}, indent=1))
    for k, v in checks.items(): print('PASS' if v else 'FAIL', k)
    print('ALL PASS' if ok else 'SOME CHECKS FAILED')


if __name__ == '__main__':
    main()
