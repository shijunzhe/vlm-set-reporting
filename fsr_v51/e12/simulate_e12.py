"""E0 gate for the E12 pipeline: simulated model outputs with known behaviour, run through the real
decoders and the prespecified analysis, with assertions on the recovered effects. No model calls.

Simulated behaviours (each acts as one "model"):
  sim_perfect  every condition reports the true set with 1 px Gaussian noise
  sim_spatial  global-frame outputs have a per-call vertical scale error y' = a_c y, a_c ~ N(1.015, 0.006) (H-SP);
               local frames apply a_c within the region
  sim_seq      global-frame outputs follow a random walk over the output index with mean drift +15 px per list
               and per-step s.d. 6/sqrt(N) px (H-SQ)
  sim_stop     direct single-region requests return one point and status "found"; everything else perfect
"""
import json, os, random, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent
SIM = ROOT / 'runs_e12_sim'
sys.path.insert(0, str(ROOT))
import build_e12 as B

MODELS = ['sim_perfect', 'sim_spatial', 'sim_seq', 'sim_stop']


def respond(q, targets, model, rng):
    reg_of = lambda x, y: f'R{int(y // 256) * 4 + int(x // 256) + 1:03d}'
    pts = [(x + rng.gauss(0, 1), y + rng.gauss(0, 1)) for x, y in targets]
    if q['kind'] == 'free':
        pts.sort(key=lambda p: (p[1], p[0]) if q['order'] == 'down' else (-p[1], p[0]))
        groups = [(None, pts)]
    else:
        groups = [(r, sorted([p for p in pts if reg_of(*p) == r], key=lambda p: (p[1], p[0]))) for r in q['requested']]
        if model == 'sim_stop' and q['condition'] in ('D-local', 'D-global'):
            groups = [(r, g[:1]) for r, g in groups]
    total = max(1, sum(len(g) for _, g in groups) - 1); idx = 0; out_groups = []
    a_c = rng.gauss(1.015, 0.006); walk = 0.0; step_sd = 6 / (total + 1) ** 0.5
    for r, g in groups:
        new = []
        for x, y in g:
            if model == 'sim_spatial':
                y = a_c * y if q['frame'] == 'global' else (y // 256) * 256 + a_c * (y % 256)
            if model == 'sim_seq' and q['frame'] == 'global':
                walk += rng.gauss(15 / total, step_sd) if idx else 0.0
                y = y + walk
            idx += 1
            new.append((min(max(x, 0), 1023.9), min(max(y, 0), 1023.9)))
        out_groups.append((r, new))
    if q['kind'] == 'free':
        return json.dumps({'status': 'found', 'objects': [{'x': round(x / 1024, 4), 'y': round(y / 1024, 4)} for x, y in out_groups[0][1]]})
    rows = []
    for r, g in out_groups:
        k = int(r[1:]) - 1; x0, y0 = (k % 4) * 256, (k // 4) * 256
        if q['frame'] == 'local':
            objs = [{'u': round(min((x - x0) / 256, .999), 3), 'v': round(min((y - y0) / 256, .999), 3)} for x, y in g]
        else:
            objs = [{'x': round(x / 1024, 4), 'y': round(y / 1024, 4)} for x, y in g]
        rows.append({'region_id': r, 'status': 'found' if objs else 'empty', 'objects': objs})
    return json.dumps(rows) if q['format'] == 'array' else '\n'.join(json.dumps(o) for o in rows)


def main():
    SIM.mkdir(exist_ok=True)
    rows = B.build(MODELS, 3)
    (SIM / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    ann = {a['sample_id']: a for a in map(json.loads, (ROOT / 'data_e12/annotations_DEV.jsonl').read_text().splitlines())}
    rng = random.Random(5)
    for m in MODELS:
        with open(SIM / f'responses_{m}.jsonl', 'w') as f:
            for q in rows:
                if q['model_key'] != m: continue
                tg = [tuple(t['point']) for t in ann[q['sample_id']]['targets']]
                if q['condition'] == 'M-local' or q['kind'] == 'rows' and len(q['requested']) == 1:
                    tg = [p for p in tg if f'R{int(p[1] // 256) * 4 + int(p[0] // 256) + 1:03d}' == q['requested'][0]]
                f.write(json.dumps({'request_id': q['request_id'], 'status': 'completed', 'raw_text': respond(q, tg, m, rng),
                                    'finish_reason': 'stop'}) + '\n')
    env = dict(os.environ, E12_RUNS='runs_e12_sim')
    subprocess.run([sys.executable, str(ROOT / 'analyze_e12.py'), 'sim'], env=env, check=True, capture_output=True)
    res = json.loads((SIM / 'analysis_sim.json').read_text())['models']
    checks = {}
    P = res['sim_perfect']
    checks['perfect: all full-image errors ~0'] = all(v[0] < 1 for v in P['A_full_error'].values())
    checks['perfect: subset execution equivalent within margin (both frames)'] = all(
        P['P1_subset_execution'][f]['dE_direct_minus_geometric_projection']['category'].startswith('equivalent') for f in ('local', 'global'))
    checks['perfect: chi_match = 0 and o = 0 under row ownership'] = all(v.get('chi_match', 0) == 0 and v.get('o', 0) == 0
        for v in P['A_row_ownership_accounting_mean_per_report'].values())
    S = res['sim_spatial']
    checks['spatial: frame effect positive (global worse)'] = S['P2_frame_global_minus_local']['category'].startswith('positive')
    Q = res['sim_seq']
    for pair in ('F-down/F-up', 'J-global/J-global-rev'):
        checks[f'spatial {pair}: flip test says spatial'] = S['S5_mechanism'][pair]['flip_test_verdict'].startswith('spatial')
        checks[f'sequential {pair}: flip test says sequential'] = Q['S5_mechanism'][pair]['flip_test_verdict'].startswith('sequential')
        checks[f'perfect {pair}: flip test says none'] = P['S5_mechanism'][pair]['flip_test_verdict'].startswith('none')
        for c in pair.split('/'):
            checks[f'order adherence measured and high ({pair}:{c})'] = (S['S5_mechanism'][pair][c]['order_adherence_fraction'] or 0) >= .9
    T = res['sim_stop']
    checks['stop: direct worse than geometric projection (both frames)'] = all(
        T['P1_subset_execution'][f]['dE_direct_minus_geometric_projection']['category'].startswith('positive') for f in ('local', 'global'))
    checks['stop: unmatched mass is all harmful (helpful mass 0), the case in which the bound is attained'] = all(
        T['P3_theorem1_on_direct_vs_projection'][f]['helpful_mass'][0] == 0 for f in ('local', 'global'))
    checks['stop: mask control better than direct'] = T['secondary']['S3_evidence_control_M_minus_D_local']['category'].startswith('negative')
    checks['stop: catastrophic under-report flagged as found'] = T['S6_failure_events']['D-local']['catastrophic_marked_found'] > 0
    checks['delta rule returns a decision for every behaviour'] = all('outcome' in res[m]['secondary']['S4_delta_rule'] for m in MODELS)
    out = {'checks': checks, 'all_pass': all(checks.values()),
           'scope': 'simulated outputs through the real decoders and analysis; no model calls'}
    (ROOT.parent / 'checks' / 'e12_pipeline_simulation.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not out['all_pass']:
        print(json.dumps({m: res[m]['S5_mechanism'] for m in MODELS}, indent=1)[:6000]); sys.exit(1)


if __name__ == '__main__':
    main()
