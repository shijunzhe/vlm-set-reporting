"""Fault-injection gate for the E12 analysis (E0). No model calls.
Builds a two-scene manifest for one pseudo-model, writes perfect responses, injects faults, and checks that every
planned request reaches the denominator, that no decoder raises, and that every fault is counted."""
import json, os, random, sys
from pathlib import Path
ROOT = Path(__file__).parent
RUNS = ROOT / 'runs_e12_faults'
os.environ['E12_RUNS'] = 'runs_e12_faults'
sys.path.insert(0, str(ROOT))
import build_e12 as B
import simulate_e12 as SIM
import analyze_e12 as A

RUNS.mkdir(exist_ok=True)
for f in RUNS.glob('*'): f.unlink()
ann = {a['sample_id']: a for a in map(json.loads, (ROOT / 'data_e12/annotations_DEV.jsonl').read_text().splitlines())}
scenes = sorted(ann)[:2]
rows = [q for q in B.build(['fault'], 3) if q['sample_id'] in scenes]
(RUNS / 'requests.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in rows))
rng = random.Random(3)


def perfect(q):
    tg = [tuple(t['point']) for t in ann[q['sample_id']]['targets']]
    if q['kind'] == 'rows' and len(q['requested']) == 1:
        tg = [p for p in tg if A.region_of(*p) == q['requested'][0]]
    return SIM.respond(q, tg, 'sim_perfect', rng)


by = {(q['sample_id'], q['condition'], q['focal'], q['rep']): q for q in rows}
s0, s1 = scenes
recs, faults = [], {}
for q in rows:
    rec = {'request_id': q['request_id'], 'status': 'completed', 'raw_text': perfect(q), 'finish_reason': 'stop'}
    k = (q['sample_id'], q['condition'], q['focal'], q['rep'])
    if k == (s0, 'J-local', None, 2): faults['missing'] = q['request_id']; continue           # no record at all
    if k == (s1, 'J-local', None, 2): rec.update(status='failed', raw_text=''); faults['failed'] = q['request_id']
    if k == (s0, 'F-down', None, 0): rec['raw_text'] = '{"status":"found","objects":null}'
    if k == (s0, 'F-down', None, 1): rec['raw_text'] = rec['raw_text'][: len(rec['raw_text']) // 2]; rec['finish_reason'] = 'max_tokens'
    if k == (s0, 'J-global', None, 0):                                                        # wrong types inside objects
        lines = rec['raw_text'].splitlines(); o = json.loads(lines[0])
        o['objects'] = o['objects'] + [{'x': '0.5', 'y': 0.1}, {'x': True, 'y': 0.1}, [0.1, 0.2], None]; lines[0] = json.dumps(o)
        rec['raw_text'] = '\n'.join(lines)
    if k == (s0, 'J-global', None, 1):                                                        # truncated mid-line
        rec['raw_text'] = rec['raw_text'][: int(len(rec['raw_text']) * .6)]; rec['finish_reason'] = 'length'
    if k == (s0, 'J-global-rev', None, 0):                                                    # duplicate row and contradictory status
        lines = rec['raw_text'].splitlines(); o = json.loads(lines[0]); o2 = dict(o); o2['status'] = 'empty'
        lines.append(json.dumps(o2)); rec['raw_text'] = '\n'.join(lines)
    if k == (s0, 'A-local', None, 0): rec['raw_text'] = rec['raw_text'][: int(len(rec['raw_text']) * .7)]   # truncated array
    if k == (s0, 'A-global', None, 0): rec['raw_text'] = '```json\n' + rec['raw_text'] + '\n```'           # fenced
    if k == (s1, 'J-global', None, 0):                                                        # out-of-row record
        lines = rec['raw_text'].splitlines(); o = json.loads(lines[0]); o['objects'].append({'x': 0.9, 'y': 0.9}); lines[0] = json.dumps(o)
        rec['raw_text'] = '\n'.join(lines)
    if q['condition'] == 'D-local' and q['sample_id'] == s1 and q['rep'] == 0:              # unrequested row
        rec['raw_text'] += '\n' + json.dumps({'region_id': 'R016' if q['focal'] != 'R016' else 'R001', 'status': 'empty', 'objects': []})
    if k == (s0, 'F-up', None, 0): rec['raw_text'] = 'I cannot complete this.'
    recs.append(rec)
(RUNS / 'responses_fault.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recs))

checks = {}
try:
    A.analyze('strict'); checks['missing record makes the strict analysis refuse'] = False
except A.ReconciliationError:
    checks['missing record makes the strict analysis refuse'] = True
out = A.analyze('faults', allow_incomplete=True)
rc = out['reconciliation']['fault']; m = out['models']['fault']; ev = m['S6_failure_events']
checks['reconciliation counts: 1 missing, 1 failed, all planned in denominator'] = (rc['missing_response'] == 1 and rc['failed'] == 1 and rc['planned'] == len(rows) and not out['complete'])
n0, n1 = len(ann[s0]['targets']), len(ann[s1]['targets'])
expected = ((0 + 0 + n0) / 3 + (0 + 0 + n1) / 3) / 2          # missing and failed reps scored as empty reports
checks['missing and failed requests scored as empty reports'] = abs(m['A_full_error']['J-local'][0] - expected) < 1e-3
checks['objects:null does not raise and is counted'] = ev['F-down']['salvaged'] >= 1
checks['truncated free list: partial points recovered'] = ev['F-down']['truncated'] == 1
checks['wrong-typed objects counted as invalid'] = ev['J-global']['invalid_objects'] >= 4
checks['truncated NDJSON: missing rows and truncation counted'] = ev['J-global']['missing_rows'] >= 1 and ev['J-global']['truncated'] == 1
checks['duplicate row counted'] = ev['J-global-rev']['dup_rows'] == 1
checks['contradictory status counted'] = ev['J-global-rev']['contradictory_rows'] >= 1
checks['truncated array salvaged'] = ev['A-local']['salvaged'] >= 1
checks['code fence is not a salvage event'] = ev['A-global']['salvaged'] == 0
checks['unrequested rows counted'] = ev['D-local']['unrequested_rows'] >= 1
checks['out-of-row record shows o > 0 under row ownership'] = m['A_row_ownership_accounting_mean_per_report']['J-global']['o'] > 0
checks['prose-only answer decoded as empty, not an exception'] = ev['F-up']['calls'] == 6
checks['missing response counted in failure events'] = ev['J-local']['missing_response'] == 1 and ev['J-local']['failed_or_missing'] == 2
res = {'checks': checks, 'all_pass': all(checks.values()), 'scope': 'fault injection through the real decoders and analysis; no model calls'}
(ROOT.parent / 'checks' / 'e12_fault_injection.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
sys.exit(0 if res['all_pass'] else 1)
