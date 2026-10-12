"""Pipeline test for analyze_v54 with copied responses (no model calls): CAP16 = copy of Adef, B1v2 = copy of B1,
RETRY = copy of the base answer. Expected: cap16 - cap32 = 0 everywhere; B1v2 - B1 = 0; STRONG = step-0 report
(identical retry adds nothing); API = strict report; no missing branch."""
import json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; T = ROOT / 'runs_e12b_supp_test'
if T.exists(): shutil.rmtree(T)
T.mkdir(); shutil.copy(ROOT / 'runs_e12b_supp/requests.jsonl', T / 'requests.jsonl')
src = {}
for f in list((ROOT / 'runs_e12b').glob('responses_*.jsonl')) + list((ROOT.parent / 'e12/runs_e12').glob('responses_*.jsonl')):
    for l in f.read_text().splitlines():
        if l.strip(): r = json.loads(l); src[r['request_id']] = r
outs = {}
for l in (T / 'requests.jsonl').read_text().splitlines():
    q = json.loads(l)
    if q['block'] == 'CAP16': base = q['request_id'].replace('E12bS:CAP16:', 'E12b:Adef:').replace(':default16k:', ':default:')
    elif q['block'] == 'B1v2': base = q['request_id'].replace('E12bS:B1v2:', 'E12b:B1:')
    else: base = q['base_id']
    r = dict(src[base], request_id=q['request_id']); outs.setdefault(q['model_key'], []).append(json.dumps(r))
for m, L in outs.items(): (T / f'responses_{m}.jsonl').write_text('\n'.join(L) + '\n')
env = dict(os.environ, E12B_SUPP='runs_e12b_supp_test')
subprocess.run([sys.executable, str(ROOT / 'supp/analyze_v54.py')], env=env, check=True, capture_output=True)
o = json.loads((T / 'analysis_v54.json').read_text())
checks = dict(
    cap_zero=all(v['cap16_minus_cap32'][0] == 0 for v in o['5_matched_cap'].values()),
    b1_zero=all(v.get(f'k{k}_v2 - k{k}_original', [0])[0] == 0 for v in o['6_B1v2'].values() for k in (2, 8)),
    no_missing=o['4_missing_retry_branches'] == [] and o['5_6_supp_missing'] == 0,
    strong_equals_step0=all(abs(v['mean_E_strong'] - v['mean_E0']) < 1e-9 for v in o['4_repair_by_source'].values()),
    api_equals_strict=all(abs(v['mean_E_api'] - v['mean_E_strict']) < 1e-9 for v in o['4_repair_by_source'].values()),
    none_for_gemini=o['1_absolute_floor_screening']['gem38']['NONE'] is True)
print(checks); shutil.rmtree(T)
(ROOT / 'checks/v54_pipeline_test.json').write_text(json.dumps(checks, indent=1))
print('ALL PASS' if all(checks.values()) else 'FAIL')
