"""v5.4 supplementary manifest for E12b (369 requests). No model calls.

S1  matched-cap control (96): GPT-5.6 and GPT-6.1, default reasoning, output cap 16,384 (setting 'default16k'),
    arms A2 and A3, grid on, the 24 block-A scenes, rep 0. Compared with block Adef (same prompts, cap 32,768).
S2  B1v2 (240): block B1 rebuilt with registry-consistent examples (supp/examples.py); every request is validated.
S3  whole-request retries (33): an identical re-request of the base report (fresh call, same model and setting) for
    (a) the 28 reports that triggered the prespecified repair (strong engineering baseline: same triggers, whole retry)
    (b) the 5 reports on which only the API/empty/parse-failure baseline would retry (missing branch of v5.3).
Output: runs_e12b_supp/requests.jsonl
"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent; E12 = ROOT.parent / 'e12'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(E12)); sys.path.insert(0, str(Path(__file__).parent))
import build_e12b as BB
import examples as X
import analyze_e12 as A

out = []
man = {json.loads(l)['request_id']: json.loads(l) for l in (ROOT / 'runs_e12b/requests.jsonl').read_text().splitlines() if l.strip()}

# ---- S1 matched cap
for q in man.values():
    if q['block'] == 'Adef' and q['model_key'] in ('gpt56', 'gpt61'):
        out.append(dict(q, request_id=q['request_id'].replace('E12b:Adef:', 'E12bS:CAP16:').replace(':default:', ':default16k:'),
                        block='CAP16', setting='default16k'))

# ---- S2 B1v2
for q in man.values():
    if q['block'] != 'B1': continue
    reg = q['registry']
    sysm = q['system'].replace(BB.EX['A2'], X.a2_example(reg))
    assert sysm != q['system']
    ok, why = X.validate(sysm, reg); assert ok, why
    out.append(dict(q, request_id=q['request_id'].replace('E12b:B1:', 'E12bS:B1v2:'), block='B1v2', system=sysm,
                    prompt_sha256=hashlib.sha256((sysm + '\n---\n' + q['user']).encode()).hexdigest()))

# ---- S3 retries
bases = [json.loads(l) for l in (ROOT / 'runs_e12b/repair_bases.jsonl').read_text().splitlines() if l.strip()]
man12 = {json.loads(l)['request_id']: json.loads(l) for l in (E12 / 'runs_e12/requests.jsonl').read_text().splitlines() if l.strip()}
raws = {}
for f in list((ROOT / 'runs_e12b').glob('responses_*.jsonl')) + list((E12 / 'runs_e12').glob('responses_*.jsonl')):
    for l in f.read_text().splitlines():
        if l.strip(): r = json.loads(l); raws[r['request_id']] = r
why = Counter()
for b in bases:
    q = man.get(b['base_id']) or man12[b['base_id']]
    rb = raws[b['base_id']]; raw = rb.get('raw_text', '') if rb['status'] == 'completed' else ''
    cp = (not raw.strip()) or A.decode(q, raw)[1]['salvaged'] > 0
    if not (b['flagged'] or cp): continue
    why['ours' if b['flagged'] else 'cp_only'] += 1
    r = dict(q, request_id='E12bS:RETRY:' + b['base_id'], block='RETRY', base_id=b['base_id'], setting='min',
             retry_for='strong_and_cp' if (b['flagged'] and cp) else ('strong' if b['flagged'] else 'cp_only'))
    if b['source'] == 'E12':
        r.update(image_path=str(E12 / q['image_path']), reference_path=str(E12 / q['reference_path']), arm=q['condition'],
                 grid='on', granularity=4, cell='e12', glyph=22)
    out.append(r)
assert len({r['request_id'] for r in out}) == len(out)
(ROOT / 'runs_e12b_supp').mkdir(exist_ok=True)
p = ROOT / 'runs_e12b_supp/requests.jsonl'; p.write_text(''.join(json.dumps(r) + '\n' for r in out))
print(len(out), 'requests', hashlib.sha256(p.read_bytes()).hexdigest())
print(Counter(r['block'] for r in out), dict(why)); print(Counter(r['model_key'] for r in out))
