"""Score and summarize FSR pilot v2 (development data; not confirmatory). tau = 11 px, maximum matching."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'pkg' / 'src'))
from fsr_ledger import Region, decode_ndjson
from fsr_metrics import score, point_relation
from analyze_pilot import strip_fence

MODELS = ['gpt56', 'sonnet55', 'opus55', 'gpt61', 'qwen3vl']
W = H = 1024
reqs = [json.loads(s) for s in (ROOT / 'runs_v2/requests.jsonl').read_text().splitlines() if s.strip()]
ann = {a['sample_id']: a for a in (json.loads(s) for s in (ROOT / 'data_v2/annotations_DEV.jsonl').read_text().splitlines())}
dose = {s: ann[s]['targets'] and len(ann[s]['targets']) for s in ann}
REL = point_relation(11.0)

def geo(px, py, reg):
    for rid, g in reg.items():
        if g.x0 <= px < g.x0 + g.width and g.y0 <= py < g.y0 + g.height: return rid
    return 'OUT'

def dec_rows_xy(raw, requested, reg, prefix, scale):
    recs, au = [], {'wrong_region_rows': 0, 'missing': 0, 'malformed': 0, 'outside_declared': 0}
    seen = set()
    for i, line in enumerate(strip_fence(raw)):
        if not line.strip(): continue
        try: o = json.loads(line)
        except Exception: au['malformed'] += 1; continue
        if not isinstance(o, dict) or not isinstance(o.get('objects'), list): au['malformed'] += 1; continue
        rid = o.get('region_id')
        if rid not in requested: au['wrong_region_rows'] += 1; continue
        seen.add(rid)
        for k, it in enumerate(o['objects']):
            try: px, py = float(it['x']) * scale, float(it['y']) * scale
            except Exception: au['malformed'] += 1; continue
            if not (0 <= px < W and 0 <= py < H): au['malformed'] += 1; continue
            au['outside_declared'] += geo(px, py, reg) != rid
            recs.append({'record_id': f'{prefix}:{i}:{k}', 'owner': rid, 'label': 'target', 'point': [px, py], 'valid': True})
    au['missing'] = len([r for r in requested if r not in seen]); return recs, au

def dec_free(raw, reg, prefix, scale):
    txt = '\n'.join(strip_fence(raw))
    try: items = json.loads(txt[txt.index('{'): txt.rindex('}') + 1]).get('objects', [])
    except Exception: return [], {'malformed': 1}
    recs = []
    for k, it in enumerate(items):
        try: px, py = float(it['x']) * scale, float(it['y']) * scale
        except Exception: continue
        if 0 <= px < W and 0 <= py < H:
            recs.append({'record_id': f'{prefix}:{k}', 'owner': geo(px, py, reg), 'label': 'target', 'point': [px, py], 'valid': True})
    return recs, {'malformed': 0}

def dups(targets, preds, matched):
    mp = {j for _, j in matched}
    return sum(1 for j, p in enumerate(preds) if j not in mp and any(REL(t, p) for t in targets))

def main():
    by = defaultdict(list)
    for q in reqs: by[(q['model_key'], q['sample_id'], q['arm'])].append(q)
    rows = []
    for m in MODELS:
        R = {}
        p = ROOT / f'runs_v2/responses_{m}.jsonl'
        if not p.exists(): continue
        for l in p.read_text().splitlines():
            x = json.loads(l)
            if x['request_id'] not in R or x['status'] == 'completed': R[x['request_id']] = x
        for sid, a in ann.items():
            for arm in ('F', 'J', 'S1', 'C1'):
                qs = by[(m, sid, arm)]
                if not qs or any(q['request_id'] not in R for q in qs): continue
                reg = {r['region_id']: Region(**r) for r in qs[0]['task_payload']['ordered_registry']}
                preds = []; au = defaultdict(int)
                for q in qs:
                    raw = R[q['request_id']]['raw_text']
                    if arm == 'F':
                        rc, x = dec_free(raw, reg, q['request_id'], 1024 / 1000 if m == 'qwen3vl' else 1024)
                    elif m == 'qwen3vl':
                        rc, x = dec_rows_xy(raw, q['requested'], reg, q['request_id'], 1024 / 1000)
                    else:
                        d = decode_ndjson(raw, q['requested'], reg, q['request_id'])
                        rc = [r for row in d.rows.values() for r in row['records']]
                        x = {'wrong_region_rows': d.audit['unrequested_rows'], 'missing': len(d.audit['missing_ids']),
                             'malformed': d.audit['malformed_lines'] + d.audit['invalid_objects']}
                    preds += rc
                    for k, v in x.items(): au[k] += v
                    au['truncated'] += (R[q['request_id']].get('finish_reason') or '') in ('length', 'max_tokens')
                    au['failed'] += R[q['request_id']]['status'] != 'completed'
                    u = R[q['request_id']].get('usage') or {}
                    au['tok_in'] += u.get('input_tokens') or 0; au['tok_out'] += u.get('output_tokens') or 0
                s = score(a['targets'], preds, REL)
                s['dup'] = dups(a['targets'], preds, s.pop('matched_pairs')); s.pop('local')
                s.update(model=m, sample_id=sid, dose=len(a['targets']), arm=arm, **au); rows.append(s)
    (ROOT / 'runs_v2/scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows

def boot(d, B=4000, seed=1):
    n = len(d); m = sum(d) / n; rng = random.Random(seed)
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(m, 2), round(bs[int(.025 * B)], 2), round(bs[int(.975 * B) - 1], 2)

def summarize(rows):
    S = {(r['model'], r['sample_id'], r['arm']): r for r in rows}
    out = {}
    for m in MODELS:
        o = {}
        for dz in [16, 64, 192, 384, 'all']:
            sids = [s for s in ann if dz == 'all' or len(ann[s]['targets']) == dz]
            blk = {}
            for arm in ('F', 'J', 'S1', 'C1'):
                v = [S[(m, s, arm)] for s in sids if (m, s, arm) in S]
                if v:
                    blk[arm] = {k: round(sum(x.get(k, 0) for x in v) / len(v), 2) for k in
                                ('set_error', 'B', 'fn', 'fp', 'dup', 'f1', 'wrong_region_rows', 'missing', 'malformed', 'truncated', 'tok_in', 'tok_out')}
                    blk[arm]['n_scenes'] = len(v); blk[arm]['exact_set_rate'] = round(sum(x['exact_set'] for x in v) / len(v), 2)
            con = {}
            for name, a_, b_, k in [('D = B(S1)-B(J)', 'S1', 'J', 'B'), ('L = B(C1)-B(S1)', 'C1', 'S1', 'B'),
                                    ('rho = E(J)-E(F)', 'J', 'F', 'set_error'), ('crop gain = E(C1)-E(J)', 'C1', 'J', 'set_error')]:
                d = [S[(m, s, a_)][k] - S[(m, s, b_)][k] for s in sids if (m, s, a_) in S and (m, s, b_) in S]
                if len(d) >= 2: con[name] = boot(d) + (len(d),)
            blk['contrasts'] = con; o[str(dz)] = blk
        out[m] = o
    (ROOT / 'runs_v2/summary.json').write_text(json.dumps(out, indent=1))
    return out

if __name__ == '__main__':
    rows = main(); s = summarize(rows)
    for m, o in s.items():
        print('==', m)
        for dz, blk in o.items():
            line = ' '.join(f"{a}:E={blk[a]['set_error']}(fn{blk[a]['fn']},fp{blk[a]['fp']})" for a in ('F', 'J', 'S1', 'C1') if a in blk)
            print(f'  n={dz:>4} {line}')
            for k, v in blk['contrasts'].items(): print(f'        {k}: {v}')
