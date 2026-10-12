"""Ingest, score and summarize the FSR pilot (development data only; nothing here is confirmatory).
Uses the package decoder/ledger (region-local contract) and scorer (maximum matching, tau = 8 px).
"""
import json, math, random, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'pkg' / 'src'))
from fsr_ledger import Region, Ledger, decode_ndjson
from fsr_metrics import score, point_relation

MODELS = ['gpt56', 'sonnet55', 'opus55', 'gpt61', 'qwen3vl']
reqs = [json.loads(s) for s in (ROOT / 'runs/pilot_requests.jsonl').read_text().splitlines() if s.strip()]
ann = {a['sample_id']: a for a in (json.loads(s) for s in (ROOT / 'data/annotations_DEV.jsonl').read_text().splitlines())}
fam = {json.loads(s)['sample_id']: json.loads(s)['family'] for s in (ROOT / 'data/samples.jsonl').read_text().splitlines()}
REL = point_relation(8.0)

def load(m):
    out = {}
    p = ROOT / f'runs/responses_{m}.jsonl'
    if not p.exists(): return out
    for l in p.read_text().splitlines():
        x = json.loads(l)
        if x['request_id'] not in out or x['status'] == 'completed': out[x['request_id']] = x   # last completed wins
    return out

def strip_fence(raw):
    L = raw.strip().splitlines()
    if L and L[0].strip().startswith('```'): L = L[1:]
    if L and L[-1].strip() == '```': L = L[:-1]
    return L

def decode_global(raw, requested, registry, rid_prefix, W, H):
    """Regional rows with scene-normalized (x,y). Owner = declared row (the contract); also log geometric owner."""
    recs, audit = [], {'malformed_lines': 0, 'unrequested_rows': 0, 'invalid_objects': 0, 'outside_declared': 0, 'missing_ids': []}
    seen = set()
    for i, line in enumerate(strip_fence(raw)):
        if not line.strip(): continue
        try: o = json.loads(line)
        except Exception: audit['malformed_lines'] += 1; continue
        if not isinstance(o, dict) or not isinstance(o.get('region_id'), str) or not isinstance(o.get('objects'), list):
            audit['malformed_lines'] += 1; continue
        rid = o['region_id']
        if rid not in requested: audit['unrequested_rows'] += 1; continue
        seen.add(rid); g = registry[rid]
        for k, it in enumerate(o['objects']):
            try:
                x, y = float(it['x']), float(it['y']); ok = 0 <= x < 1 and 0 <= y < 1
            except Exception: ok = False
            if not ok: audit['invalid_objects'] += 1; continue
            px, py = W * x, H * y
            inside = g.x0 <= px < g.x0 + g.width and g.y0 <= py < g.y0 + g.height
            audit['outside_declared'] += (not inside)
            recs.append({'record_id': f'{rid_prefix}:{i}:{k}', 'owner': rid, 'geo_owner': geo(px, py, registry),
                         'label': 'target', 'point': [px, py], 'valid': True})
    audit['missing_ids'] = [r for r in requested if r not in seen]
    return recs, audit

def geo(px, py, registry):
    for rid, g in registry.items():
        if g.x0 <= px < g.x0 + g.width and g.y0 <= py < g.y0 + g.height: return rid
    return 'OUT'

def decode_free(raw, W, H, registry, prefix):
    txt = '\n'.join(strip_fence(raw)); recs = []; bad = 0
    try:
        o = json.loads(txt[txt.index('{'): txt.rindex('}') + 1]); items = o.get('objects', [])
    except Exception: return [], {'format_failure': True}
    for k, it in enumerate(items):
        try:
            x, y = float(it['x']), float(it['y']); assert 0 <= x < 1 and 0 <= y < 1
        except Exception: bad += 1; continue
        px, py = W * x, H * y
        recs.append({'record_id': f'{prefix}:{k}', 'owner': geo(px, py, registry), 'label': 'target', 'point': [px, py], 'valid': True})
    return recs, {'format_failure': False, 'invalid_objects': bad}

def main():
    rows = []
    by = defaultdict(list)
    for q in reqs: by[(q['model_key'], q['sample_id'], q['arm'], q['batch_size'])].append(q)
    for m in MODELS:
        R = load(m)
        for sid, a in ann.items():
            tg = a['targets']
            def finish(cond, preds, extra):
                s = score(tg, preds, REL); s.pop('matched_pairs'); loc = s.pop('local')
                s.update(model=m, sample_id=sid, family=fam[sid], condition=cond, **extra); rows.append(s)
            for arm, b in [('E1', 1), ('E1', 4), ('E1', 16), ('G', 1), ('G', 4), ('G', 16), ('M', 1), ('F', 16)]:
                qs = by.get((m, sid, arm, b), [])
                if not qs: continue
                if any(q['request_id'] not in R for q in qs): continue   # incomplete arm: not scored yet
                reg = {r['region_id']: Region(**r) for r in qs[0]['task_payload']['ordered_registry']}
                fails = sum(R[q['request_id']]['status'] != 'completed' for q in qs)
                trunc = sum((R[q['request_id']].get('finish_reason') or '') in ('length', 'max_tokens') for q in qs)
                tok_in = sum((R[q['request_id']].get('usage') or {}).get('input_tokens') or 0 for q in qs)
                tok_out = sum((R[q['request_id']].get('usage') or {}).get('output_tokens') or 0 for q in qs)
                extra = dict(failed_calls=fails, truncated_calls=trunc, calls=len(qs), tok_in=tok_in, tok_out=tok_out)
                if arm == 'E1':
                    led = Ledger(reg, qs[0]['evidence_sha256']); fmt = 0
                    for q in sorted(qs, key=lambda x: x['request_id']):
                        info = led.apply(q['request_id'], q['requested'], R[q['request_id']]['raw_text'], q['evidence_sha256'])
                        fmt += sum(info.get(k, 0) for k in ('malformed_lines', 'invalid_objects', 'duplicate_region_rows', 'contradictory_rows', 'unrequested_rows'))
                    rep = led.report(); extra.update(format_events=fmt, unresolved=len(rep['unresolved_regions']))
                    finish(f'S{b}' if b < 16 else 'J16', rep['predictions'], extra)
                elif arm == 'M':
                    preds = []; fmt = 0; unres = 0
                    for q in qs:
                        d = decode_ndjson(R[q['request_id']]['raw_text'], q['requested'], reg, q['request_id'])
                        for rid in q['requested']:
                            row = d.rows.get(rid)
                            if row is None or row['status'] not in ('empty', 'found'): unres += 1
                            if row: preds += row['records']
                        fmt += sum(d.audit.get(k, 0) for k in ('malformed_lines', 'invalid_objects', 'duplicate_region_rows', 'contradictory_rows', 'unrequested_rows'))
                    extra.update(format_events=fmt, unresolved=unres); finish('M1', preds, extra)
                elif arm == 'G':
                    preds = []; out = 0; fmt = 0; miss = 0
                    for q in qs:
                        rc, au = decode_global(R[q['request_id']]['raw_text'], q['requested'], reg, q['request_id'], 768, 768)
                        preds += rc; out += au['outside_declared']; fmt += au['malformed_lines'] + au['invalid_objects'] + au['unrequested_rows']; miss += len(au['missing_ids'])
                    extra.update(format_events=fmt, unresolved=miss, outside_declared=out, n_records=len(preds))
                    finish(f'G{b}', preds, extra)
                    # code-side ownership variant: same records, owner = geometric region
                    finish(f'G{b}geo', [dict(p, owner=p['geo_owner']) for p in preds], dict(extra))
                else:
                    q = qs[0]; rc, au = decode_free(R[q['request_id']]['raw_text'], 768, 768, reg, q['request_id'])
                    extra.update(format_events=int(au.get('format_failure', False)), unresolved=0); finish('F', rc, extra)
    (ROOT / 'runs/pilot_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows

def boot(d, B=4000, seed=1):
    if not d: return (float('nan'),) * 3
    rng = random.Random(seed); n = len(d); m = sum(d) / n
    bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return m, bs[int(.025 * B)], bs[int(.975 * B) - 1]

def summarize(rows):
    S = {(r['model'], r['sample_id'], r['condition']): r for r in rows}
    out = {}
    for m in MODELS:
        o = {}
        sids = sorted(ann)
        def diff(c1, c2, key, ss=None):
            ss = ss or sids
            return [S[(m, s, c1)][key] - S[(m, s, c2)][key] for s in ss if (m, s, c1) in S and (m, s, c2) in S]
        for c in ['F', 'J16', 'S4', 'S1', 'M1', 'G16', 'G4', 'G1']:
            v = [S[(m, s, c)] for s in sids if (m, s, c) in S]
            if v:
                o[c] = {k: round(sum(x[k] for x in v) / len(v), 3) for k in ('set_error', 'B', 'chi', 'f1', 'fp', 'fn', 'unresolved', 'format_events', 'truncated_calls', 'tok_in', 'tok_out')}
                o[c]['n_scenes'] = len(v)
        contr = {'D1 = B(S1)-B(J16)': ('S1', 'J16', 'B'), 'D4 = B(S4)-B(J16)': ('S4', 'J16', 'B'),
                 'L effect = B(M1)-B(S1)': ('M1', 'S1', 'B'), 'rho = E(J16)-E(F)': ('J16', 'F', 'set_error'),
                 'coord contract b16 = E(G16)-E(J16)': ('G16', 'J16', 'set_error'), 'coord contract b1 = E(G1)-E(S1)': ('G1', 'S1', 'set_error'),
                 'Eset S1-J16': ('S1', 'J16', 'set_error'), 'Eset M1-J16 (full crop gain)': ('M1', 'J16', 'set_error')}
        o['contrasts'] = {}
        for name, (a, b, k) in contr.items():
            d = diff(a, b, k)
            if d:
                mean, lo, hi = boot(d); sd = (sum((x - mean) ** 2 for x in d) / max(1, len(d) - 1)) ** .5
                o['contrasts'][name] = {'mean': round(mean, 2), 'ci95': [round(lo, 2), round(hi, 2)], 'sd': round(sd, 2), 'n': len(d),
                                        'by_family': {f: round(sum(diff(a, b, k, [s for s in sids if fam[s] == f])) / max(1, len([s for s in sids if fam[s] == f])), 2) for f in 'ABC'}}
        g = [S[(m, s, c)] for s in sids for c in ('G1', 'G4', 'G16') if (m, s, c) in S]
        if g: o['G_outside_declared_rate'] = round(sum(x['outside_declared'] for x in g) / max(1, sum(x['n_records'] for x in g)), 3)
        out[m] = o
    (ROOT / 'runs/pilot_summary.json').write_text(json.dumps(out, indent=1))
    return out

if __name__ == '__main__':
    rows = main(); s = summarize(rows)
    print(json.dumps(s, indent=1)[:6000])
