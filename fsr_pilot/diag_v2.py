"""Region-level diagnostics for FSR pilot v2 (development data; descriptive only).
Writes runs_v2/region_rows.jsonl (one row per model x scene x arm x region) and runs_v2/diag.json.
"""
import json, math, re, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'pkg' / 'src'))
import analyze_v2 as A
from fsr_ledger import Region, decode_ndjson
from fsr_metrics import score, point_relation, maximum_matching

TAU = 11.0
DIS = json.loads((ROOT / 'data_v2/distractors_DEV.json').read_text())
MODELS = A.MODELS

def load(m):
    R = {}
    for l in (ROOT / f'runs_v2/responses_{m}.jsonl').read_text().splitlines():
        x = json.loads(l)
        if x['request_id'] not in R or x['status'] == 'completed': R[x['request_id']] = x
    return R

def qwen_status(raw):
    st = {}
    for line in A.strip_fence(raw):
        try: o = json.loads(line); st[o['region_id']] = o.get('status')
        except Exception: pass
    return st

def nearest(p, pts):
    best = (1e9, None)
    for t in pts:
        d = math.hypot(p[0] - t[0], p[1] - t[1])
        if d < best[0]: best = (d, t)
    return best

def get_preds(m, qs, R, arm):
    reg = {r['region_id']: Region(**r) for r in qs[0]['task_payload']['ordered_registry']}
    preds = []
    for q in qs:
        raw = R[q['request_id']]['raw_text']
        if arm == 'F': rc, _ = A.dec_free(raw, reg, q['request_id'], 1024 / 1000 if m == 'qwen3vl' else 1024)
        elif m == 'qwen3vl': rc, _ = A.dec_rows_xy(raw, q['requested'], reg, q['request_id'], 1024 / 1000)
        else:
            d = decode_ndjson(raw, q['requested'], reg, q['request_id']); rc = [x for row in d.rows.values() for x in row['records']]
        preds += rc
    return preds

def tau_sweep(taus=(6, 11, 16)):
    by = defaultdict(list)
    for q in A.reqs: by[(q['model_key'], q['sample_id'], q['arm'])].append(q)
    res = []
    for m in MODELS:
        R = load(m)
        for sid, a in A.ann.items():
            for arm in ('F', 'J', 'S1', 'C1'):
                p = get_preds(m, by[(m, sid, arm)], R, arm)
                row = dict(model=m, sid=sid, dose=len(a['targets']), arm=arm)
                for t in taus: row[f'E{t}'] = score(a['targets'], p, point_relation(float(t)))['set_error']
                res.append(row)
    (ROOT / 'runs_v2/tau_sweep.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in res))

def main():
    by = defaultdict(list)
    for q in A.reqs: by[(q['model_key'], q['sample_id'], q['arm'])].append(q)
    out, preds_all = [], []
    for m in MODELS:
        R = load(m)
        for sid, a in A.ann.items():
            tg = a['targets']; n = len(tg)
            for arm in ('F', 'J', 'S1', 'C1'):
                qs = by[(m, sid, arm)]
                reg = {r['region_id']: Region(**r) for r in qs[0]['task_payload']['ordered_registry']}
                preds, meta = [], {}
                for q in qs:
                    r = R[q['request_id']]; raw = r['raw_text']
                    tok = (r.get('usage') or {}).get('output_tokens') or 0
                    if arm == 'F':
                        rc, _ = A.dec_free(raw, reg, q['request_id'], 1024 / 1000 if m == 'qwen3vl' else 1024)
                        st = {}
                    elif m == 'qwen3vl':
                        rc, _ = A.dec_rows_xy(raw, q['requested'], reg, q['request_id'], 1024 / 1000)
                        st = qwen_status(raw)
                    else:
                        d = decode_ndjson(raw, q['requested'], reg, q['request_id'])
                        rc = [x for row in d.rows.values() for x in row['records']]
                        st = {k: v['status'] for k, v in d.rows.items()}
                    for i, rid in enumerate(q['requested']):
                        meta[rid] = {'pos': i + 1 if arm == 'J' else None, 'status': st.get(rid, 'missing') if arm != 'F' else None,
                                     'call_tok_out': tok, 'call_regions': len(q['requested'])}
                    preds += rc
                s = score(tg, preds, point_relation(TAU))
                matched = s['matched_pairs']; mt = {i for i, _ in matched}; mp = {j for _, j in matched}
                # prediction-level classification
                tpts = [t['point'] for t in tg]
                for j, p in enumerate(preds):
                    dist, near = nearest(p['point'], tpts)
                    if j in mp:
                        i = [i for i, jj in matched if jj == j][0]; t = tg[i]['point']
                        cls = 'tp'; dx, dy = p['point'][0] - t[0], p['point'][1] - t[1]
                    else:
                        dx = dy = None
                        un = [tg[i]['point'] for i in range(n) if i not in mt]
                        du, _ = nearest(p['point'], un) if un else (1e9, None)
                        if dist <= TAU: cls = 'dup'             # within tau of an already-matched target
                        elif nearest(p['point'], [(x, y) for x, y, _ in DIS[sid]])[0] <= TAU: cls = 'on_distractor'
                        elif du <= 3 * TAU: cls = 'near_miss'   # an unmatched target within 3 tau: localisation error
                        else:
                            # frame shift: would it match a target after shifting by a whole region?
                            sh = any(nearest((p['point'][0] + ox, p['point'][1] + oy), tpts)[0] <= TAU
                                     for ox in (-256, 0, 256) for oy in (-256, 0, 256) if (ox, oy) != (0, 0))
                            cls = 'shifted_region' if sh else 'far'
                    preds_all.append({'model': m, 'sid': sid, 'dose': n, 'arm': arm, 'owner': p['owner'], 'cls': cls,
                                      'dist': round(dist, 2), 'dx': dx, 'dy': dy, 'x': round(p['point'][0], 1), 'y': round(p['point'][1], 1)})
                for rid in reg:
                    loc = s['local'].get(rid, {'n': 0, 'r': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'd': 0})
                    out.append(dict(model=m, sid=sid, dose=n, arm=arm, region=rid, k=loc['n'], r=loc['r'], t=loc['tp'],
                                    fp=loc['fp'], fn=loc['fn'], d=loc['d'], **meta.get(rid, {})))
    (ROOT / 'runs_v2/region_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out))
    (ROOT / 'runs_v2/pred_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in preds_all))
    print(len(out), len(preds_all))

if __name__ == '__main__':
    main() if len(sys.argv) == 1 else tau_sweep()
