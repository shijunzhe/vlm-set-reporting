"""Score E3 responses into one row per request (no labels are used by decoding or flags; labels only for scoring).
Usage: python score_e3.py runs/requests_smoke.jsonl runs_smoke   -> runs_smoke/scored.jsonl and a printed summary."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L

ROOT = Path(__file__).parent
import os
TAG = os.environ.get('E3_TAG', '')          # '' = original labels; '_corr' = audit-corrected labels (audit/correct_labels.py)
PRICE = {'gpt61': (2, 10), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}
METRICS = {'point': 'point', 'iou50': ('box', .5), 'iou75': ('box', .75)}


def load_items(tag=None):
    out = {}; tag = TAG if tag is None else tag
    for ds in ('fsc147', 'sku110k', 'dota', 'doclaynet'):
        for l in (ROOT / f'data/{ds}/items{tag}.jsonl').read_text().splitlines():
            if l.strip(): x = json.loads(l); out[x['id']] = x
    return out


def score(manifest, rdir):
    items = load_items(); dec_items = load_items(''); reqs = [json.loads(l) for l in (ROOT / manifest).read_text().splitlines() if l.strip()]
    resp = {}
    for f in (ROOT / rdir).glob('responses_*.jsonl'):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r['status'] == 'completed' or r['request_id'] not in resp: resp[r['request_id']] = r
    rows = []
    for q in reqs:
        it = items[q['item_id']]; r = resp.get(q['request_id'])
        ok = r is not None and r['status'] == 'completed'; raw = r.get('raw_text', '') if ok else ''
        recs, au = L.decode(dec_items[q['item_id']], q['arm'], raw, q['requested'])     # decoding always uses the original label set
        fl = L.flags(dec_items[q['item_id']], q['arm'], raw, (r or {}).get('finish_reason'), ok, au)
        if TAG == '_agn' and it['labels'] == ['obj']: recs = [dict(x, label='obj') for x in recs]   # class-agnostic scoring view
        u = (r or {}).get('usage') or {}
        row = dict(request_id=q['request_id'], block=q['block'], dataset=q['dataset'], item=q['item_id'], model=q['model_key'], cand=q['cand'],
                   rep=q['rep'], quadrant=q.get('quadrant'), n=it['n'], stratum=it['stratum'], n_pred=len(recs), flags=fl, missing=r is None,
                   in_tok=u.get('input_tokens') or 0, out_tok=u.get('output_tokens') or 0,
                   think=u.get('thinking_tokens') or u.get('reasoning_tokens') or 0,
                   recs=[[round(x['cx'], 1), round(x['cy'], 1), x['label'], [round(v, 1) for v in x['box']] if x['box'] else None] for x in recs])
        for name, mtr in METRICS.items():
            if it['task'] == 'point' and name != 'point': continue
            E, tp, npred = L.error(it, recs, mtr); row[f'E_{name}'] = E
            row[f'F1_{name}'] = (2 * tp / (it['n'] + npred)) if (it['n'] + npred) else 1.0
        rows.append(row)
    (ROOT / rdir / f'scored{TAG}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
    return rows


def summary(rows):
    g = defaultdict(list)
    for x in rows: g[(x['dataset'], x['model'], x['cand'])].append(x)
    for k, v in sorted(g.items()):
        line = {m: round(float(np.mean([x[f'F1_{m}'] for x in v])), 3) for m in METRICS if f'F1_{m}' in v[0]}
        e = {m: round(float(np.mean([x[f'E_{m}'] for x in v])), 1) for m in METRICS if f'E_{m}' in v[0]}
        cost = sum((x['in_tok'] * PRICE[x['model']][0] + x['out_tok'] * PRICE[x['model']][1]) / 1e6 for x in v)
        print(k, 'n', round(float(np.mean([x['n'] for x in v])), 1), 'pred', round(float(np.mean([x['n_pred'] for x in v])), 1), 'F1', line, 'E', e,
              'flags', round(float(np.mean([x['flags']['any'] for x in v])), 2), 'out_tok', int(np.mean([x['out_tok'] for x in v])), 'cost/call', round(cost / len(v), 4),
              'missing', sum(x['missing'] for x in v))


if __name__ == '__main__':
    summary(score(sys.argv[1], sys.argv[2]))
