"""B1' analysis: FSC count error and point matching (tau) for count-first, list-only and tiled list-only (merged in code).
Usage: python -I analyze_b1p.py <subset>"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
SUB = sys.argv[1] if len(sys.argv) > 1 else 'b1p'
Q = C.load_requests(SUB, 'main'); RS = C.load_responses(SUB, 'main'); items = json.load(open(f'{C.ROOT}/runs/{SUB}/items.json'))
rng = np.random.default_rng(0); spend = collections.Counter()
by = collections.defaultdict(dict)
for rid, q in Q.items():
    r = RS.get(rid); mt = q['meta']; m = q['model_key']
    if r is None or r['status'] != 'completed': continue
    spend[m] += C.cost(m, r); it = items[mt['item']]; W, H = it['W'], it['H']
    if mt['arm'] in ('cf', 'lo'):
        recs, dec, au = C.decode_shapes(r['raw_text'], mt['mode'], W, H, *mt['sent'], list_key='points', count_key='count')
        try: dec = int(dec) if dec is not None else None
        except Exception: dec = None
        by[(mt['item'], m)][mt['arm']] = dict(recs=recs, dec=dec)
    else:
        c = mt['crop']; cw, ch = (c[2] - c[0]) * W, (c[3] - c[1]) * H
        recs, _, _ = C.decode_shapes(r['raw_text'], mt['mode'], cw, ch, *mt['sent'], list_key='points', count_key='count')
        recs = [dict(x, cx=x['cx'] + c[0] * W, cy=x['cy'] + c[1] * H, crop=tuple(c)) for x in recs]
        by[(mt['item'], m)].setdefault('tl_parts', []).append(recs)


def merge(parts, tau):
    out = []
    for P in parts:
        for p in P:
            dup = any(q['crop'] != p['crop'] and np.hypot(q['cx'] - p['cx'], q['cy'] - p['cy']) <= tau for q in out)
            if not dup: out.append(p)
    return out


def score(recs, it):
    G = [dict(type='obj', cx=x, cy=y) for x, y in it['points']]; P = [dict(type='obj', cx=x['cx'], cy=x['cy']) for x in recs]
    m = C.match_typed(P, G, it['tau'], typed=False)
    return dict(cnt=abs(len(P) - len(G)), E=len(P) + len(G) - 2 * len(m), F1=2 * len(m) / (len(P) + len(G)) if (P or G) else 1.0, k=len(P))


rows = []
for (iid, m), d in by.items():
    it = items[iid]
    if 'cf' in d:
        s = score(d['cf']['recs'], it); rows.append(dict(item=iid, model=m, stratum=it['stratum'], arm='cf_list', **s))
        if d['cf']['dec'] is not None: rows.append(dict(item=iid, model=m, stratum=it['stratum'], arm='cf_written', cnt=abs(d['cf']['dec'] - it['n']), E=None, F1=None, k=d['cf']['dec']))
    if 'lo' in d: rows.append(dict(item=iid, model=m, stratum=it['stratum'], arm='lo', **score(d['lo']['recs'], it)))
    if len(d.get('tl_parts', [])) == 4: rows.append(dict(item=iid, model=m, stratum=it['stratum'], arm='tl', **score(merge(d['tl_parts'], it['tau']), it)))


def boot(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / max(1e-9, a[s].mean()))
    return dict(a=round(float(a.mean()), 2), b=round(float(b.mean()), 2), rel=round(float((b.mean() - a.mean()) / max(1e-9, a.mean())) * 100, 1),
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(a))


out = dict(cost={k: round(v, 2) for k, v in spend.items()}, cells={})
for m in [mm_ for mm_ in C.MODELS if any(x['model'] == mm_ for x in rows)]:
    for st in (None, 0, 1, 2):
        R_ = [x for x in rows if x['model'] == m and (st is None or x['stratum'] == st)]
        if not R_: continue
        A_ = {a: {x['item']: x for x in R_ if x['arm'] == a} for a in ('cf_list', 'cf_written', 'lo', 'tl')}
        cell = {a: dict(cnt=round(float(np.mean([x['cnt'] for x in v.values()])), 2), F1=(round(float(np.mean([x['F1'] for x in v.values() if x['F1'] is not None])), 3) if any(x['F1'] is not None for x in v.values()) else None), n=len(v)) for a, v in A_.items() if v}
        comp = {}
        for a1, a2 in (('cf_written', 'lo'), ('cf_list', 'lo'), ('cf_written', 'tl'), ('lo', 'tl'), ('cf_written', 'cf_list')):
            ids = sorted(set(A_[a1]) & set(A_[a2]))
            if len(ids) >= 3: comp[f'{a2}_vs_{a1}'] = boot([A_[a1][i]['cnt'] for i in ids], [A_[a2][i]['cnt'] for i in ids])
        ids = sorted(set(A_['lo']) & set(A_['tl']))
        if len(ids) >= 3: comp['F1_tl_minus_lo'] = round(float(np.mean([A_['tl'][i]['F1'] - A_['lo'][i]['F1'] for i in ids])), 3)
        cell['comparisons'] = comp; out['cells'][f'{m}|{"all" if st is None else "s" + str(st)}'] = cell
json.dump(dict(summary=out, rows=rows), open(f'{C.ROOT}/runs/{SUB}/b1p_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:5000])
