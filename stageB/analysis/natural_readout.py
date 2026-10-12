"""EXP-NR analysis: same-response readout on natural multi-class images (DOTA windows, DocLayNet pages). Labels only in scoring.
Per response: decoded records (class + centre), stated counts_by_type, both mapped by the alias map declared before the run
(runs/nr/alias_map.json); records whose class is not in the vocabulary after mapping are not counted.
  pure readout   same response: per-type error of the stated count vs the list count (arm cf, and arm lf)
  full protocol  stated count of a count-first response (cf) vs the readout of a list-first response (lf): list count, or
                 the stated count if the list carries a degeneracy flag and stated counts exist
  totals         same response (lf): |list length - n| vs |stated total - n|
  list quality   one-to-one matches of a record centre inside a true box of the same class; recall and precision of the list
Paired image bootstrap (4,000 draws), per dataset and pooled. Output: out/natural_readout.json"""
import collections, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from scipy.optimize import linear_sum_assignment
OUT = Path(__file__).parent / 'out'
RUN = f'{C.ROOT}/runs/nr'
items = {i['id']: i for i in json.load(open(f'{RUN}/items.json'))}
ALIAS = json.load(open(f'{RUN}/alias_map.json'))
rng = np.random.default_rng(20261013)


def norm(name, ds):
    t = str(name).lower().strip().replace('_', '-').replace(' ', '-')
    return ALIAS[ds].get(t, t)


def matches(recs, targets):
    if not recs or not targets: return 0
    M = np.zeros((len(recs), len(targets)), bool)
    for a, r in enumerate(recs):
        for b, t in enumerate(targets):
            x1, y1, x2, y2 = t['box']; M[a, b] = r['type'] == t['label'].lower() and x1 <= r['cx'] <= x2 and y1 <= r['cy'] <= y2
    ri, ci = linear_sum_assignment(-M.astype(float))
    return int(M[ri, ci].sum())


def rows():
    Q = C.load_requests('nr', 'main'); RS = C.load_responses('nr', 'main'); out = collections.defaultdict(dict)
    for rid, q in Q.items():
        r = RS.get(rid); mt = q['meta']; it = items[mt['item']]; ds = it['dataset']
        if r is None or r['status'] != 'completed': continue
        voc = [l.lower() for l in it['labels']]
        recs, dec, au = C.decode_shapes(r['raw_text'], mt['mode'], it['width'], it['height'], *mt['sent'], list_key='objects', count_key='counts_by_type')
        mrecs = [dict(x, type=norm(x['type'], ds)) for x in recs]; valid = [x for x in mrecs if x['type'] in voc]
        stated = None
        if isinstance(dec, dict):
            stated = collections.Counter()
            for k, v in dec.items():
                try: stated[norm(k, ds)] += int(v or 0)
                except (TypeError, ValueError): pass
        truth = collections.Counter(t['label'].lower() for t in it['targets'])
        lc = collections.Counter(x['type'] for x in valid)
        err = lambda c: sum(abs(c.get(t, 0) - truth.get(t, 0)) for t in voc)
        fl = _flags(mrecs, voc, stated, r)
        m = matches(valid, it['targets'])
        out[(mt['item'], q['model_key'], mt['arm'], mt['setting'])] = dict(
            dataset=ds, n=it['n'], stated=err(stated) if stated is not None else np.nan, listc=err(lc),
            gap=sum(abs(stated.get(t, 0) - lc.get(t, 0)) for t in voc) if stated is not None else np.nan,
            readout=(err(stated) if (fl and stated is not None) else err(lc)), flagged=bool(fl),
            tot_list=abs(len(valid) - it['n']), tot_stated=abs(sum(stated.get(t, 0) for t in voc) - it['n']) if stated is not None else np.nan,
            recall=m / it['n'] if it['n'] else 1.0, precision=m / len(valid) if valid else 1.0, unknown=len(mrecs) - len(valid),
            stated_equals_list=bool(stated is not None and all(stated.get(t, 0) == lc.get(t, 0) for t in voc)))
    return out


def _flags(mrecs, voc, stated, r):
    """the four degeneracy flags of Section 5 with this task's vocabulary"""
    n = len(mrecs); fl = []
    if n and sum(x['type'] not in voc for x in mrecs) / n > 0.1: fl.append('schema')
    if stated is not None:
        tot = sum(stated.get(t, 0) for t in voc)
        if tot and (n > 1.5 * tot or n < 0.5 * tot): fl.append('length')
    seen, rep = set(), 0
    for x in mrecs:
        k = (round(x['cx']), round(x['cy'])); rep += k in seen; seen.add(k)
    if n and rep / n > 0.1: fl.append('repeat')
    if str(r.get('finish_reason')) in ('length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS'): fl.append('trunc')
    return fl


def paired(A, ka, B, kb):
    ids = [i for i in sorted(set(A) & set(B)) if np.isfinite(A[i][ka]) and np.isfinite(B[i][kb])]
    if not ids: return None
    a = np.array([A[i][ka] for i in ids], float); b = np.array([B[i][kb] for i in ids], float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(ids), len(ids)); bs.append((b[s].mean() - a[s].mean()) / a[s].mean() if a[s].mean() else 0.0)
    return dict(n=len(ids), a=round(float(a.mean()), 3), b=round(float(b.mean()), 3),
                rel=round(float((b.mean() - a.mean()) / a.mean()) * 100, 1) if a.mean() else None,
                ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)])


D = rows(); res = {}
for ds in ('dota', 'doclaynet', 'pooled'):
    for m in C.MODELS:
        pick = lambda arm, st: {k[0]: v for k, v in D.items() if k[1] == m and k[2] == arm and k[3] == st and (ds == 'pooled' or v['dataset'] == ds)}
        cf, lf = pick('cf', 'min'), pick('lf', 'min')
        cell = dict(pure_readout_cf=paired(cf, 'stated', cf, 'listc'), pure_readout_lf=paired(lf, 'stated', lf, 'listc'),
                    full_protocol=paired(cf, 'stated', lf, 'readout'),
                    totals_lf=dict(list_mae=round(float(np.mean([v['tot_list'] for v in lf.values()])), 2),
                                   stated_mae=round(float(np.nanmean([v['tot_stated'] for v in lf.values()])), 2)),
                    readout_gap_cf=round(float(np.nanmean([v['gap'] for v in cf.values()])), 2), readout_gap_lf=round(float(np.nanmean([v['gap'] for v in lf.values()])), 2),
                    list_recall_lf=round(float(np.mean([v['recall'] for v in lf.values()])), 3),
                    list_precision_lf=round(float(np.mean([v['precision'] for v in lf.values()])), 3),
                    flagged_lf=round(float(np.mean([v['flagged'] for v in lf.values()])), 3),
                    flagged_cf=round(float(np.mean([v['flagged'] for v in cf.values()])), 3),
                    stated_equals_list_cf=round(float(np.mean([v['stated_equals_list'] for v in cf.values()])), 3),
                    stated_equals_list_lf=round(float(np.mean([v['stated_equals_list'] for v in lf.values()])), 3),
                    n_images=len(lf))
        if m == 'gem38':
            dl = pick('lf', 'default16k')
            cell['default16k_readout_vs_cf_stated'] = paired(cf, 'stated', dl, 'readout')
            cell['default16k_pure_readout'] = paired(dl, 'stated', dl, 'listc')
            cell['default16k_flagged'] = round(float(np.mean([v['flagged'] for v in dl.values()])), 3) if dl else None
        res[f'{ds}|{m}'] = cell
        print(ds, m, json.dumps({k: cell[k] for k in ('pure_readout_cf', 'pure_readout_lf', 'full_protocol')}), flush=True)
cost = 0.0; calls = 0
for sub in ('nr', 'nr_smoke'):
    Q = C.load_requests(sub, 'main'); RS = C.load_responses(sub, 'main')
    for rid, q in Q.items():
        r = RS.get(rid)
        if r: calls += 1; cost += C.cost(q['model_key'], r)
res['_run'] = dict(calls=calls, cost_usd=round(cost, 2), images={ds: sum(1 for i in items.values() if i['dataset'] == ds) for ds in ('dota', 'doclaynet')},
                   mean_objects={ds: round(float(np.mean([i['n'] for i in items.values() if i['dataset'] == ds])), 1) for ds in ('dota', 'doclaynet')},
                   mean_classes={ds: round(float(np.mean([i['n_classes'] for i in items.values() if i['dataset'] == ds])), 2) for ds in ('dota', 'doclaynet')})
json.dump(res, open(OUT / 'natural_readout.json', 'w'), indent=1)
print(json.dumps(res['_run']))
