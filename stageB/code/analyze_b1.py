"""B1 analysis (labels used only in scoring). Usage: python -I analyze_b1.py <subset>"""
import json, os, sys, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
SUB = sys.argv[1] if len(sys.argv) > 1 else 'b1'
Q = C.load_requests(SUB, 'main'); RS = C.load_responses(SUB, 'main')
T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
GT = {('scp', i): json.load(open(f'/home/claude/release/ground_truth/gt_{i:03d}.json')) for i in range(100)}
GT.update({('scpd', i): json.load(open(f'{C.ROOT}/data/scpd_gen/ground_truth/gt_{i:03d}.json')) for i in range(50)})
TOL = {'scp': 30, 'scpd': 20}
rng = np.random.default_rng(0)
rows = []; audit = collections.Counter(); spend = collections.Counter()
for rid, q in Q.items():
    r = RS.get(rid); m = q['model_key']; mt = q['meta']
    if r is None or r['status'] != 'completed': audit['missing'] += 1; continue
    spend[m] += C.cost(m, r)
    trunc = str(r.get('finish_reason')) in ('length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS')
    sw, sh = mt['sent']
    if mt['task'] in ('scp', 'scpd'):
        allrecs, dec, au = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, sw, sh)
        recs = [x for x in allrecs if x['type'] in T5]; untyped = len(allrecs) - len(recs)
        G = GT[(mt['task'], mt['item'])]['shapes']; gc = collections.Counter(s['type'] for s in G); lc = collections.Counter(s['type'] for s in recs)
        tol = TOL[mt['task']]; mm = C.match_typed(recs, G, tol); mi = {a for a, _ in mm}; mag = C.match_typed(allrecs, G, tol, typed=False)
        g = np.array([[s['cx'], s['cy']] for s in G]); ext = collections.Counter(); nd = 0
        for k_, s in enumerate(recs):
            if k_ in mi: continue
            ext[s['type']] += 1; d = np.sqrt(((g - [s['cx'], s['cy']]) ** 2).sum(1)); j = int(d.argmin())
            nd += int(d[j] <= tol and G[j]['type'] == s['type'])
        dec_ok = isinstance(dec, dict)
        written = sum(abs(int(dec.get(t, 0) or 0) - gc[t]) for t in T5) if dec_ok else None
        types_seq = [s['type'] for s in recs]; runs = [t for k_, t in enumerate(types_seq) if k_ == 0 or t != types_seq[k_ - 1]]
        rows.append(dict(task=mt['task'], model=m, item=mt['item'], arm=mt['arm'], rep=mt['rep'], n=len(G), k=len(allrecs), err_code=sum(abs(lc[t] - gc[t]) for t in T5),
                         err_total=abs(len(allrecs) - len(G)), untyped=untyped, E_agn=len(allrecs) + len(G) - 2 * len(mag),
                         written_total=(abs(sum(int(dec.get(t, 0) or 0) for t in T5) - len(G)) if isinstance(dec, dict) else None),
                         err_written=written, extras=sum(ext.values()), near_dup=nd, missed=len(G) - len(mm), E=len(recs) + len(G) - 2 * len(mm),
                         grouped=int(len(runs) == len(set(runs)) and len(set(runs)) > 1), trunc=int(trunc), json_ok=int(au['json_ok']),
                         pad=[(int(dec.get(t, 0) or 0) - gc[t], ext[t]) for t in T5] if dec_ok else None, hint=mt.get('hint')))
    else:
        recs, dec, au = C.decode_shapes(r['raw_text'], mt['mode'], mt['w'], mt['h'], sw, sh, list_key='points', count_key='count')
        try: written = abs(int(dec) - mt['n']) if dec is not None else None
        except Exception: written = None
        rows.append(dict(task='pixmo', model=m, item=mt['item'], arm=mt['arm'], rep=0, n=mt['n'], k=len(recs), err_code=abs(len(recs) - mt['n']),
                         err_written=written, trunc=int(trunc), json_ok=int(au['json_ok']), hint=mt.get('hint')))
    audit['ok'] += 1


def boot(a, b, n=4000):
    a = np.asarray(a, float); b = np.asarray(b, float); rel = (b.mean() - a.mean()) / a.mean() if a.mean() else float('nan'); bs = []
    for _ in range(n):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / a[s].mean() if a[s].mean() else 0)
    return dict(a=round(float(a.mean()), 3), b=round(float(b.mean()), 3), rel=round(float(rel) * 100, 1), ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(a))


def paired(task, m, arm1, arm2, key='err_code', key2=None, avg_reps=True):
    A_ = collections.defaultdict(list); B_ = collections.defaultdict(list)
    for x in rows:
        if x['task'] != task or x['model'] != m: continue
        if x['arm'] == arm1 and x.get(key) is not None: A_[x['item']].append(x[key])
        if x['arm'] == arm2 and x.get(key2 or key) is not None: B_[x['item']].append(x[key2 or key])
    ids = sorted(set(A_) & set(B_))
    if len(ids) < 3: return None
    return boot([np.mean(A_[i]) for i in ids], [np.mean(B_[i]) for i in ids])


out = dict(audit=dict(audit), cost={k: round(v, 2) for k, v in spend.items()}, cost_total=round(sum(spend.values()), 2), scp={}, scpd={}, pixmo={})
MODELS_PRESENT = [mm_ for mm_ in C.MODELS + ['qwen3vl'] if any(x['model'] == mm_ for x in rows)]
for TK, m in [(t, mm_) for t in ('scp', 'scpd') for mm_ in MODELS_PRESENT]:
    S = [x for x in rows if x['task'] == TK and x['model'] == m]
    if not S: continue
    cell = {}
    for arm in sorted({x['arm'] for x in S}):
        X = [x for x in S if x['arm'] == arm]
        cell[arm] = {k: round(float(np.mean([x[k] for x in X if x.get(k) is not None])), 3) if any(x.get(k) is not None for x in X) else None
                     for k in ('err_code', 'err_written', 'err_total', 'written_total', 'untyped', 'E_agn', 'extras', 'near_dup', 'missed', 'E', 'grouped', 'trunc', 'json_ok')}
        cell[arm]['n_rows'] = len(X)
        P = [p for x in X if x.get('pad') for p in x['pad']]
        if P:
            xs = np.array([p[0] for p in P]); ys = np.array([p[1] for p in P])
            cell[arm]['pad_slope'] = round(float(np.polyfit(xs, ys, 1)[0]), 3) if len(set(xs)) > 1 else None
    # commitment weight: list length response to the supplied number
    K = [x for x in S if x['arm'] in ('k1', 'k2')]
    if len(K) > 5:
        xs = np.array([x['hint'] - x['n'] for x in K], float); ys = np.array([x['k'] - x['n'] for x in K], float)
        b = np.polyfit(xs, ys, 1)[0]; bs = []
        for _ in range(2000):
            s = rng.integers(0, len(xs), len(xs))
            if len(set(xs[s])) > 1: bs.append(np.polyfit(xs[s], ys[s], 1)[0])
        cell['commitment_weight'] = dict(slope=round(float(b), 3), ci=[round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], n=len(K))
    comps = {'written_vs_code_a': paired(TK, m, 'a', 'a', 'err_written', 'err_code'), 'b_vs_a': paired(TK, m, 'a', 'b'), 'c_vs_b': paired(TK, m, 'b', 'c'),
             'd_vs_b': paired(TK, m, 'b', 'd'), 'd_vs_a': paired(TK, m, 'a', 'd'), 'e_vs_d': paired(TK, m, 'd', 'e'), 'e_vs_d_missed': paired(TK, m, 'd', 'e', 'missed'),
             'g_vs_f': paired(TK, m, 'f', 'g'), 'g_vs_h': paired(TK, m, 'h', 'g'), 'f_vs_e': paired(TK, m, 'e', 'f'), 'g_vs_e': paired(TK, m, 'e', 'g'),
             'i_vs_j': paired(TK, m, 'j', 'i'), 'i_vs_d': paired(TK, m, 'd', 'i'), 'l_vs_d': paired(TK, m, 'd', 'l'),
             'written_c_vs_code_c': paired(TK, m, 'c', 'c', 'err_written', 'err_code'), 'E_d_vs_a': paired(TK, m, 'a', 'd', 'E')}
    comps['total_d_vs_a'] = paired(TK, m, 'a', 'd', 'err_total'); comps['total_i_vs_d'] = paired(TK, m, 'd', 'i', 'err_total'); comps['written_total_vs_code_total_a'] = paired(TK, m, 'a', 'a', 'written_total', 'err_total')
    cell['comparisons'] = {k: v for k, v in comps.items() if v}
    out[TK][m] = cell
    if TK != 'scp': continue
    Px = [x for x in rows if x['task'] == 'pixmo' and x['model'] == m]
    if Px:
        pc = {arm: {k: round(float(np.mean([x[k] for x in Px if x['arm'] == arm and x.get(k) is not None])), 3) if any(x['arm'] == arm and x.get(k) is not None for x in Px) else None
                    for k in ('err_code', 'err_written', 'trunc', 'json_ok')} for arm in sorted({x['arm'] for x in Px})}
        K = [x for x in Px if x['arm'] in ('k1', 'k2')]
        if len(K) > 5:
            xs = np.array([x['hint'] - x['n'] for x in K], float); ys = np.array([x['k'] - x['n'] for x in K], float)
            pc['commitment_weight'] = dict(slope=round(float(np.polyfit(xs, ys, 1)[0]), 3) if len(set(xs)) > 1 else None, n=len(K))
        pc['comparisons'] = {k: v for k, v in {'written_vs_code_a': paired('pixmo', m, 'a', 'a', 'err_written', 'err_code'), 'c_vs_a': paired('pixmo', m, 'a', 'c'),
                                               'd_vs_a': paired('pixmo', m, 'a', 'd'), 'i_vs_d': paired('pixmo', m, 'd', 'i')}.items() if v}
        out['pixmo'][m] = pc
json.dump(dict(summary=out, rows=rows), open(f'{C.ROOT}/runs/{SUB}/b1_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:6000])
