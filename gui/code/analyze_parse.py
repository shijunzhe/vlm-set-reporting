"""Stage 3 analysis: screen parsing (no model calls; labels used only in scoring).
Arms (per model x screen):
  A1 single draft D1 | A2i / A2u: D1,D2 intersection / union | A2m: majority of D1..D3 |
  A3 / A4: FSR-Direct (same-view / crop-view) committed; A3n / A4n: all valid proposals applied without verification |
  A5s / A5i: tiles T1 / T1 intersect T2.
Outputs runs/main/parse_analysis.json: per arm and cell (model x source, and model x group) mean E, F1, FP, FN, centre-hit
metrics, cost; D3 quantities for A3/A4 (G+, G-, h+, h-, interaction); stable error shares from D1..D3; one-bit p1; tiered
use (anchored / plain) and platform presets.
"""
import json, os, sys, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrgui as G
sys.path.insert(0, '/home/claude/fsr_v51/e3'); from score_e3 import PRICE
os.chdir(G.ROOT)
SUB = sys.argv[1] if len(sys.argv) > 1 else 'main'
MODELS = ['gpt61', 'sonnet55', 'gem38']
ARMS = ['A1', 'A2i', 'A2u', 'A2m', 'A3', 'A3n', 'A4', 'A4n', 'A5s', 'A5i']
P = {i['id']: i for i in json.load(open('data/parse_items.json'))}


def load(stage):
    Q = {}; R = {}
    f = f'runs/{SUB}/requests_{stage}.jsonl'
    if os.path.exists(f):
        for l in open(f): q = json.loads(l); Q[q['request_id']] = q
    for m in MODELS:
        f = f'runs/{SUB}/responses_{stage}_{m}.jsonl'
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['status'] == 'completed': R[r['request_id']] = r
    return Q, R


QA, RA = load('A'); QB, RB = load('B'); QC, RC = load('C')
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
def cost(m, r): return (r['usage']['input_tokens'] * PRICE[m][0] + r['usage']['output_tokens'] * PRICE[m][1]) / 1e6 if r else 0.0


rows = []; d3 = []; stab = []; onebit = []
for m in MODELS:
    for iid, it in P.items():
        gt = it['targets']; k = lambda s: f'{iid}|{m}|{s}'
        if not all(k(f'D{j}') in RA for j in (1, 2, 3)): continue
        D = [G.decode_list(RA[k(f'D{j}')]['raw_text'], sc(QA[k(f'D{j}')])[0])[0] for j in (1, 2, 3)]
        c = {f'D{j}': cost(m, RA[k(f'D{j}')]) for j in (1, 2, 3)}
        out = {'A1': ([b for b, n in D[0]], c['D1']), 'A2i': (G.construct(D[:2], 'inter'), c['D1'] + c['D2']),
               'A2u': (G.construct(D[:2], 'union'), c['D1'] + c['D2']), 'A2m': (G.construct(D, 'majority'), c['D1'] + c['D2'] + c['D3'])}
        for arm, crop in (('A3', False), ('A4', True)):
            ra = RB.get(k(f'{arm}audit')); rv = RC.get(k(f'{arm}verify'))
            if ra is None: continue
            s = sc(QB[k(f'{arm}audit')])
            edits, au = G.decode_edits(ra['raw_text'], len(D[0]), crop, s[0], s[1:] if crop and s[0] else None)
            if edits and rv is not None: dec, _ = G.decode_decisions(rv['raw_text'], len(edits))
            else: dec = ['keep'] * len(edits)
            fin, led = G.commit(D[0], edits, dec); finn, _ = G.commit(D[0], edits, ['apply'] * len(edits))
            out[arm] = ([b for b, n in fin], c['D1'] + cost(m, ra) + cost(m, rv)); out[arm + 'n'] = ([b for b, n in finn], c['D1'] + cost(m, ra))
            base = G.score_boxes([b for b, n in D[0]], gt)['E']
            for e, d, lg in zip(edits, dec, led):
                g = base - G.score_boxes([b for b, n in G.apply_one(D[0], e)], gt)['E']
                d3.append(dict(model=m, item=iid, source=it['source'], arm=arm, op=e['op'], n_new=len(e['new']), n_ids=len(e['ids']), g=g,
                               accepted=int(lg['status'] == 'committed'), decision=d))
            d3.append(dict(model=m, item=iid, source=it['source'], arm=arm, op='_total', g_actual=base - G.score_boxes(out[arm][0], gt)['E'],
                           g_actual_noverify=base - G.score_boxes(out[arm + 'n'][0], gt)['E'], n_edits=len(edits), proposed=au['proposed'], invalid=au['invalid']))
        if k('T1') in RA:
            T = [G.decode_tiles(RA[k(f'T{j}')]['raw_text'], sc(QA[k(f'T{j}')]) if sc(QA[k(f'T{j}')])[0] else None)[0] for j in (1, 2) if k(f'T{j}') in RA]
            out['A5s'] = ([b for b, n in T[0]], cost(m, RA[k('T1')]))
            if len(T) == 2: out['A5i'] = (G.construct(T, 'inter'), cost(m, RA[k('T1')]) + cost(m, RA[k('T2')]))
        for arm, (S, cst) in out.items():
            s = G.score_boxes(S, gt); rows.append(dict(model=m, item=iid, source=it['source'], group=it['group'], arm=arm, cost=cst, **s))
        # stable shares from D1..D3 (per-draft matching for targets; label-free clusters for false reports)
        sup = np.zeros(len(gt))
        for Dj in D:
            for i, j in G.lex_match([b for b, n in Dj], gt, 0.5): sup[j] += 1
        fn = float(np.sum((3 - sup) / 3)); fn_st = float(np.sum(sup == 0))
        cl = G.clusters(D); fused = [x['fused'] for x in cl]; mt = {i for i, j in G.lex_match(fused, gt, 0.5)}
        fp = sum(len(x['sup']) / 3 for q, x in enumerate(cl) if q not in mt); fp_st = sum(1 for q, x in enumerate(cl) if q not in mt and len(x['sup']) == 3)
        stab.append(dict(model=m, item=iid, source=it['source'], fn=fn, fn_st=fn_st, fp=fp, fp_st=fp_st))
        # one-bit: among D1 records not confirmed by D2, share that are true (matched in D1's own matching)
        pairs = G.lex_match([b for b, n in D[0]], [b for b, n in D[1]], 0.5); conf = {i for i, j in pairs}
        own = {i for i, j in G.lex_match([b for b, n in D[0]], gt, 0.5)}
        unc = [i for i in range(len(D[0])) if i not in conf]
        onebit.append(dict(model=m, item=iid, source=it['source'], unconf=len(unc), unconf_true=sum(i in own for i in unc)))

json.dump(dict(rows=rows, d3=d3, stab=stab, onebit=onebit), open(f'runs/{SUB}/parse_rows.json', 'w'))
print('rows', len(rows), 'd3', len(d3))

# ---------------- summaries
def mean(v): return float(np.mean(v)) if len(v) else float('nan')
summ = {}
for m in MODELS:
    for src in ('groundcua', 'screenparse'):
        R = [r for r in rows if r['model'] == m and r['source'] == src]; items = sorted({r['item'] for r in R})
        by = {a: {r['item']: r for r in R if r['arm'] == a} for a in ARMS}
        common = [i for i in items if all(i in by[a] for a in ARMS)]
        base = mean([by['A1'][i]['E'] for i in common]); baseF = mean([by['A1'][i]['f1'] for i in common]); bc = mean([by['A1'][i]['cost'] for i in common])
        cell = dict(n_screens=len(common), A1_E=round(base, 2), A1_F1=round(baseF, 4), A1_cost=round(bc, 4), mean_n=round(mean([by['A1'][i]['n'] for i in common]), 1))
        for a in ARMS:
            E = mean([by[a][i]['E'] for i in common]); F = mean([by[a][i]['f1'] for i in common]); Ec = mean([by[a][i]['E_c'] for i in common])
            cell[a] = dict(E=round(E, 2), rel=round((base - E) / base, 4), F1=round(F, 4), dF1=round(100 * (F - baseF), 2),
                           FP=round(mean([by[a][i]['fp'] for i in common]), 2), FN=round(mean([by[a][i]['fn'] for i in common]), 2),
                           E_c=round(Ec, 2), rel_c=round((mean([by['A1'][i]['E_c'] for i in common]) - Ec) / mean([by['A1'][i]['E_c'] for i in common]), 4),
                           cost_ratio=round(mean([by[a][i]['cost'] for i in common]) / bc, 2),
                           worse_share=round(mean([by[a][i]['E'] > by['A1'][i]['E'] for i in common]), 3))
        summ[f'{src}|{m}'] = cell
# D3 quantities
d3s = {}
for m in MODELS:
    for src in ('groundcua', 'screenparse'):
        for arm in ('A3', 'A4'):
            E_ = [d for d in d3 if d['model'] == m and d['source'] == src and d['arm'] == arm and d['op'] != '_total']
            T_ = [d for d in d3 if d['model'] == m and d['source'] == src and d['arm'] == arm and d['op'] == '_total']
            n = len(T_) or 1
            Gp = sum(max(0, d['g']) for d in E_) / n; Gm = sum(max(0, -d['g']) for d in E_) / n
            hp = sum(d['accepted'] * max(0, d['g']) for d in E_) / max(1e-9, sum(max(0, d['g']) for d in E_)) if Gp > 0 else None
            hm = sum(d['accepted'] * max(0, -d['g']) for d in E_) / max(1e-9, sum(max(0, -d['g']) for d in E_)) if Gm > 0 else None
            pred = sum(d['accepted'] * d['g'] for d in E_) / n; act = sum(d['g_actual'] for d in T_) / n
            d3s[f'{src}|{m}|{arm}'] = dict(edits_per_screen=round(len(E_) / n, 2), G_plus=round(Gp, 3), G_minus=round(Gm, 3),
                                           h_plus=None if hp is None else round(hp, 3), h_minus=None if hm is None else round(hm, 3),
                                           sum_accepted_g=round(pred, 3), actual_gain=round(act, 3), kappa=round(pred - act, 3),
                                           noverify_gain=round(sum(d['g_actual_noverify'] for d in T_) / n, 3),
                                           ops=dict(collections.Counter(d['op'] for d in E_)),
                                           add_records=round(sum(d['n_new'] for d in E_ if d['op'] == 'add') / n, 2))
st = {}
for m in MODELS:
    for src in ('groundcua', 'screenparse'):
        S = [s for s in stab if s['model'] == m and s['source'] == src]
        fn = sum(s['fn'] for s in S); fp = sum(s['fp'] for s in S)
        st[f'{src}|{m}'] = dict(single_FN=round(fn / len(S), 2), stable_FN_share=round(sum(s['fn_st'] for s in S) / fn, 3) if fn else None,
                                single_FP=round(fp / len(S), 2), stable_FP_share=round(sum(s['fp_st'] for s in S) / fp, 3) if fp else None)
        O = [o for o in onebit if o['model'] == m and o['source'] == src]
        st[f'{src}|{m}']['p1'] = round(sum(o['unconf_true'] for o in O) / max(1, sum(o['unconf'] for o in O)), 3)
        st[f'{src}|{m}']['unconfirmed_per_screen'] = round(sum(o['unconf'] for o in O) / len(O), 2)

# tiers and presets
TIER_OPTS = ['A1', 'A2i', 'A2u', 'A2m', 'A3', 'A4', 'A5s', 'A5i']
def Emap(m, src, a): return {r['item']: r['E'] for r in rows if r['model'] == m and r['source'] == src and r['arm'] == a}
tiers = {}; presets = {}
for m in MODELS:
    for src in ('groundcua', 'screenparse'):
        M = {a: Emap(m, src, a) for a in TIER_OPTS}; ids = sorted(set.intersection(*[set(v) for v in M.values()]))
        grp = {i: P[i]['group'] for i in ids}; res = {}
        for L in (0, 1, 2, 3, 5, 10, 20):
            rng = random.Random(f'gt:{m}:{src}:{L}'); ga, gp = [], []
            for _ in range(400):
                lab = rng.sample(ids, L); ev = [i for i in ids if i not in lab]; base = mean([M['A1'][i] for i in ev])
                pick = 'A2i'
                if L >= 2:
                    best, bv = None, 0.0
                    for o in TIER_OPTS:
                        d = np.array([M['A2i'][i] - M[o][i] for i in lab], float); sd = d.std(ddof=1)
                        lb = d.mean() - (sd / np.sqrt(L) if sd > 0 else 0)
                        if lb > 0 and d.mean() > bv: best, bv = o, d.mean()
                    if best: pick = best
                elif L == 1:
                    o = min(['A1', 'A2i', 'A2u'], key=lambda o: M[o][lab[0]]); pick = o if M[o][lab[0]] < M['A2i'][lab[0]] else 'A2i'
                ga.append((base - mean([M[pick][i] for i in ev])) / base)
                if L >= 1:
                    pp = min(TIER_OPTS, key=lambda o: mean([M[o][i] for i in lab])); gp.append((base - mean([M[pp][i] for i in ev])) / base)
            res[L] = dict(anchored=round(mean(ga), 4), plain=round(mean(gp), 4) if gp else None)
        orc = min(TIER_OPTS, key=lambda o: mean([M[o][i] for i in ids])); base = mean([M['A1'][i] for i in ids])
        res['oracle'] = [orc, round((base - mean([M[orc][i] for i in ids])) / base, 4)]
        tiers[f'{src}|{m}'] = res
        # presets across groups (same source, same model): choose on the other groups, score on this group
        for g in sorted(set(grp.values())):
            tg = [i for i in ids if grp[i] == g]; og = [i for i in ids if grp[i] != g]
            rel = lambda o, S: (mean([M['A1'][i] for i in S]) - mean([M[o][i] for i in S])) / mean([M['A1'][i] for i in S])
            pre = max(TIER_OPTS, key=lambda o: rel(o, og)); best = max(TIER_OPTS, key=lambda o: rel(o, tg))
            eps = max(abs(rel(o, tg) - rel(o, og)) for o in TIER_OPTS)
            presets[f'{src}|{m}|{g}'] = dict(preset=pre, preset_gain=round(rel(pre, tg), 4), best=best, best_gain=round(rel(best, tg), 4),
                                             zero_label_default_gain=round(rel('A2i', tg), 4), eps_rel=round(eps, 4))
    # cross-source presets (choose on the other source, same model)
    for src, oth in (('groundcua', 'screenparse'), ('screenparse', 'groundcua')):
        Ms = {a: Emap(m, src, a) for a in TIER_OPTS}; Mo = {a: Emap(m, oth, a) for a in TIER_OPTS}
        ids_s = sorted(set.intersection(*[set(v) for v in Ms.values()])); ids_o = sorted(set.intersection(*[set(v) for v in Mo.values()]))
        rs = lambda o: (mean([Ms['A1'][i] for i in ids_s]) - mean([Ms[o][i] for i in ids_s])) / mean([Ms['A1'][i] for i in ids_s])
        ro = lambda o: (mean([Mo['A1'][i] for i in ids_o]) - mean([Mo[o][i] for i in ids_o])) / mean([Mo['A1'][i] for i in ids_o])
        pre = max(TIER_OPTS, key=ro)
        presets[f'{src}|{m}|cross_source'] = dict(preset=pre, preset_gain=round(rs(pre), 4), best=max(TIER_OPTS, key=rs), best_gain=round(rs(max(TIER_OPTS, key=rs)), 4),
                                                  eps_rel=round(max(abs(rs(o) - ro(o)) for o in TIER_OPTS), 4))
json.dump(dict(summary=summ, d3=d3s, stable=st, tiers=tiers, presets=presets), open(f'runs/{SUB}/parse_analysis.json', 'w'), indent=1)
for k, v in summ.items():
    print(k, 'A1 E', v['A1_E'], 'n', v['mean_n'], {a: (v[a]['rel'], v[a]['dF1'], v[a]['cost_ratio']) for a in ARMS})
for k, v in d3s.items(): print(k, v)
for k, v in st.items(): print(k, v)
for k, v in tiers.items(): print(k, v)
for k, v in presets.items(): print(k, v)
