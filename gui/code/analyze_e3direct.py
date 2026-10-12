"""E3 Direct mini test analysis (no model calls; labels used only in scoring). Per dataset x model on the 20 selected images:
draft (L-min rep 0), two-call intersection (rep 0, rep 1), three-call majority, FSR-Direct A3 / A4 (committed) and their
no-verification ablations; primary E3 metric (points: tau; boxes: IoU 0.75) and IoU 0.5; costs; D3 quantities."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrgui as G
import e3direct as X
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import e3lib as L
import analyze_e3 as A0
import offline_agg2                      # installs the lexicographic correspondence
import aggregate_v2 as AG
from score_e3 import PRICE
OUT = '/home/claude/gui/runs/e3direct'; MODELS = X.MODELS
items, sel = X.pick(); D = X.drafts()
rows_e3 = A0.load_rows('runs_main/scored.jsonl')
tok = {(x['item'], x['model'], x['rep']): (x['in_tok'], x['out_tok']) for x in rows_e3 if x['cand'] == 'L-min' and x['block'] in ('MAIN', 'AGG')}


def load(stage):
    Q, R = {}, {}
    f = f'{OUT}/requests_{stage}.jsonl'
    if os.path.exists(f):
        for l in open(f): q = json.loads(l); Q[q['request_id']] = q
    for m in MODELS:
        f = f'{OUT}/responses_{stage}_{m}.jsonl'
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['status'] == 'completed': R[r['request_id']] = r
    return Q, R


QB, RB = load('B'); QC, RC = load('C')
def cost(m, r): return (r['usage']['input_tokens'] * PRICE[m][0] + r['usage']['output_tokens'] * PRICE[m][1]) / 1e6 if r else 0.0
def ecost(m, key): t = tok.get(key); return (t[0] * PRICE[m][0] + t[1] * PRICE[m][1]) / 1e6 if t else 0.0


rows = []; d3 = []
for m in MODELS:
    for it in sel:
        reps = [D.get((it['id'], m, r)) for r in (0, 1, 2)]
        if any(r is None for r in reps): continue
        mt = A0.metric_of(it, A0.primary(it)); m50 = A0.metric_of(it, 'iou50') if it['task'] == 'box' else mt
        def E(recs, met=mt): return L.error(it, recs, met)[0]
        cl2 = AG.build_clusters(it, reps[:2]); cl3 = AG.build_clusters(it, reps)
        out = {'draft': (reps[0], ecost(m, (it['id'], m, 0))),
               'inter2': ([AG.fuse(x[1]) for x in cl2 if len(x[2]) == 2], ecost(m, (it['id'], m, 0)) + ecost(m, (it['id'], m, 1))),
               'maj3': ([AG.fuse(x[1]) for x in cl3 if len(x[2]) >= 2], sum(ecost(m, (it['id'], m, r)) for r in (0, 1, 2)))}
        for arm, crop in (('A3', False), ('A4', True)):
            ra = RB.get(f"{it['id']}|{m}|{arm}audit"); rv = RC.get(f"{it['id']}|{m}|{arm}verify")
            if ra is None: continue
            edits, au = X.decode_edits(it, ra['raw_text'], len(reps[0]), crop)
            dec = G.decode_decisions(rv['raw_text'], len(edits))[0] if (edits and rv is not None) else ['keep'] * len(edits)
            fin, led = G.commit(reps[0], edits, dec); finn, _ = G.commit(reps[0], edits, ['apply'] * len(edits))
            c0 = ecost(m, (it['id'], m, 0))
            out[arm] = (fin, c0 + cost(m, ra) + cost(m, rv)); out[arm + 'n'] = (finn, c0 + cost(m, ra))
            base = E(reps[0])
            for e, d, lg in zip(edits, dec, led):
                d3.append(dict(model=m, ds=it['dataset'], arm=arm, op=e['op'], g=base - E(G.apply_one(reps[0], e)), accepted=int(lg['status'] == 'committed')))
            d3.append(dict(model=m, ds=it['dataset'], arm=arm, op='_total', g_actual=base - E(fin), g_noverify=base - E(finn), n_edits=len(edits),
                           proposed=au['proposed'], invalid=au['invalid'], lenient=au.get('lenient', 0)))
        for arm, (recs, c) in out.items():
            rows.append(dict(model=m, ds=it['dataset'], item=it['id'], arm=arm, E=E(recs), E50=E(recs, m50), n=it['n'], cost=c))
summ = {}
ARMS = ['draft', 'inter2', 'maj3', 'A3', 'A3n', 'A4', 'A4n']
for ds in ('sku110k', 'dota', 'fsc147'):
    for m in MODELS:
        R = [r for r in rows if r['ds'] == ds and r['model'] == m]
        ids = sorted({r['item'] for r in R}); by = {a: {r['item']: r for r in R if r['arm'] == a} for a in ARMS}
        common = [i for i in ids if all(i in by[a] for a in ARMS)]
        b = np.mean([by['draft'][i]['E'] for i in common]); b50 = np.mean([by['draft'][i]['E50'] for i in common]); bc = np.mean([by['draft'][i]['cost'] for i in common])
        cell = dict(n_images=len(common), draft_E=round(float(b), 2), mean_n=round(float(np.mean([by['draft'][i]['n'] for i in common])), 1))
        for a in ARMS:
            e = np.mean([by[a][i]['E'] for i in common]); e50 = np.mean([by[a][i]['E50'] for i in common])
            cell[a] = dict(E=round(float(e), 2), rel=round(float((b - e) / b), 4), rel50=round(float((b50 - e50) / b50), 4),
                           cost_ratio=round(float(np.mean([by[a][i]['cost'] for i in common]) / bc), 2),
                           worse_share=round(float(np.mean([by[a][i]['E'] > by['draft'][i]['E'] for i in common])), 3))
        for arm in ('A3', 'A4'):
            E_ = [d for d in d3 if d['ds'] == ds and d['model'] == m and d['arm'] == arm and d['op'] != '_total']
            T_ = [d for d in d3 if d['ds'] == ds and d['model'] == m and d['arm'] == arm and d['op'] == '_total']; n = max(1, len(T_))
            Gp = sum(max(0, d['g']) for d in E_); Gm = sum(max(0, -d['g']) for d in E_)
            cell[arm + '_d3'] = dict(edits_per_image=round(len(E_) / n, 2), G_plus=round(Gp / n, 3), G_minus=round(Gm / n, 3),
                                     h_plus=round(sum(d['accepted'] * max(0, d['g']) for d in E_) / Gp, 3) if Gp else None,
                                     h_minus=round(sum(d['accepted'] * max(0, -d['g']) for d in E_) / Gm, 3) if Gm else None,
                                     actual=round(sum(d['g_actual'] for d in T_) / n, 3), noverify=round(sum(d['g_noverify'] for d in T_) / n, 3),
                                     kappa=round((sum(d['accepted'] * d['g'] for d in E_) - sum(d['g_actual'] for d in T_)) / n, 3),
                                     invalid_share=round(sum(d['invalid'] for d in T_) / max(1, sum(d['proposed'] for d in T_)), 3),
                                     ops=dict(collections.Counter(d['op'] for d in E_)))
        summ[f'{ds}|{m}'] = cell
json.dump(dict(summary=summ, rows=rows, d3=d3), open(f'{OUT}/e3direct_analysis.json', 'w'), indent=1)
for k, v in summ.items():
    print(k, 'draft E', v['draft_E'], 'n', v['mean_n'], {a: (v[a]['rel'], v[a]['cost_ratio']) for a in ARMS})
    print('   ', 'A3', v.get('A3_d3')); print('   ', 'A4', v.get('A4_d3'))
