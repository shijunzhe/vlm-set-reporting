"""Check the same-view audit bound h+ - h- <= TV(phi | good edit, phi | bad edit) on the GUI Direct data (A3 same-view audit,
A4 crop-view audit; verify step decides apply / keep). phi = label-free features available without the verifier:
operation, support of the involved record(s) in the other two drafts D2 and D3 (0, 1, 2), and size tercile.
h+ = share of good edits (gain > 0) accepted, h- = share of bad edits (gain < 0) accepted.
TV is estimated two ways: plug-in over the phi cells (biased upward) and a split-half estimate of the best phi-cell rule
(about a lower bound). No model calls. Usage: python -I audit_bound.py"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, '/home/claude/gui/code')
import fsrgui as G
os.chdir(G.ROOT)
MODELS = ['gpt61', 'sonnet55', 'gem38']
P = {i['id']: i for i in json.load(open('data/parse_items.json'))}
def load(stage):
    Q, R = {}, {}
    for l in open(f'runs/main/requests_{stage}.jsonl'): q = json.loads(l); Q[q['request_id']] = q
    for m in MODELS:
        for l in open(f'runs/main/responses_{stage}_{m}.jsonl'):
            r = json.loads(l)
            if r['status'] == 'completed': R[r['request_id']] = r
    return Q, R
QA, RA = load('A'); QB, RB = load('B'); QC, RC = load('C')
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
E = []
for m in MODELS:
    for iid, it in P.items():
        gt = it['targets']; k = lambda s: f'{iid}|{m}|{s}'
        if not all(k(f'D{j}') in RA for j in (1, 2, 3)): continue
        D = [G.decode_list(RA[k(f'D{j}')]['raw_text'], sc(QA[k(f'D{j}')])[0])[0] for j in (1, 2, 3)]
        others = [[b for b, n in D[1]], [b for b, n in D[2]]]
        def support(box): return sum(1 for O in others if O and max(G.iou(box, o) for o in O) >= 0.5)
        base = G.score_boxes([b for b, n in D[0]], gt)['E']
        for arm, crop in (('A3', False), ('A4', True)):
            ra = RB.get(k(f'{arm}audit')); rv = RC.get(k(f'{arm}verify'))
            if ra is None or rv is None: continue
            s = sc(QB[k(f'{arm}audit')])
            edits, _ = G.decode_edits(ra['raw_text'], len(D[0]), crop, s[0], s[1:] if crop and s[0] else None)
            if not edits: continue
            dec, _ = G.decode_decisions(rv['raw_text'], len(edits))
            for e, d in zip(edits, dec):
                g = base - G.score_boxes([b for b, n in G.apply_one(D[0], e)], gt)['E']
                if g == 0: continue
                boxes = [D[0][i - 1][0] for i in e['ids']] if e['ids'] else [b for b, n in e['new']]
                sup = int(round(np.mean([support(b) for b in boxes]))) if e['op'] != 'add' else max(support(b) for b in boxes)
                area = float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
                E.append(dict(model=m, source=it['source'], arm=arm, op=e['op'] if e['op'] in ('add', 'delete', 'replace') else 'other',
                              sup=sup, area=area, good=int(g > 0), acc=int(d == 'apply')))
rng = np.random.default_rng(0)
def cells(X, q):
    return [f"{x['op']}|{x['sup']}|{int(np.digitize(x['area'], q))}" for x in X]
def analyse(X):
    if sum(x['good'] for x in X) < 10 or sum(1 - x['good'] for x in X) < 10: return None
    q = np.quantile([x['area'] for x in X], [1 / 3, 2 / 3]); c = np.array(cells(X, q)); gd = np.array([x['good'] for x in X]); ac = np.array([x['acc'] for x in X])
    hp, hm = ac[gd == 1].mean(), ac[gd == 0].mean()
    keys = sorted(set(c)); pg = {k: np.mean(c[gd == 1] == k) for k in keys}; pb = {k: np.mean(c[gd == 0] == k) for k in keys}
    tv_plugin = 0.5 * sum(abs(pg[k] - pb[k]) for k in keys)
    tv_split = []
    idx = np.arange(len(X))
    for _ in range(200):
        msk = rng.random(len(X)) < 0.5; a, b = idx[msk], idx[~msk]
        if gd[a].sum() < 3 or (1 - gd[a]).sum() < 3 or gd[b].sum() < 3 or (1 - gd[b]).sum() < 3: continue
        A = {k for k in keys if np.mean(c[a][gd[a] == 1] == k) > np.mean(c[a][gd[a] == 0] == k)}
        inA = np.array([x in A for x in c[b]]); tv_split.append(inA[gd[b] == 1].mean() - inA[gd[b] == 0].mean())
    return dict(n_good=int(gd.sum()), n_bad=int((1 - gd).sum()), h_plus=round(float(hp), 3), h_minus=round(float(hm), 3), h_gap=round(float(hp - hm), 3),
                tv_plugin=round(float(tv_plugin), 3), tv_split=round(float(np.mean(tv_split)), 3), bound_holds_vs_split=bool(hp - hm <= np.mean(tv_split) + 1e-9))
out = {}
for m in MODELS:
    for arm in ('A3', 'A4'):
        r = analyse([x for x in E if x['model'] == m and x['arm'] == arm])
        if r: out[f'{m}|{arm}'] = r
out['pooled'] = analyse(E)
json.dump(out, open('/home/claude/stageB/offline/audit_bound.json', 'w'), indent=1)
for k_, v in out.items(): print(k_, v)
