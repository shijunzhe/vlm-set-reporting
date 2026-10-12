"""Cross-model fusion on GUI screen parsing (EXP-GUI stage 3: GroundCUA 60 and ScreenParse 60 screens; drafts D1..D3 per model;
boxes in normalized coordinates, class-agnostic, IoU 0.5 as in the GUI analysis). No model calls; labels only for scoring.
Constructions: same model (D1; D1 with D2 intersection / union; D1..D3 unanimous / majority / union) vs cross model (D1 of A with D1 of B,
intersection / union; D1 of all three: unanimous / majority / union). Diagnostics: THM-INTERSECT sign check (p1 of A's D1 records not
confirmed by the partner) and partner discrimination; few-label menu with and without cross options.
usage: python -I xmodel_parse.py"""
import json, os, sys, random
import numpy as np
sys.path.insert(0, '/home/claude/gui/code')
import fsrgui as G
os.chdir(G.ROOT)
M = ['gpt61', 'sonnet55', 'gem38']; rng = np.random.default_rng(0)
P = {i['id']: i for i in json.load(open('data/parse_items.json'))}
Q, R = {}, {}
for l in open('runs/main/requests_A.jsonl'): q = json.loads(l); Q[q['request_id']] = q
for m in M:
    for l in open(f'runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': R[r['request_id']] = r
sc = lambda q: [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
def draft(iid, m, j):
    k = f'{iid}|{m}|D{j}'
    return G.decode_list(R[k]['raw_text'], sc(Q[k])[0])[0] if k in R else None
def E(boxes, gt): return G.score_boxes(boxes, gt)['E']
def keep(reports, need):
    return [c['fused'] for c in G.clusters(reports) if len(c['sup']) >= need]
out = {}; per_image = {}
for src in ('groundcua', 'screenparse'):
    Ei = {}; diag = {}
    ids = []
    for iid, it in P.items():
        if it['source'] != src: continue
        D = {m: [draft(iid, m, j) for j in (1, 2, 3)] for m in M}
        if any(d is None for v in D.values() for d in v): continue
        ids.append(iid); gt = it['targets']
        def put(k, v): Ei.setdefault(k, {})[iid] = v
        for m in M:
            d = D[m]; put(f'single|{m}', E([b for b, n in d[0]], gt))
            put(f'same2_inter|{m}', E(keep(d[:2], 2), gt)); put(f'same2_union|{m}', E(keep(d[:2], 1), gt))
            put(f'same3_unan|{m}', E(keep(d, 3), gt)); put(f'same3_maj|{m}', E(keep(d, 2), gt)); put(f'same3_union|{m}', E(keep(d, 1), gt))
        for a in M:
            for b in M:
                part = D[a][1] if a == b else D[b][0]
                if a != b:
                    put(f'cross2_inter|{a}|{b}', E(keep([D[a][0], part], 2), gt)); put(f'cross2_union|{a}|{b}', E(keep([D[a][0], part], 1), gt))
                cl = [c for c in G.clusters([D[a][0], part]) if 0 in c['sup']]
                recs = [c['members'][0] if 'members' in c else c['fused'] for c in cl]
                mt = {i for i, j in G.lex_match(recs, gt, 0.5)} if recs and gt else set()
                t = np.array([i in mt for i in range(len(recs))], dtype=bool); cf = np.array([len(c['sup']) == 2 for c in cl], dtype=bool)
                dd = diag.setdefault(f'{a}|{"same" if a == b else b}', dict(T=0, F=0, Tc=0, Fc=0))
                dd['T'] += int(t.sum()); dd['F'] += int((~t).sum()); dd['Tc'] += int((t & cf).sum()); dd['Fc'] += int((~t & cf).sum())
        three = [D[m][0] for m in M]
        put('cross3_unan', E(keep(three, 3), gt)); put('cross3_maj', E(keep(three, 2), gt)); put('cross3_union', E(keep(three, 1), gt))
    mean = {k: float(np.mean([v[i] for i in ids])) for k, v in Ei.items()}
    def ci(a, b):
        x = np.array([Ei[a][i] for i in ids], float); y = np.array([Ei[b][i] for i in ids], float); bs = []
        for _ in range(4000):
            s = rng.integers(0, len(x), len(x)); bs.append((y[s].mean() - x[s].mean()) / x[s].mean())
        return dict(a=a, b=b, va=round(float(x.mean()), 2), vb=round(float(y.mean()), 2), rel=round(float((y.mean() - x.mean()) / x.mean()) * 100, 1),
                    ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)])
    same = [f'{k}|{m}' for m in M for k in ('single', 'same2_inter', 'same2_union', 'same3_unan', 'same3_maj', 'same3_union')]
    cross = [k for k in Ei if k.startswith('cross')]
    bs_, bc = min(same, key=mean.get), min(cross, key=mean.get)
    Ei['avg_same3_unan'] = {i: float(np.mean([Ei[f'same3_unan|{m}'][i] for m in M])) for i in ids}
    sign = []
    for k, d in diag.items():
        a, p = k.split('|'); Tu = d['T'] - d['Tc']; Fu = d['F'] - d['Fc']; p1 = Tu / max(1, Tu + Fu)
        E2 = mean[f'same2_inter|{a}'] if p == 'same' else mean[f'cross2_inter|{a}|{p}']
        gain = mean[f'single|{a}'] - E2
        sign.append(dict(pair=k, p1=round(p1, 3), gain=round(gain, 2), agree=bool((gain > 0) == (p1 < 0.5)),
                         delta=round(d['Tc'] / max(1, d['T']) - d['Fc'] / max(1, d['F']), 3)))
    dsame = {s['pair'].split('|')[0]: s['delta'] for s in sign if s['pair'].endswith('same')}
    # few-label menu
    menu = {}
    for L in (5, 10):
        r = random.Random(L); rel = []; pick = 0
        for _ in range(200):
            lab = r.sample(ids, L); rest = [i for i in ids if i not in lab]
            ch = lambda opts: min(opts, key=lambda k: np.mean([Ei[k][i] for i in lab]))
            a, b = ch(same), ch(same + cross); pick += b in cross
            ea = np.mean([Ei[a][i] for i in rest]); eb = np.mean([Ei[b][i] for i in rest]); rel.append((eb - ea) / ea)
        menu[f'L{L}'] = dict(median=round(float(np.median(rel)) * 100, 2), mean=round(float(np.mean(rel)) * 100, 2), worse_share=round(float(np.mean(np.array(rel) > 0)), 3), cross_picked=round(pick / 200, 3))
    out[src] = dict(images=len(ids), single={m: round(mean[f'single|{m}'], 2) for m in M}, means={k: round(v, 2) for k, v in mean.items()},
                    best_cross_vs_best_same=ci(bs_, bc), cross3_unan_vs_best_same=ci(bs_, 'cross3_unan'), cross3_maj_vs_best_same=ci(bs_, 'cross3_maj'),
                    cross3_unan_vs_avg_same3_unan=ci('avg_same3_unan', 'cross3_unan'),
                    sign_check=sign, sign_agree_cross=f"{sum(s['agree'] for s in sign if not s['pair'].endswith('same'))}/{sum(1 for s in sign if not s['pair'].endswith('same'))}",
                    sign_agree_same=f"{sum(s['agree'] for s in sign if s['pair'].endswith('same'))}/3",
                    delta_cross_gt_same=f"{sum(1 for s in sign if not s['pair'].endswith('same') and s['delta'] > dsame[s['pair'].split('|')[0]])}/6", menu=menu)
    per_image[src] = {k: v for k, v in Ei.items()}
    print(src, json.dumps({k: v for k, v in out[src].items() if k not in ('means', 'sign_check')}, default=float))
    for s in sign: print('  ', s)
json.dump(dict(out=out, per_image=per_image), open('/home/claude/stageB/xmodel/xmodel_parse.json', 'w'), indent=1, default=float)
