"""P2 (written before B2 is run): zero-label view-planner prediction of the recall change when GroundCUA screens are read as
four separate native-resolution crop calls (B2 blind arm, merged) instead of one full-view call. Model per element:
q = logistic(a + b log s) fitted on full-view drafts D1..D3 (binary hits, IoU 0.5); s = apparent size in effective pixels
(GPT, Sonnet: sent pixels; Gemini: view-relative units, fixed per-image budget). Crop zoom factor z per model from the
image sizes the crop will be sent at. Elements cut by every crop get the truncation factor measured in the existing
one-call tile listings (q_cut / q_uncut under tiles). Composition cost is assumed absent (separate calls).
Only labels of the existing D1..D3 / T1..T2 runs are used, as in the planner a reader would calibrate once per model."""
import json, os, sys, time, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '/home/claude/gui/code'); os.chdir('/home/claude/gui')
import fsrgui as G
P = {i['id']: i for i in json.load(open('data/parse_items.json'))}
QA, RA = {}, {}
for l in open('runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in ('gpt61', 'sonnet55', 'gem38'):
    for l in open(f'runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
def inside(b, c): return b[0] >= c[0] - 1e-9 and b[1] >= c[1] - 1e-9 and b[2] <= c[2] + 1e-9 and b[3] <= c[3] + 1e-9
out = {'written': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cells': {}}
for m in ('gpt61', 'sonnet55', 'gem38'):
    X, Y, cutflag, s_list, zs = [], [], [], [], []
    tq_cut, tq_unc = [], []
    for iid, it in P.items():
        if it['source'] != 'groundcua': continue
        k = lambda s: f'{iid}|{m}|{s}'
        if not all(k(f'D{j}') in RA for j in (1, 2, 3)): continue
        W, H = QA[k('D1')]['meta']['sizes'][0]; Wo, Ho = it['width'], it['height']
        cw, ch = 0.53 * Wo, 0.53 * Ho
        if m == 'sonnet55': r = min(1.0, 1568 / max(cw, ch), (1_150_000 / (cw * ch)) ** 0.5)
        else: r = min(1.0, 2048 / max(cw, ch))
        z = (cw * r) / (0.53 * W) if m != 'gem38' else 1 / 0.53
        gt = it['targets']; hits = np.zeros(len(gt))
        for j in (1, 2, 3):
            S = [b for b, n in G.decode_list(RA[k(f'D{j}')]['raw_text'], sc(QA[k(f'D{j}')])[0])[0]]
            for a, b in G.lex_match(S, gt, 0.5): hits[b] += 1
        T = [[b for b, n in G.decode_tiles(RA[k(f'T{j}')]['raw_text'], sc(QA[k(f'T{j}')]) if sc(QA[k(f'T{j}')])[0] else None)[0]] for j in (1, 2) if k(f'T{j}') in RA]
        th = np.zeros(len(gt))
        for S in T:
            for a, b in G.lex_match(S, gt, 0.5): th[b] += 1 / max(1, len(T))
        for e, b in enumerate(gt):
            rel = np.sqrt(max(1e-9, (b[2] - b[0]) * (b[3] - b[1])))
            s = rel * 1000 if m == 'gem38' else rel * np.sqrt(W * H)
            cut = not any(inside(b, c) for c in G.CROPS)
            for h in range(3): X.append(np.log(s)); Y.append(int(hits[e] > h))
            s_list.append(s); cutflag.append(cut); zs.append(z)
            (tq_cut if cut else tq_unc).append(th[e])
    X = np.array(X); Y = np.array(Y)
    # binary hits replicated: hit in call j approximated by sorted support (exchangeable)
    A = np.column_stack([np.ones(len(X)), X])
    nll = lambda w: np.sum(np.logaddexp(0, A @ w) - Y * (A @ w))
    w = minimize(nll, np.zeros(2), method='BFGS').x
    s = np.array(s_list); z = np.array(zs); cut = np.array(cutflag)
    q0 = 1 / (1 + np.exp(-(w[0] + w[1] * np.log(s)))); q1 = 1 / (1 + np.exp(-(w[0] + w[1] * np.log(s * z))))
    trunc = (np.mean(tq_cut) / np.mean(tq_unc)) if tq_unc and np.mean(tq_unc) > 0 else 1.0
    q1 = np.where(cut, q1 * trunc, q1)
    out['cells'][f'{m}|groundcua'] = dict(logit=[round(float(v), 3) for v in w], median_zoom=round(float(np.median(z)), 3), truncation_factor=round(float(trunc), 3),
                                          recall_full_pred=round(float(q0.mean()), 4), recall_crops_pred=round(float(q1.mean()), 4),
                                          predicted_recall_change=round(float(q1.mean() - q0.mean()), 4),
                                          predicted_sign='+' if q1.mean() > q0.mean() else '-')
    print(m, out['cells'][f'{m}|groundcua'])
json.dump(out, open('/home/claude/stageB/offline/P2_view_planner_predictions.json', 'w'), indent=1)
