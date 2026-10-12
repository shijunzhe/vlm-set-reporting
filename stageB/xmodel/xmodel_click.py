"""Cross-model fusion on GUI click grounding (EXP-GUI, ScreenSpot-Pro / OSWorld-G items, C1 and C2 per model). No model calls.
Per item: same-model pair (C1, C2 of one model) vs cross-model pair (C1 of A, C1 of B); three models' C1.
  agreement alarm: act if the two points agree (distance <= tau of the larger side), else alarm (THM-FLOOR alarm).
    reported: coverage, accuracy when acting, share of errors that agree (consistent but wrong), error rate under alarm.
  cross3 vote: if >= 2 models agree, act with the mean of the agreeing points; else fall back to a fixed model's C1.
Baselines: each model's C1; the best model (by accuracy, chosen with labels, reference only).
usage: python -I xmodel_click.py"""
import json, os, sys, math, itertools
import numpy as np
sys.path.insert(0, '/home/claude/gui/code')
import fsrgui as G
os.chdir(G.ROOT)
MODELS = ['gpt61', 'sonnet55', 'gem38']; TAU = 0.015; rng = np.random.default_rng(0)
C = {c['id']: c for c in json.load(open('data/click_items_prepped.json'))}
Q, R = {}, {}
for l in open('runs/main/requests_A.jsonl'): q = json.loads(l); Q[q['request_id']] = q
for m in MODELS:
    for l in open(f'runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': R[r['request_id']] = r
sc = lambda q: None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
def pt(cid, m, s):
    k = f'{cid}|{m}|{s}'
    return G.decode_click(R[k]['raw_text'], sc(Q[k]))[0] if k in R else 'missing'
def dist(p, q, c): return math.hypot((p[0] - q[0]) * c['width'], (p[1] - q[1]) * c['height']) / max(c['width'], c['height'])
items = []
for cid, c in C.items():
    P = {(m, s): pt(cid, m, s) for m in MODELS for s in ('C1', 'C2')}
    if any(v == 'missing' for v in P.values()): continue
    items.append((cid, c, P))
out = dict(n=len(items))
acc = {m: np.mean([G.hit(P[(m, 'C1')], c['bbox_norm']) for _, c, P in items]) for m in MODELS}
out['single_acc'] = {m: round(float(v), 4) for m, v in acc.items()}; best = max(MODELS, key=acc.get); out['best'] = best


def pair_stats(pairs):
    """pairs: list of (p_act, p_other, c). act with p_act when they agree."""
    act = err_all = agree_wrong = alarm = alarm_err = 0; hits_act = 0
    for p, q, c in pairs:
        h = G.hit(p, c['bbox_norm']); ag = p is not None and q is not None and dist(p, q, c) <= TAU
        err_all += 1 - h
        if ag: act += 1; hits_act += h; agree_wrong += 1 - h
        else: alarm += 1; alarm_err += 1 - h
    n = len(pairs)
    return dict(coverage=round(act / n, 4), acc_when_acting=round(hits_act / max(1, act), 4), consistent_but_wrong_share=round(agree_wrong / max(1, err_all), 4),
                err_rate_under_alarm=round(alarm_err / max(1, alarm), 4), errors_caught=round(alarm_err / max(1, err_all), 4))


for a in MODELS:
    out[f'same|{a}'] = pair_stats([(P[(a, 'C1')], P[(a, 'C2')], c) for _, c, P in items])
    for b in MODELS:
        if b != a: out[f'cross|{a}|{b}'] = pair_stats([(P[(a, 'C1')], P[(b, 'C1')], c) for _, c, P in items])


def vote(P, c, fallback):
    pts = [P[(m, 'C1')] for m in MODELS]; best_pair = None
    for i, j in itertools.combinations(range(3), 2):
        if pts[i] is not None and pts[j] is not None and dist(pts[i], pts[j], c) <= TAU:
            grp = [pts[k] for k in range(3) if pts[k] is not None and (dist(pts[k], pts[i], c) <= TAU or dist(pts[k], pts[j], c) <= TAU)]
            best_pair = [float(np.mean([g[0] for g in grp])), float(np.mean([g[1] for g in grp]))]; break
    return (best_pair, True) if best_pair else (P[(fallback, 'C1')], False)


def boot_diff(h_a, h_b):
    h_a = np.array(h_a, float); h_b = np.array(h_b, float); d = h_b - h_a; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return dict(a=round(float(h_a.mean()), 4), b=round(float(h_b.mean()), 4), diff=round(float(d.mean()), 4), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])


for fb in MODELS:
    hv = []; agreed = 0
    for _, c, P in items:
        p, ag = vote(P, c, fb); hv.append(G.hit(p, c['bbox_norm'])); agreed += ag
    out[f'cross3_vote_fallback_{fb}'] = dict(vs_best_single=boot_diff([G.hit(P[(best, 'C1')], c['bbox_norm']) for _, c, P in items], hv),
                                            vs_fallback_single=boot_diff([G.hit(P[(fb, 'C1')], c['bbox_norm']) for _, c, P in items], hv), agreed_share=round(agreed / len(items), 4))
# same-model two-call alternative at equal calls for the best model: C1 (C2 cannot change the point without a rule) -> report only single
json.dump(out, open('/home/claude/stageB/xmodel/xmodel_click.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
# GPT-fallback vote against each single model (one variant for all three comparisons)
hv = [G.hit(vote(P, c, 'gpt61')[0], c['bbox_norm']) for _, c, P in items]
out['cross3_vote_fallback_gpt61_vs_each'] = {m: boot_diff([G.hit(P[(m, 'C1')], c['bbox_norm']) for _, c, P in items], hv) for m in MODELS}
json.dump(out, open('/home/claude/stageB/xmodel/xmodel_click.json', 'w'), indent=1)
print(out['cross3_vote_fallback_gpt61_vs_each'])
