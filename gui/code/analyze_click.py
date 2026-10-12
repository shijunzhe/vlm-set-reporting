"""Stage 2 analysis: click grounding (no model calls; labels used only in scoring).
Arms per model x instruction:
  B1 single click C1 | B2 two clicks: act if C1 and C2 agree (distance <= tau of the image's larger side), else ask |
  B3 click + verification: correct/uncertain -> C1; wrong with a point -> corrected point; wrong without a point -> ask |
  B4 two-stage zoom: refined point if found in the crop, else C1.
Outputs runs/main/click_analysis.json: accuracy (ask counts as not done), coverage, accuracy when acting, alarm statistics,
per platform and target-size tercile; presets across platforms.
"""
import json, os, sys, collections, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrgui as G
sys.path.insert(0, '/home/claude/fsr_v51/e3'); from score_e3 import PRICE
os.chdir(G.ROOT)
SUB = sys.argv[1] if len(sys.argv) > 1 else 'main'
MODELS = ['gpt61', 'sonnet55', 'gem38']
C = {c['id']: c for c in json.load(open('data/click_items_prepped.json'))}


def load(stage):
    Q, R = {}, {}
    for l in open(f'runs/{SUB}/requests_{stage}.jsonl'): q = json.loads(l); Q[q['request_id']] = q
    for m in MODELS:
        f = f'runs/{SUB}/responses_{stage}_{m}.jsonl'
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['status'] == 'completed': R[r['request_id']] = r
    return Q, R


QA, RA = load('A'); QB, RB = load('B')
def sc(q): return None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
def cost(m, r): return (r['usage']['input_tokens'] * PRICE[m][0] + r['usage']['output_tokens'] * PRICE[m][1]) / 1e6 if r else 0.0


def dist(p, q, c):
    W, H = c['width'], c['height']; return math.hypot((p[0] - q[0]) * W, (p[1] - q[1]) * H) / max(W, H)


rows = []
for m in MODELS:
    for cid, c in C.items():
        k = lambda s: f'{cid}|{m}|{s}'
        if k('C1') not in RA or k('C2') not in RA: continue
        p1, _ = G.decode_click(RA[k('C1')]['raw_text'], sc(QA[k('C1')])); p2, _ = G.decode_click(RA[k('C2')]['raw_text'], sc(QA[k('C2')]))
        bb = c['bbox_norm']; area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        r = dict(model=m, item=cid, source=c['source'], platform=c['platform'], group=c['group'], ui_type=c['ui_type'], area=area,
                 h1=int(G.hit(p1, bb)), h2=int(G.hit(p2, bb)), none1=int(p1 is None),
                 d12=(dist(p1, p2, c) if p1 and p2 else None), c1=cost(m, RA[k('C1')]), c2=cost(m, RA[k('C2')]))
        # B3
        rb = RB.get(k('B3')); r['b3_dec'] = None; r['b3_final'] = p1; r['b3_ask'] = int(p1 is None); r['c3'] = cost(m, rb)
        if rb is not None and p1 is not None:
            qb = QB[k('B3')]; d, pt, _ = G.decode_vclick(rb['raw_text'], sc(qb)); w = qb['meta']['window']
            r['b3_dec'] = d
            if d == 'wrong':
                if pt is not None: r['b3_final'] = [w[0] + pt[0] * (w[2] - w[0]), w[1] + pt[1] * (w[3] - w[1])]
                else: r['b3_final'] = None; r['b3_ask'] = 1
        r['h3'] = int(G.hit(r['b3_final'], bb))
        # B4
        r4 = RB.get(k('B4')); r['b4_final'] = p1; r['c4'] = cost(m, r4); r['b4_found'] = None
        if r4 is not None and p1 is not None:
            q4 = QB[k('B4')]; pt, _ = G.decode_click(r4['raw_text'], sc(q4)); w = q4['meta']['window']
            r['b4_found'] = int(pt is not None)
            if pt is not None: r['b4_final'] = [w[0] + pt[0] * (w[2] - w[0]), w[1] + pt[1] * (w[3] - w[1])]
        r['h4'] = int(G.hit(r['b4_final'], bb))
        for key in ('b3_final', 'b4_final'): r.pop(key)
        rows.append(r)
json.dump(rows, open(f'runs/{SUB}/click_rows.json', 'w'))


def mean(v): v = [x for x in v if x is not None]; return float(np.mean(v)) if v else float('nan')


def summarize(R, tau=0.015):
    out = dict(n=len(R), B1=mean([r['h1'] for r in R]), B4=mean([r['h4'] for r in R]), B3=mean([r['h3'] for r in R]),
               B3_ask=mean([r['b3_ask'] for r in R]), none1=mean([r['none1'] for r in R]))
    act = [r for r in R if r['d12'] is not None and r['d12'] <= tau]; dis = [r for r in R if not (r['d12'] is not None and r['d12'] <= tau)]
    out['B2_coverage'] = len(act) / max(1, len(R)); out['B2_acc_when_act'] = mean([r['h1'] for r in act])
    out['B2_err_rate_when_alarm'] = mean([1 - r['h1'] for r in dis]); out['B2_acc'] = sum(r['h1'] for r in act) / max(1, len(R))
    out['agree_but_wrong_share_of_errors'] = sum(1 - r['h1'] for r in act) / max(1, sum(1 - r['h1'] for r in R))
    a3 = [r for r in R if not r['b3_ask']]; out['B3_acc_when_act'] = mean([r['h3'] for r in a3])
    out['B3_fixed'] = sum(1 for r in R if r['h1'] == 0 and r['h3'] == 1); out['B3_broke'] = sum(1 for r in R if r['h1'] == 1 and r['h3'] == 0 and not r['b3_ask'])
    out['B3_asked_correct'] = sum(1 for r in R if r['h1'] == 1 and r['b3_ask'] and r['none1'] == 0)
    out['B4_fixed'] = sum(1 for r in R if r['h1'] == 0 and r['h4'] == 1); out['B4_broke'] = sum(1 for r in R if r['h1'] == 1 and r['h4'] == 0)
    out['cost_B1'] = mean([r['c1'] for r in R]); out['cost_B4'] = mean([r['c1'] + r['c4'] for r in R]); out['cost_B3'] = mean([r['c1'] + r['c3'] for r in R])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


S = {}
for m in MODELS:
    Rm = [r for r in rows if r['model'] == m]
    S[f'{m}|all_sspro'] = summarize([r for r in Rm if r['source'] == 'sspro'])
    S[f'{m}|all_ssv2'] = summarize([r for r in Rm if r['source'] == 'ssv2'])
    for p in ('windows', 'macos', 'linux', 'desktop', 'web'): S[f'{m}|{p}'] = summarize([r for r in Rm if r['platform'] == p])
    sp = [r for r in Rm if r['source'] == 'sspro']; q = np.percentile([r['area'] for r in sp], [33.3, 66.7])
    for name, f in (('small', lambda a: a <= q[0]), ('mid', lambda a: q[0] < a <= q[1]), ('large', lambda a: a > q[1])):
        S[f'{m}|sspro_size_{name}'] = summarize([r for r in sp if f(r['area'])])
    S[f'{m}|tau_sweep'] = {str(t): summarize([r for r in sp], t)['B2_err_rate_when_alarm'] for t in (0.005, 0.01, 0.015, 0.02, 0.03)}
# presets across platforms (ScreenSpot-Pro): choose B1/B3/B4 on the other platforms, score on this one
pre = {}
for m in MODELS:
    for p in ('windows', 'macos', 'linux'):
        tg = [r for r in rows if r['model'] == m and r['platform'] == p]; og = [r for r in rows if r['model'] == m and r['source'] == 'sspro' and r['platform'] != p]
        acc = lambda R, a: mean([r[{'B1': 'h1', 'B3': 'h3', 'B4': 'h4'}[a]] for r in R])
        ch = max(('B1', 'B3', 'B4'), key=lambda a: acc(og, a)); best = max(('B1', 'B3', 'B4'), key=lambda a: acc(tg, a))
        pre[f'{m}|{p}'] = dict(preset=ch, preset_acc=round(acc(tg, ch), 4), best=best, best_acc=round(acc(tg, best), 4), B1=round(acc(tg, 'B1'), 4),
                               eps=round(max(abs(acc(tg, a) - acc(og, a)) for a in ('B1', 'B3', 'B4')), 4))
json.dump(dict(summary=S, presets=pre), open(f'runs/{SUB}/click_analysis.json', 'w'), indent=1)
for k, v in S.items(): print(k, v)
for k, v in pre.items(): print(k, v)
