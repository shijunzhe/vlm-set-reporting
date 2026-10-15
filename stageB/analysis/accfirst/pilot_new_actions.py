"""Accuracy-first pilot: analysis of the new actions on the 260 click items (uses responses collected 2026-10-14).
  CMP  fixed-candidate comparison (Gemini, evidence packet, both orders): ranking accuracy on one-right-one-wrong pairs, order
       consistency, and the selection rule 'tournament over clusters, verdicts kept only when both orders agree, otherwise the
       Gemini-score cluster rule'; G1 gate quantities (g, h, r, f) against the Gemini-score cluster rule.
  R/RS re-grounding: a' of a model re-grounding on a rival hypothesis vs its partner; coverage gain; selection with R points.
  S2   second-click zoomed points that are new hypotheses, scored by Gemini: coverage and selection.
  RG   region-first regeneration where the first click gave no point.
  DA/DB default reasoning setting: first-click and zoomed accuracy against the 'min' runs (same prompts).
Every item stays in the denominator. Labels are used only for scoring and for the gate quantities. Output: accfirst/pilot_new_actions.json"""
import collections, itertools, json, math, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; AN = HERE.parent
src = open(AN / 'autoselect.py').read(); src = src[:src.index('rng = np.random.default_rng')]
ns = {'__file__': str(AN / 'autoselect.py')}; exec(src, ns)
items, Z, Z2, C, M, HIT, dist, score, DELTA, cost, G = [ns[k] for k in ('items', 'Z', 'Z2', 'C', 'M', 'HIT', 'dist', 'score', 'DELTA', 'cost', 'G')]
ROOT = '/home/claude/gui'; GEM = 'gem38'
sc = lambda q: None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])


def load(stage):
    Q, R = {}, {}
    p = f'{ROOT}/runs/main/requests_{stage}.jsonl'
    if os.path.exists(p):
        for l in open(p): q = json.loads(l); Q[q['request_id']] = q
    for m in M:
        f = f'{ROOT}/runs/main/responses_{stage}_{m}.jsonl'
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                if r['status'] == 'completed': R[r['request_id']] = r
    return Q, R


def usd(R): return sum(cost(r['request_id'].split('|')[1], r) for r in R.values())


def hyps(c):
    H = []
    for m in M:
        p = Z[(c, m)]
        if p is None: continue
        for h in H:
            if any(dist(p, q, c) <= DELTA for _, q in h): h.append((m, p)); break
        else: H.append([(m, p)])
    return H


def gem_p(c, m):
    s = score(c, GEM, m); return (s or {}).get('p')


def cluster_gem(c, h):
    ps = [gem_p(c, m) for m, _ in h if gem_p(c, m) is not None]
    return np.mean(ps) if ps else 0.0


def rep(c, h): return max(h, key=lambda x: gem_p(c, x[0]) or 0)[1]


res = {}
# ---------------- base rule: Gemini-score cluster selection (label-free version: size first, then Gemini mean) ---------
def base_choice(c):
    H = hyps(c)
    if not H: return None
    return rep(c, max(H, key=lambda h: (cluster_gem(c, h), len(h))))


base_err = {c: 1 - HIT(base_choice(c), c) for c in items}
res['base_gem_cluster_error'] = round(100 * np.mean(list(base_err.values())), 2)
# ---------------- CMP ----------------
Qc, Rc = load('CMP'); res['cmp_calls'] = len(Rc); res['cmp_usd'] = round(usd(Rc), 3)
verd = {}  # (c, i, j) -> {order: preferred hyp index or None}
for rid, r in Rc.items():
    q = Qc[rid]; c = q['meta']['item']; ids = q['meta']['ids']; i, j = rid.split('|')[3][1], rid.split('|')[3][3]
    v = G.first_json(r['raw_text']); pid = v.get('preferred_id') if isinstance(v, dict) else None
    verd.setdefault((c, int(i), int(j)), {})[q['meta']['order']] = ids.get(pid) if pid in ids else None
pairs = []
for (c, i, j), d in verd.items():
    H = hyps(c); yi = int(any(HIT(p, c) for _, p in H[i])); yj = int(any(HIT(p, c) for _, p in H[j]))
    pairs.append(dict(c=c, i=i, j=j, yi=yi, yj=yj, ab=d.get('ab'), ba=d.get('ba')))
disc = [p for p in pairs if p['yi'] != p['yj']]
def right_of(p): return p['i'] if p['yi'] else p['j']
acc_single = np.mean([v == right_of(p) for p in disc for v in (p['ab'], p['ba'])])
agree = [p for p in disc if p['ab'] is not None and p['ab'] == p['ba']]
res['cmp'] = dict(pairs=len(pairs), discordant_pairs=len(disc), single_call_accuracy=round(float(acc_single), 3),
                  both_orders_agree=len(agree), agree_accuracy=round(float(np.mean([p['ab'] == right_of(p) for p in agree])), 3) if agree else None,
                  undecided_or_invalid=int(sum(v is None for p in pairs for v in (p['ab'], p['ba']))))
# Gemini pointwise ranking on the same discordant cluster pairs
gp = []
for p in disc:
    H = hyps(p['c']); si, sj = cluster_gem(p['c'], H[p['i']]), cluster_gem(p['c'], H[p['j']])
    gp.append(1.0 if (si > sj) == bool(p['yi']) and si != sj else (0.5 if si == sj else 0.0))
res['cmp']['gemini_pointwise_accuracy_same_pairs'] = round(float(np.mean(gp)), 3)


def cmp_choice(c):
    H = hyps(c)
    if len(H) < 2: return base_choice(c)
    wins = collections.Counter(); decided = False
    for i, j in itertools.combinations(range(len(H)), 2):
        d = verd.get((c, i, j), {})
        if d.get('ab') is not None and d.get('ab') == d.get('ba'): wins[d['ab']] += 1; decided = True
    if not decided: return base_choice(c)
    top = max(wins.values()); best = [k for k, v in wins.items() if v == top]
    if len(best) > 1: return rep(c, max([H[k] for k in best], key=lambda h: cluster_gem(c, h)))
    return rep(c, H[best[0]])


cmp_err = {c: 1 - HIT(cmp_choice(c), c) for c in items}
fixed = sum(1 for c in items if base_err[c] and not cmp_err[c]); broken = sum(1 for c in items if not base_err[c] and cmp_err[c])
res['cmp_rule_error'] = round(100 * np.mean(list(cmp_err.values())), 2); res['cmp_rule_fixed_broken'] = [fixed, broken]
# G1 gate quantities relative to the base rule on alarmed items (switch = cmp picks a different cluster)
g = h = rr = ff = 0
for c in items:
    if len(hyps(c)) < 2: continue
    b, a = base_choice(c), cmp_choice(c)
    if HIT(b, c) == HIT(a, c): continue
    if not HIT(b, c): g += 1
    else: h += 1
res['gate'] = dict(switches_fixing=fixed, switches_breaking=broken, note='g r and h f realised counts over 260 items')
print('base', res['base_gem_cluster_error'], 'CMP', res['cmp'], 'CMP rule', res['cmp_rule_error'], 'fixed/broken', fixed, broken)
# ---------------- R / RS ----------------
Qr, Rr = load('R'); Qrs, Rrs = load('RS'); res['r_calls'] = len(Rr); res['r_usd'] = round(usd(Rr) + usd({k: v for k, v in Rrs.items() if '|gem38|' in k}), 3)
RP = {}
for rid, r in Rr.items():
    q = Qr[rid]; m = q['model_key']; p, _ = G.decode_click(r['raw_text'], sc(q)); w = q['meta']['window']
    RP[rid] = [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else None
ar = {}
for base in M:
    W = [c for c in items if len(hyps(c)) >= 2 and not HIT(Z[(c, base)], c)]
    hit_r = [int(any(HIT(p, c) for rid, p in RP.items() if rid.startswith(c + '|' + base + '|R|'))) for c in W]
    partner = {m: np.mean([HIT(Z[(c, m)], c) for c in W]) for m in M if m != base}
    ar[base] = dict(n=len(W), a_prime_reground=round(float(np.mean(hit_r)), 3) if W else None, a_prime_other_models={k: round(float(v), 3) for k, v in partner.items()})
res['reground_a_prime'] = ar
floor3 = [c for c in items if not any(HIT(Z[(c, m)], c) for m in M)]
res['reground_floor'] = dict(floor_items=len(floor3), rescued=int(sum(any(HIT(p, c) for rid, p in RP.items() if rid.startswith(c + '|')) for c in floor3)))
print('R', ar, res['reground_floor'])
# ---------------- S2 ----------------
Qs2, Rs2 = load('S2')
S2P = {}
for rid, r in Rs2.items():
    q = Qs2[rid]; v = G.first_json(r['raw_text'])
    try: pv = float(v.get('p_hit'))
    except Exception: pv = None
    S2P[rid] = (q['meta']['click'], pv)
res['s2'] = dict(n=len(S2P), hits=int(sum(HIT(p, rid.split('|')[0]) for rid, (p, _) in S2P.items())),
                 floor_rescued=int(sum(any(HIT(p, c) for rid, (p, _) in S2P.items() if rid.startswith(c + '|')) for c in floor3)))
# ---------------- RS points and S2 points as extra candidates in the Gemini cluster rule ----------------
RSP = {}
for rid, r in Rrs.items():
    if '|gem38|' not in rid: continue
    q = Qrs[rid]; v = G.first_json(r['raw_text'])
    try: pv = float(v.get('p_hit'))
    except Exception: pv = None
    RSP[rid] = (q['meta']['click'], pv)


def extended_choice(c, use_r=True, use_s2=True):
    cands = [(p, gem_p(c, m)) for m, p in [(m, Z[(c, m)]) for m in M] if p is not None]
    if use_r: cands += [v for k, v in RSP.items() if k.startswith(c + '|')]
    if use_s2: cands += [v for k, v in S2P.items() if k.startswith(c + '|')]
    if not cands: return None
    H = []
    for p, s in cands:
        for h in H:
            if any(dist(p, q, c) <= DELTA for q, _ in h): h.append((p, s)); break
        else: H.append([(p, s)])
    best = max(H, key=lambda h: (np.mean([s for _, s in h if s is not None] or [0]), len(h)))
    return max(best, key=lambda x: x[1] or 0)[0]


for nm, kw in (('with_R', dict(use_r=True, use_s2=False)), ('with_S2', dict(use_r=False, use_s2=True)), ('with_R_S2', dict(use_r=True, use_s2=True))):
    e = {c: 1 - HIT(extended_choice(c, **kw), c) for c in items}
    res[f'gem_cluster_{nm}_error'] = round(100 * np.mean(list(e.values())), 2)
    res[f'gem_cluster_{nm}_fixed_broken'] = [sum(1 for c in items if base_err[c] and not e[c]), sum(1 for c in items if not base_err[c] and e[c])]
    cov = np.mean([1 - any(HIT(p, c) for p in [Z[(c, m)] for m in M] + ([v[0] for k, v in RSP.items() if k.startswith(c + '|')] if kw['use_r'] else []) + ([v[0] for k, v in S2P.items() if k.startswith(c + '|')] if kw['use_s2'] else [])) for c in items])
    res[f'coverage_floor_{nm}'] = round(100 * float(cov), 2)
print({k: v for k, v in res.items() if k.startswith('gem_cluster') or k.startswith('coverage')}, 'S2', res['s2'])
# ---------------- RG ----------------
Qg, Rg = load('RGZ'); out = []
for rid, r in Rg.items():
    q = Qg[rid]; c = q['meta']['item']; p, _ = G.decode_click(r['raw_text'], sc(q)); w = q['meta']['window']
    pt = [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else q['meta']['click']
    out.append(dict(item=c, model=q['model_key'], hit=HIT(pt, c)))
res['region_regeneration'] = out; print('RG', out)
# ---------------- DA / DB ----------------
Qa, Ra = load('DA'); Qb, Rb = load('DB'); res['default_usd'] = round(usd(Ra) + usd(Rb), 3)
QA0, RA0 = load('A'); QB0, RB0 = load('B')
dd = {}
for m in M:
    def zp(c, Qx, Rx, first_key, zoom_key):
        r1 = Rx[0].get(first_key)
        if r1 is None: return None
        p1, _ = G.decode_click(r1['raw_text'], sc(Qx[0][first_key]))
        r4 = Rx[1].get(zoom_key)
        if r4 is None or p1 is None: return p1
        q4 = Qx[1][zoom_key]; p, _ = G.decode_click(r4['raw_text'], sc(q4)); w = q4['meta']['window']
        return [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else p1
    e1 = [1 - HIT(G.decode_click(Ra[f'{c}|{m}|C1d']['raw_text'], sc(Qa[f'{c}|{m}|C1d']))[0], c) for c in items if f'{c}|{m}|C1d' in Ra]
    e0 = [1 - HIT(G.decode_click(RA0[f'{c}|{m}|C1']['raw_text'], sc(QA0[f'{c}|{m}|C1']))[0], c) for c in items]
    ez = [1 - HIT(zp(c, (Qa, Qb), (Ra, Rb), f'{c}|{m}|C1d', f'{c}|{m}|B4d'), c) for c in items]
    ez0 = [1 - HIT(Z[(c, m)], c) for c in items]
    think = [(r['usage'].get('thinking_tokens') or r['usage'].get('reasoning_tokens') or 0) for k, r in Ra.items() if f'|{m}|' in k]   # GPT reports reasoning_tokens
    dd[m] = dict(first_click_min=round(100 * np.mean(e0), 2), first_click_default=round(100 * np.mean(e1), 2) if e1 else None,
                 zoom_min=round(100 * np.mean(ez0), 2), zoom_default=round(100 * np.mean(ez), 2), n_default=len(e1),
                 mean_thinking_tokens=round(float(np.mean(think)), 1) if think else None,
                 zoom_fixed_broken=[sum(1 for a, b in zip(ez0, ez) if a and not b), sum(1 for a, b in zip(ez0, ez) if not a and b)])
res['default_reasoning'] = dd; print('default', dd, 'usd', res['default_usd'])
json.dump(res, open(HERE / 'pilot_new_actions.json', 'w'), indent=1, default=float)
