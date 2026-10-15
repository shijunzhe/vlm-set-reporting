"""Four-source click tier (D4): the D tier of decision.py plus the crop-first (GLF) point as a fourth candidate source.
Run directory runs/conf_v1 (the 600 confirmation items). New calls only for what D lacks: the judge's score of the GLF point
(stage SG) and the comparisons whose representative pair was not already asked in CMPA_D (stage CMPA_D4). A comparison
request depends only on the item, the two representatives and the order, so a pair already answered in CMPA_D is reused.

Rules (fixed in this file before the new calls; identical to decision.py except for the added source):
  candidates     = the three D candidates (decision.atoms, config D) + the GLF final point (gl_stages.final_point), model 'glf',
                   setting 'default', parent = the last GLF request that answered; identifier = decision.cand_id
  source order   = gpt61 0, sonnet55 1, gem38 2, glf 3 (the added source ranks last in every tie)
  scores         = Gemini 3.8 lowest, same prompt and crop as S/SD (stage SG for the GLF point); missing score -> -1
  clusters, representative, S key, compared set (top 3), wins, C rule: as in decision.py
Usage: python3 -I code/d4_stages.py --stage SG|CMPA_D4 [--check]      (writes runs/conf_v1/requests_<stage>.jsonl)
       python3 -I code/d4_stages.py --eval                              (evaluation; no calls)"""
import argparse, collections, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_stages as F
import accfirst_stages as AF
import gl_stages as GL
import decision as DC
import conf_stages as CS
B, G = F.B, F.G
SUB = 'conf_v1'; GEM = DC.GEM; ROOT = os.getcwd()
ORDER4 = dict(DC.ORDER, glf=3)


def glf_candidate(rr, cid, C):
    L = {n: rr.stage(f'GLF{n}')[:2] for n in (1, 2, 3)}
    p, calls = GL.final_point(cid, C, L, 'GLF')
    if p is None: return None
    parent = f'{cid}|gpt61|GLF{calls}'
    return dict(id=DC.cand_id(cid, 'glf', 'default', parent, p), model='glf', setting='default', parent=parent, point=p)


def scores4(rr, cid, A, flags):
    base = [a for a in A if a['model'] != 'glf']; sc = DC.scores(rr, cid, base, flags)
    for a in A:
        if a['model'] != 'glf': continue
        k = f'{cid}|{GEM}|SG|glf'; s = rr.status('SG', k)
        if s != 'completed': flags.add('incomplete' if s in ('not_executed', 'not_built') else 'fallback:score_failed:glf'); sc[a['id']] = None; continue
        Q, R, _ = rr.stage('SG')
        assert all(abs(u - v) < 1e-9 for u, v in zip(Q[k]['meta']['click'], a['point'])), ('scored point differs from candidate', k)
        v = G.first_json(R[k]['raw_text'])
        try: p = float(v.get('p_hit'))
        except Exception: p = None
        if p is None or not (0 <= p <= 1): flags.add('fallback:score_unreadable:glf'); p = None
        sc[a['id']] = p
    return sc


def clusters4(c, A):
    n = len(A); par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i, j in itertools.combinations(range(n), 2):
        if DC.dist(A[i]['point'], A[j]['point'], c) <= DC.DELTA: par[f(i)] = f(j)
    g = collections.defaultdict(list)
    for i in range(n): g[f(i)].append(A[i])
    return sorted(g.values(), key=lambda x: min(ORDER4[a['model']] for a in x))


def structure4(rr, cid, c, C, with_scores=True):
    flags = set(); A = DC.atoms(rr, 'D', cid, flags); g = glf_candidate(rr, cid, C)
    if g is not None: A = A + [g]
    sc = scores4(rr, cid, A, flags) if with_scores else {}
    cls = []
    for grp in clusters4(c, A):
        rep = max(grp, key=lambda a: (DC.scval(sc, a), -ORDER4[a['model']]))
        cls.append(dict(members=[a['id'] for a in grp], rep=rep, size=len(grp), rank=min(ORDER4[a['model']] for a in grp)))
    return A, sc, cls, flags


def verdict(rr, cid, x, y, flags):
    """both-order verdict for a representative pair; CMPA_D first (same request), then CMPA_D4"""
    ia, ib = sorted([x['rep']['id'], y['rep']['id']]); out = {}
    for o, (u, v) in (('ab', (x, y)), ('ba', (y, x))):
        found = False
        for st in ('CMPA_D', 'CMPA_D4'):
            for rid in (f'{cid}|{GEM}|{st}|{u["rep"]["id"]}|{v["rep"]["id"]}|{o}', f'{cid}|{GEM}|{st}|{v["rep"]["id"]}|{u["rep"]["id"]}|{o}'):
                s = rr.status(st, rid)
                if s == 'not_built': continue
                found = True
                if s == 'completed':
                    Q, R, _ = rr.stage(st); q = Q[rid]; assert set(q['meta']['points']) == {ia, ib}
                    for i, p in q['meta']['points'].items():
                        tgt = x if x['rep']['id'] == i else y
                        assert all(abs(s1 - s2) < 1e-9 for s1, s2 in zip(p, tgt['rep']['point'])), ('drawn point differs', rid)
                    rv = G.first_json(R[rid]['raw_text']); lab = rv.get('preferred_id') if isinstance(rv, dict) else None
                    out[q['meta']['order']] = q['meta']['labels'].get(lab)
                elif s == 'failed': flags.add('fallback:cmp_failed')
                else: flags.add('incomplete')
                break
            if found: break
        if not found: flags.add('incomplete')
    return out


def rule4(rr, cid, c, C, mode='C'):
    A, sc, cls, flags = structure4(rr, cid, c, C)
    if not cls: return None, flags, (A, sc, cls)
    S = max(cls, key=lambda cl: DC.s_key(sc, cl))
    if mode == 'S': return S['rep']['point'], flags, (A, sc, cls)
    comp = DC.compared(sc, cls)
    if len(comp) < 2: return S['rep']['point'], flags, (A, sc, cls)
    wins = collections.Counter()
    for x, y in itertools.combinations(comp, 2):
        v = verdict(rr, cid, x, y, flags)
        if v.get('ab') is not None and v.get('ab') == v.get('ba'): wins[v['ab']] += 1
    if not wins: return S['rep']['point'], flags, (A, sc, cls)
    top = max(wins.values()); best = [cl for cl in comp if wins.get(cl['rep']['id'], 0) == top]
    return max(best, key=lambda cl: DC.s_key(sc, cl))['rep']['point'], flags, (A, sc, cls)


def stage_SG(C):
    rr = CS.run(); R = []; j = GEM
    for cid, c in C.items():
        g = glf_candidate(rr, cid, C)
        if g is None: continue
        p = g['point']; w = G.window(p[0], p[1], 0.2)
        f, s = B.view(j, c['image'], f'{SUB}/{j}/{cid}/score_SG_glf', crop=w, marker=p)
        q = B.req(f'{cid}|{j}|SG|glf', j, [f], F.SYSTEM_S, F.judge_user(c, j, s), [s], item=cid, arm='SG', judge=j,
                  candidate_model='glf', candidate_id=g['id'], window=w, click=p)
        R.append(CS.tag(q, 'min', 0, parent=g['parent']))
    return R


def stage_CMPA_D4(C):
    rr = CS.run(); R = []; m = GEM; reused = 0
    for cid, c in C.items():
        A, sc, cls, flags = structure4(rr, cid, c, C)
        comp = sorted(DC.compared(sc, cls), key=lambda cl: cl['rank'])
        if len(comp) < 2: continue
        for x, y in itertools.combinations(comp, 2):
            for order, (u, v) in (('ab', (x, y)), ('ba', (y, x))):
                ra, rb = u['rep'], v['rep']
                if any(rr.status('CMPA_D', rid) != 'not_built' for rid in (f'{cid}|{m}|CMPA_D|{ra["id"]}|{rb["id"]}|{order}', f'{cid}|{m}|CMPA_D|{rb["id"]}|{ra["id"]}|{order}')):
                    reused += 1; continue
                used = set(); la = CS.label(ra['id'], used); used.add(la); lb = CS.label(rb['id'], used)
                pa, pb = ra['point'], rb['point']
                ov, so = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_D4_ov_{ra["id"]}_{rb["id"]}', marks=((pa, F.RED, la), (pb, F.BLUE, lb)))
                wa, wb = G.window(pa[0], pa[1], 0.2), G.window(pb[0], pb[1], 0.2)
                ca, sa = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_D4_crop_{ra["id"]}_red', crop=wa, marks=((pa, F.RED, la),))
                cb, sb = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_D4_crop_{rb["id"]}_blue', crop=wb, marks=((pb, F.BLUE, lb),))
                user = (G.size_note(B.MODE[m], [so, sa, sb], ['overview', f'crop_{la}', f'crop_{lb}']) +
                        f'Original task (platform {c["platform"]}): "{c["instruction"]}"\n'
                        f'Candidate IDs: {la} (red), {lb} (blue).\nViews in order: overview, crop_{la}, crop_{lb}.\n'
                        'Which candidate satisfies the original task? Follow the output contract exactly.')
                q = B.req(f'{cid}|{m}|CMPA_D4|{ra["id"]}|{rb["id"]}|{order}', m, [ov, ca, cb], AF.SYSTEM_CMP, user, [so, sa, sb], item=cid,
                          arm='CMPA_D4', order=order, labels={la: ra['id'], lb: rb['id']}, points={ra['id']: pa, rb['id']: pb},
                          pair=sorted([ra['id'], rb['id']]))
                R.append(CS.tag(q, 'min', 0, parent=[ra['parent'], rb['parent']]))
    print('pairs reused from CMPA_D (requests):', reused)
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage'); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    C = CS.items('all'); assert len(C) == 600
    R = dict(SG=stage_SG, CMPA_D4=stage_CMPA_D4)[a.stage](C)
    ids = [q['request_id'] for q in R]; assert len(ids) == len(set(ids))
    for q in R:
        for p in q['images']: assert os.path.exists(p), p
    sysref = {json.loads(l)['system'] for l in open(f'runs/{SUB}/requests_{"SD" if a.stage == "SG" else "CMPA_D"}.jsonl')}
    print(a.stage, len(R), 'requests; system prompt identical to the D stage for', sum(q['system'] in sysref for q in R))
    with open(f'runs/{SUB}/requests_{a.stage}.jsonl', 'w') as f:
        for q in R: f.write(json.dumps(q) + '\n')
