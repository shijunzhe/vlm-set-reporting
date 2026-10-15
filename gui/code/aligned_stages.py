"""Candidate registry with strict ID binding, and the aligned stages built on it (subset main; no model calls here).

Candidate (atom): one model's zoomed point for one generation configuration. candidate_id = sha256 of
(item, model, setting, parent request id, rounded coordinates); it is bound to one immutable point in the screenshot frame.
Configurations:
  M  minimal setting for all three models: zoomed point = B4 around C1 (C1 if the zoom gave no point).
  D  GPT-6.1 and Sonnet 5.5 with the default setting (C1d -> B4d), Gemini 3.8 minimal (C1 -> B4).
Clusters: connected components of the 'within delta = 1.5% of the longer side' graph (order-independent).
Representative of a cluster: chosen BEFORE any comparison call, by the Gemini judge's p_hit of each member (stage S for
minimal candidates, stage SD for default candidates; the scored point must equal the candidate's point), ties broken by
the fixed order gpt61, sonnet55, gem38. Members without a score count as p = -1.
Comparison (stage CMPA_<config>): for every pair of clusters, the two REPRESENTATIVES are drawn (overview + native crops
centred on the representative) and named by their candidate labels; the reply's preferred label maps back to exactly that
candidate's point. Both orders. Assertions check that drawn point = registry point = returned point.
Stage SD: Gemini judge scores of default-setting candidates (same prompt and crop as stage S).
Usage: python3 -I code/aligned_stages.py --stage SD | CMPA_M | CMPA_D
"""
import argparse, collections, hashlib, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_stages as F
import accfirst_stages as AF
B, G = F.B, F.G
M = F.M; SUB = 'main'; GEM = 'gem38'; ORDER = {m: i for i, m in enumerate(M)}


def _zoomed(Q1, R1, Q4, R4, k1, k4, m):
    r1 = R1.get(k1)
    if r1 is None: return None, None
    p1, _ = G.decode_click(r1['raw_text'], B.scale_of(m, Q1[k1]['meta']['sizes'][0]))
    r4 = R4.get(k4)
    if r4 is None or p1 is None: return p1, k1
    q4 = Q4[k4]; p, _ = G.decode_click(r4['raw_text'], B.scale_of(m, q4['meta']['sizes'][0])); w = q4['meta']['window']
    return ([w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])], k4) if p is not None else (p1, k1)


def cand_id(item, m, setting, parent, p):
    return hashlib.sha256(f'{item}|{m}|{setting}|{parent}|{p[0]:.6f}|{p[1]:.6f}'.encode()).hexdigest()[:16]


def atoms(config, C):
    QA, RA = F.load(SUB, 'A'); QB, RB = F.load(SUB, 'B'); QDA, RDA = F.load(SUB, 'DA'); QDB, RDB = F.load(SUB, 'DB')
    out = {}
    for cid in C:
        L = []
        for m in M:
            default = config == 'D' and m != GEM
            if default: p, parent = _zoomed(QDA, RDA, QDB, RDB, f'{cid}|{m}|C1d', f'{cid}|{m}|B4d', m)
            else: p, parent = _zoomed(QA, RA, QB, RB, f'{cid}|{m}|C1', f'{cid}|{m}|B4', m)
            if p is None: continue
            setting = 'default' if default else 'min'
            L.append(dict(id=cand_id(cid, m, setting, parent, p), item=cid, model=m, setting=setting, parent=parent, point=p))
        out[cid] = L
    return out


def gem_scores(config, C, A):
    """candidate_id -> Gemini p_hit; asserts the scored point equals the candidate point"""
    QS, RS = F.load(SUB, 'S'); QSD, RSD = F.load(SUB, 'SD') if os.path.exists(f'runs/{SUB}/requests_SD.jsonl') else ({}, {})
    sc = {}
    for cid, L in A.items():
        for a in L:
            if a['setting'] == 'min': k = f'{cid}|{GEM}|S|{a["model"]}'; Q, R = QS, RS
            else: k = f'{cid}|{GEM}|SD|{a["model"]}'; Q, R = QSD, RSD
            if k not in Q: continue
            assert all(abs(u - v) < 1e-9 for u, v in zip(Q[k]['meta']['click'], a['point'])), ('scored point differs', k)
            r = R.get(k); v = G.first_json(r['raw_text']) if r else None
            try: p = float(v.get('p_hit'))
            except Exception: p = None
            sc[a['id']] = p if p is not None and 0 <= p <= 1 else None
    return sc


def clusters(c, L):
    n = len(L); parent = list(range(n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in itertools.combinations(range(n), 2):
        if F.dist(L[i]['point'], L[j]['point'], c) <= F.DELTA: parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in range(n): groups[find(i)].append(L[i])
    return sorted(groups.values(), key=lambda g: min(ORDER[a['model']] for a in g))


def representative(g, sc):
    return max(g, key=lambda a: (sc.get(a['id']) if sc.get(a['id']) is not None else -1, -ORDER[a['model']]))


def registry(config, C):
    A = atoms(config, C); sc = gem_scores(config, C, A); reg = {}
    for cid, L in A.items():
        cl = clusters(C[cid], L)
        reg[cid] = [dict(members=[a['id'] for a in g], rep=representative(g, sc)) for g in cl]
    return A, sc, reg


def label(cand): return 'C' + cand[:4].upper()


def stage_SD(C):
    A = atoms('D', C); R = []; j = GEM
    for cid, L in A.items():
        c = C[cid]
        for a in L:
            if a['setting'] != 'default': continue
            p = a['point']; w = G.window(p[0], p[1], 0.2)
            f, s = B.view(j, c['image'], f'{SUB}/{j}/{cid}/score_D_{a["model"]}', crop=w, marker=p)
            R.append(B.req(f'{cid}|{j}|SD|{a["model"]}', j, [f], F.SYSTEM_S, F.judge_user(c, j, s), [s], item=cid, arm='SD', judge=j,
                           candidate_model=a['model'], candidate_id=a['id'], window=w, click=p))
    return R


def stage_CMPA(config, C):
    A, sc, reg = registry(config, C); R = []; m = GEM
    for cid, cls in reg.items():
        if len(cls) < 2: continue
        c = C[cid]
        for i, j in itertools.combinations(range(len(cls)), 2):
            for order, (a, b) in (('ab', (i, j)), ('ba', (j, i))):
                ra, rb = cls[a]['rep'], cls[b]['rep']; la, lb = label(ra['id']), label(rb['id']); assert la != lb
                pa, pb = ra['point'], rb['point']
                ov, so = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_{config}_ov_{ra["id"]}_{rb["id"]}', marks=((pa, F.RED, la), (pb, F.BLUE, lb)))
                wa, wb = G.window(pa[0], pa[1], 0.2), G.window(pb[0], pb[1], 0.2)
                ca, sa = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_{config}_crop_{ra["id"]}_red', crop=wa, marks=((pa, F.RED, la),))
                cb, sb = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_{config}_crop_{rb["id"]}_blue', crop=wb, marks=((pb, F.BLUE, lb),))
                user = (G.size_note(B.MODE[m], [so, sa, sb], ['overview', f'crop_{la}', f'crop_{lb}']) +
                        f'Original task (platform {c["platform"]}): "{c["instruction"]}"\n'
                        f'Candidate IDs: {la} (red), {lb} (blue).\nViews in order: overview, crop_{la}, crop_{lb}.\n'
                        'Which candidate satisfies the original task? Follow the output contract exactly.')
                R.append(B.req(f'{cid}|{m}|CMPA_{config}|{ra["id"]}|{rb["id"]}|{order}', m, [ov, ca, cb], AF.SYSTEM_CMP, user, [so, sa, sb],
                               item=cid, arm=f'CMPA_{config}', order=order, labels={la: ra['id'], lb: rb['id']},
                               points={ra['id']: pa, rb['id']: pb}, pair=sorted([ra['id'], rb['id']])))
    return R


def items_main():
    C = F.items_of(SUB); keep = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}
    return {k: v for k, v in C.items() if k in keep}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); a = ap.parse_args(); C = items_main()
    if a.stage == 'SD': R = stage_SD(C)
    elif a.stage.startswith('CMPA_'): R = stage_CMPA(a.stage.split('_')[1], C)
    else: raise SystemExit('unknown stage')
    with open(f'runs/{SUB}/requests_{a.stage}.jsonl', 'w') as f:
        for q in R: f.write(json.dumps(q) + '\n')
    print(a.stage, 'requests', len(R), collections.Counter((q['model_key'], q['meta']['arm']) for q in R))
