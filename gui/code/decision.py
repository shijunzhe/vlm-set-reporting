"""Frozen decision rules for the 600-item confirmation (protocol v3), also used to re-validate the pilot. No model calls.

Every flow returns, for each item, (point or None, flags). Flags: 'incomplete' = a request the flow needs was never executed
(study_incomplete, not a model failure); 'fallback:<what>' = a technical failure or missing output was handled by the fixed
fallback; a flow with no legal candidate returns None (counted as an error).

Configurations (explicit; provider parameters are those of run_e12b.REGISTRY / MAX_TOKENS):
  M: gpt61, sonnet55, gem38 all 'min'  (first click C1, zoom B4)
  D: gpt61 'default' (C1d, B4d), sonnet55 'default' (C1d, B4d), gem38 'min' (C1, B4)
Rules (protocol v3, clause B):
  candidate      = a model's zoomed point (B4 around C1; the first click if the zoom gave no point or failed)
  cluster        = connected component of the graph 'distance <= 1.5% of the longer side'
  representative = argmax over members of (Gemini p_hit, -source_rank); a missing score counts as -1 (fixed fallback)
  S rule         = argmax over clusters of (representative p_hit, cluster size, -source_rank of the cluster)
  compared set   = the top 3 clusters under the S key (all clusters when there are at most 3)
  C rule         = a representative gains one win when BOTH orders of a pair choose it; undecided, invalid or split verdicts
                   give no win; if any cluster has a win: among the clusters with the most wins, the S key decides; else S.
  GPT x3         = three independent GPT default two-stage zooms (rep 0, 1, 2); largest connected component; ties: the
                   cluster holding the earliest valid rep; output = the earliest valid rep's own point in that cluster.
  GLF            = gl_stages rounds GLF1..GLF3 (at most 3 calls); unreadable final reply -> centre of the last region (flagged).
"""
import collections, hashlib, itertools, json, math, os
DELTA = 0.015; ORDER = {'gpt61': 0, 'sonnet55': 1, 'gem38': 2}; M = ['gpt61', 'sonnet55', 'gem38']; GEM = 'gem38'; TOP = 3
CONFIG = {'M': {'gpt61': ('min', 'C1', 'B4'), 'sonnet55': ('min', 'C1', 'B4'), 'gem38': ('min', 'C1', 'B4')},
          'D': {'gpt61': ('default', 'C1d', 'B4d'), 'sonnet55': ('default', 'C1d', 'B4d'), 'gem38': ('min', 'C1', 'B4')}}
STAGE_OF = {'C1': 'A', 'B4': 'B', 'C1d': 'DA', 'B4d': 'DB', 'C1d1': 'DA', 'B4d1': 'DB', 'C1d2': 'DA', 'B4d2': 'DB'}


class Run:
    """request and response access for one run directory (runs/<subset>)"""

    def __init__(self, root, subset, G, B):
        self.root, self.sub, self.G, self.B = root, subset, G, B; self.cache = {}

    def stage(self, st):
        if st not in self.cache:
            Q, R, Fl = {}, {}, {}
            p = f'{self.root}/runs/{self.sub}/requests_{st}.jsonl'
            if os.path.exists(p):
                for l in open(p): q = json.loads(l); Q[q['request_id']] = q
            for m in M:
                f = f'{self.root}/runs/{self.sub}/responses_{st}_{m}.jsonl'
                if os.path.exists(f):
                    for l in open(f):
                        r = json.loads(l)
                        (R if r['status'] == 'completed' else Fl)[r['request_id']] = r
            for k in R: Fl.pop(k, None)
            self.cache[st] = (Q, R, Fl)
        return self.cache[st]

    def status(self, st, rid):
        Q, R, Fl = self.stage(st)
        if rid in R: return 'completed'
        if rid in Fl: return 'failed'
        return 'not_built' if rid not in Q else 'not_executed'

    def click(self, st, rid, m):
        Q, R, _ = self.stage(st); r = R[rid]
        return self.G.decode_click(r['raw_text'], self.B.scale_of(m, Q[rid]['meta']['sizes'][0]))[0]


def cand_id(item, m, setting, parent, p):
    return hashlib.sha256(f'{item}|{m}|{setting}|{parent}|{p[0]:.6f}|{p[1]:.6f}'.encode()).hexdigest()[:16]


def dist(p, q, c):
    return math.hypot((p[0] - q[0]) * c['width'], (p[1] - q[1]) * c['height']) / max(c['width'], c['height'])


def zoomed(run, cid, m, setting, first, zoom, flags):
    k1 = f'{cid}|{m}|{first}'; s1 = run.status(STAGE_OF[first], k1)
    if s1 in ('not_executed', 'not_built'): flags.add('incomplete'); return None, None
    if s1 == 'failed': flags.add(f'fallback:first_failed:{m}'); return None, None
    p1 = run.click(STAGE_OF[first], k1, m)
    if p1 is None: return None, None
    k4 = f'{cid}|{m}|{zoom}'; s4 = run.status(STAGE_OF[zoom], k4)
    if s4 in ('not_executed',): flags.add('incomplete'); return p1, k1
    if s4 == 'not_built': flags.add('incomplete'); return p1, k1
    if s4 == 'failed': flags.add(f'fallback:zoom_failed:{m}'); return p1, k1
    Q, R, _ = run.stage(STAGE_OF[zoom]); q = Q[k4]
    p = run.click(STAGE_OF[zoom], k4, m); w = q['meta']['window']
    if p is None: return p1, k1
    return [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])], k4


def atoms(run, cfg, cid, flags):
    out = []
    for m in M:
        setting, first, zoom = CONFIG[cfg][m]
        p, parent = zoomed(run, cid, m, setting, first, zoom, flags)
        if p is None: continue
        out.append(dict(id=cand_id(cid, m, setting, parent, p), model=m, setting=setting, parent=parent, point=p))
    return out


def scores(run, cid, A, flags):
    sc = {}
    for a in A:
        st = 'S' if a['setting'] == 'min' else 'SD'; k = f'{cid}|{GEM}|{st}|{a["model"]}'
        s = run.status(st, k)
        if s != 'completed':
            flags.add('incomplete' if s in ('not_executed', 'not_built') else f'fallback:score_failed:{a["model"]}'); sc[a['id']] = None; continue
        Q, R, _ = run.stage(st)
        assert all(abs(u - v) < 1e-9 for u, v in zip(Q[k]['meta']['click'], a['point'])), ('scored point differs from candidate', k)
        v = run.G.first_json(R[k]['raw_text'])
        try: p = float(v.get('p_hit'))
        except Exception: p = None
        if p is None or not (0 <= p <= 1): flags.add(f'fallback:score_unreadable:{a["model"]}'); p = None
        sc[a['id']] = p
    return sc


def clusters(c, A):
    n = len(A); par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i, j in itertools.combinations(range(n), 2):
        if dist(A[i]['point'], A[j]['point'], c) <= DELTA: par[f(i)] = f(j)
    g = collections.defaultdict(list)
    for i in range(n): g[f(i)].append(A[i])
    return sorted(g.values(), key=lambda x: min(ORDER[a['model']] for a in x))


def scval(sc, a): return sc.get(a['id']) if sc.get(a['id']) is not None else -1.0


def structure(run, cfg, cid, c):
    flags = set(); A = atoms(run, cfg, cid, flags); sc = scores(run, cid, A, flags)
    cls = []
    for g in clusters(c, A):
        rep = max(g, key=lambda a: (scval(sc, a), -ORDER[a['model']]))
        cls.append(dict(members=[a['id'] for a in g], rep=rep, size=len(g), rank=min(ORDER[a['model']] for a in g)))
    return A, sc, cls, flags


def s_key(sc, cl): return (scval(sc, cl['rep']), cl['size'], -cl['rank'])


def compared(sc, cls):
    return sorted(cls, key=lambda cl: s_key(sc, cl), reverse=True)[:TOP]


def s_rule(run, cfg, cid, c):
    A, sc, cls, flags = structure(run, cfg, cid, c)
    if not cls: return None, flags
    return max(cls, key=lambda cl: s_key(sc, cl))['rep']['point'], flags


def c_rule(run, cfg, cid, c):
    A, sc, cls, flags = structure(run, cfg, cid, c)
    if not cls: return None, flags
    S = max(cls, key=lambda cl: s_key(sc, cl))
    comp = compared(sc, cls)
    if len(comp) < 2: return S['rep']['point'], flags
    st = f'CMPA_{cfg}'; Q, R, _ = run.stage(st); wins = collections.Counter()
    for x, y in itertools.combinations(comp, 2):
        ia, ib = sorted([x['rep']['id'], y['rep']['id']]); verdict = {}
        for o, (u, v) in (('ab', (x, y)), ('ba', (y, x))):
            # request ids were built with the pair in a given order; look up both spellings
            for rid in (f'{cid}|{GEM}|{st}|{u["rep"]["id"]}|{v["rep"]["id"]}|{o}', f'{cid}|{GEM}|{st}|{v["rep"]["id"]}|{u["rep"]["id"]}|{o}'):
                s = run.status(st, rid)
                if s == 'not_built': continue
                if s == 'completed':
                    q = Q[rid]; assert set(q['meta']['points']) == {ia, ib}
                    for i, p in q['meta']['points'].items():
                        tgt = x if x['rep']['id'] == i else y
                        assert all(abs(s1 - s2) < 1e-9 for s1, s2 in zip(p, tgt['rep']['point'])), ('drawn point differs', rid)
                    rv = run.G.first_json(R[rid]['raw_text']); lab = rv.get('preferred_id') if isinstance(rv, dict) else None
                    verdict[q['meta']['order']] = q['meta']['labels'].get(lab)
                elif s == 'failed': flags.add('fallback:cmp_failed')
                else: flags.add('incomplete')
                break
        if verdict.get('ab') is not None and verdict.get('ab') == verdict.get('ba'): wins[verdict['ab']] += 1
    if not wins: return S['rep']['point'], flags
    top = max(wins.values()); best = [cl for cl in comp if wins.get(cl['rep']['id'], 0) == top]
    return max(best, key=lambda cl: s_key(sc, cl))['rep']['point'], flags


def single_zoom(run, cfg, m, cid):
    flags = set(); setting, first, zoom = CONFIG[cfg][m]
    p, _ = zoomed(run, cid, m, setting, first, zoom, flags); return p, flags


def gpt_x3(run, cid, c):
    flags = set(); pts = []
    for rep, (first, zoom) in enumerate((('C1d', 'B4d'), ('C1d1', 'B4d1'), ('C1d2', 'B4d2'))):
        p, _ = zoomed(run, cid, 'gpt61', 'default', first, zoom, flags)
        if p is not None: pts.append((rep, p))
    if not pts: return None, flags
    n = len(pts); par = list(range(n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i, j in itertools.combinations(range(n), 2):
        if dist(pts[i][1], pts[j][1], c) <= DELTA: par[f(i)] = f(j)
    g = collections.defaultdict(list)
    for i in range(n): g[f(i)].append(pts[i])
    best = max(g.values(), key=lambda cl: (len(cl), -min(r for r, _ in cl)))
    return min(best, key=lambda x: x[0])[1], flags


def glf(run, cid, final_point_fn, C):
    flags = set(); L = {n: run.stage(f'GLF{n}')[:2] for n in (1, 2, 3)}
    if run.status('GLF1', f'{cid}|gpt61|GLF1') in ('not_executed', 'not_built'): flags.add('incomplete'); return None, flags
    p, calls = final_point_fn(cid, C, L, 'GLF')
    return p, flags
