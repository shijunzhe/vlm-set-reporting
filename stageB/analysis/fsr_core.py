"""FSR analysis core: the objects of Sections 3 and 4 of the paper.
One implementation of records, clusters, feasible sets, matching rank, losses and the constrained oracle, shared by every
analysis script. No model calls. Labels are read only by score(), compat() and the oracle.
Sources: E3 (LIST-min, 3 calls x 3 models x 4 datasets x 60 images) via fsr_v51/e3; GUI screen parsing via gui/."""
import hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

E3 = Path('/home/claude/fsr_v51/e3'); GUI = Path('/home/claude/gui')
MODELS = ['gpt61', 'sonnet55', 'gem38']


def stable_seed(*parts):
    """process-independent seed (replaces Python hash(), which is salted per process)"""
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def maxmatch(R):
    R = np.asarray(R, bool)
    if R.size == 0 or not R.any(): return 0, np.full(R.shape[0], -1)
    m = maximum_bipartite_matching(csr_matrix(R.astype(np.int8)), perm_type='column')
    return int((m >= 0).sum()), m


# ------------------------------------------------------------------------------------------------ E3
class E3Data:
    def __init__(self):
        sys.path.insert(0, str(E3)); cwd = os.getcwd(); os.chdir(E3)
        import e3lib as L, analyze_e3 as A0, offline_agg2, aggregate_v2 as G   # offline_agg2 installs the lexicographic correspondence
        from score_e3 import load_items
        self.L, self.A0, self.G = L, A0, G
        self.items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl'); os.chdir(cwd)
        self.R = defaultdict(dict); self.cost = defaultdict(dict)
        PRICE = {'gpt61': (2, 10), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}
        for x in rows:
            if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min':
                self.R[(x['dataset'], x['model'])][(x['item'], x['rep'])] = x
                self.cost[(x['dataset'], x['model'])][(x['item'], x['rep'])] = ((x.get('in_tok') or 0) * PRICE[x['model']][0] + (x.get('out_tok') or 0) * PRICE[x['model']][1]) / 1e6
        self.datasets = sorted({k[0] for k in self.R})

    def images(self, ds):
        return sorted(set.intersection(*[{i for (i, r) in self.R[(ds, m)] if all((i, k) in self.R[(ds, m)] for k in (0, 1, 2))} for m in MODELS]))

    def metric(self, i):
        it = self.items[i]; return self.A0.metric_of(it, self.A0.primary(it))

    def reps(self, ds, m, i):
        it = self.items[i]
        return [[r for r in self.A0.recs_of(self.R[(ds, m)][(i, k)]) if it['task'] == 'point' or r['box'] is not None] for k in (0, 1, 2)]

    def clusters(self, i, reports):
        return self.G.build_clusters(self.items[i], reports)

    def fuse(self, members):
        return self.G.fuse(members)

    def compat(self, i, recs):
        """records x targets compatibility rho (draft 1.1)"""
        it = self.items[i]; mt = self.metric(i); Y = it['targets']; L = self.L
        if not recs or not Y: return np.zeros((len(recs), len(Y)), bool)
        if mt == 'point':
            P = np.array(L._pts(recs, it['labels']), float)
            T = np.array([((t['point'][0] if 'point' in t else (t['box'][0] + t['box'][2]) / 2) + L.LABEL_OFFSET * it['labels'].index(t['label']),
                           (t['point'][1] if 'point' in t else (t['box'][1] + t['box'][3]) / 2)) for t in Y], float)
            return np.hypot(P[:, None, 0] - T[None, :, 0], P[:, None, 1] - T[None, :, 1]) <= it['tau_point']
        M = L.iou_matrix([r['box'] for r in recs], [t['box'] for t in Y]) >= mt[1] - 1e-12
        return M & np.array([[r['label'] == t['label'] for t in Y] for r in recs])

    def score(self, i, recs):
        """draft 1.5: r = maximum matching, c = scored records (ignore regions), E, F1, recall"""
        it = self.items[i]; E, r, c = self.L.error(it, recs, self.metric(i)); n = len(it['targets'])
        return dict(E=E, r=r, c=c, n=n, F1=(1.0 if n + c == 0 else 2 * r / (n + c)), recall=(r / n if n else 1.0),
                    precision=(r / c if c else 1.0))


# ------------------------------------------------------------------------------------------------ GUI parse
class GUIData:
    def __init__(self):
        sys.path.insert(0, str(GUI / 'code')); cwd = os.getcwd(); os.chdir(GUI)
        import fsrgui as F
        self.F = F
        self.items = {i['id']: i for i in json.load(open('data/parse_items.json'))}
        Q, R = {}, {}
        for l in open('runs/main/requests_A.jsonl'): q = json.loads(l); Q[q['request_id']] = q
        for m in MODELS:
            for l in open(f'runs/main/responses_A_{m}.jsonl'):
                r = json.loads(l)
                if r['status'] == 'completed': R[r['request_id']] = r
        os.chdir(cwd); self.Q, self.R = Q, R

    def draft(self, iid, m, j):
        k = f'{iid}|{m}|D{j}'
        if k not in self.R: return None
        q = self.Q[k]; sc = [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
        return self.F.decode_list(self.R[k]['raw_text'], sc[0])[0]

    def images(self, src):
        return sorted(i for i, it in self.items.items() if it['source'] == src and all(self.draft(i, m, j) is not None for m in MODELS for j in (1, 2, 3)))

    def score(self, iid, boxes):
        s = self.F.score_boxes(boxes, self.items[iid]['targets']); n = s['n']
        return dict(E=s['E'], r=s['tp'], c=s['cnt'], n=n, F1=s['f1'], recall=(s['tp'] / n if n else 1.0), precision=(s['tp'] / s['cnt'] if s['cnt'] else 1.0))

    def compat(self, iid, boxes):
        Y = self.items[iid]['targets']
        if not boxes or not Y: return np.zeros((len(boxes), len(Y)), bool)
        return self.F.iou_mat(boxes, Y) >= 0.5


def constrained_oracle(compat_by_cluster, n):
    """Proposition 8: clusters x targets graph (edge if some candidate of the cluster is compatible); E* = n - |M|.
    compat_by_cluster: list of (n_candidates x n) bool arrays. Returns (E*, |M|, choice) with choice[q] = (cand index, target) or None."""
    if not compat_by_cluster or n == 0: return n, 0, [None] * len(compat_by_cluster)
    CT = np.array([c.any(0) for c in compat_by_cluster], bool)
    size, m = maxmatch(CT)
    choice = []
    for q, c in enumerate(compat_by_cluster):
        t = m[q]
        choice.append((int(np.argmax(c[:, t])), int(t)) if t >= 0 else None)
    return n - size, size, choice


def unrestricted_ceiling(compat_atoms, n):
    size, _ = maxmatch(compat_atoms)
    return n - size, size
