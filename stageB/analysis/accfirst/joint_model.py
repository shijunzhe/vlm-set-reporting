"""Coherent joint model for cross-model click fusion and the exact accuracy-first acquisition policy (pilot; no model calls).

Model (one per item; exchangeable, so the law of any subset of models is the marginal of the full law):
  h ~ Cat(pi)                         difficulty class (K classes)
  z_m | h ~ Bern(q[m, h])             model m's zoomed point hits the target, independently given h
  partition of the hit points   ~ CRP(alpha1)   (hits usually coincide; large elements can split them)
  partition of the wrong points ~ CRP(alpha0)   (small alpha0: wrong models fall on the same distractor)
  a hit and a wrong point never share an observed cluster (clusters at delta = 1.5% of the longer side)
  judge bin b_{j,m} | z_m, h ~ Cat(theta[j, z_m, h])  (judges share the difficulty class: shared deception)
Fitted label-free by EM on the full state (all three models), 2-fold cross-fitting by screenshot. Partial-state posteriors and
predictive distributions come from the same model by marginalization (CRP is projective).
Policy: actions = acquire model m (its zoomed point + the Gemini judge score of it; cost = measured mean dollars and 3 calls).
Exact backward induction per item over acquisition subsets and observed outcomes; objective P(error) + lam * dollars, with a hard
cap on acquisitions. The first acquisition is chosen by the policy (no forced base model).
Outputs: accfirst/joint_model.json (calibration, coherence, static errors, policy curve)."""
import collections, itertools, json, math
from pathlib import Path
import numpy as np
from scipy.special import gammaln
HERE = Path(__file__).resolve().parent; AN = HERE.parent
src = open(AN / 'autoselect.py').read(); src = src[:src.index('rng = np.random.default_rng')]
ns = {'__file__': str(AN / 'autoselect.py')}; exec(src, ns)
items, Z, C, M, HIT, dist, score, DELTA, RA, RB, cost = [ns[k] for k in ('items', 'Z', 'C', 'M', 'HIT', 'dist', 'score', 'DELTA', 'RA', 'RB', 'cost')]
BINS = [0.0, 0.6, 0.85, 0.95, 1.01]; NB = 4; K = 2; GEM = 'gem38'
binof = lambda p: None if p is None else int(np.digitize([p], BINS)[0] - 1)
PT = {(c, m): Z[(c, m)] for c in items for m in M}
JB = {(c, j, m): (binof(score(c, j, m)['p']) if score(c, j, m) else None) for c in items for j in M for m in M}
ACQ_USD = {m: float(np.mean([cost(m, RA.get(f'{c}|{m}|C1')) + cost(m, RB.get(f'{c}|{m}|B4')) + ((score(c, GEM, m) or {}).get('cost') or 0) for c in items])) for m in M}


def crp_logp(sizes, a):
    n = sum(sizes)
    if n == 0: return 0.0
    return len(sizes) * math.log(a) + gammaln(a) - gammaln(a + n) + sum(gammaln(s) for s in sizes)


def observed(c, S):
    """clusters (lists of models) among acquired models S that returned a point; models with no point are listed apart"""
    H = []; nopt = []
    for m in S:
        p = PT[(c, m)]
        if p is None: nopt.append(m); continue
        for h in H:
            if any(dist(p, PT[(c, o)], c) <= DELTA for o in h): h.append(m); break
        else: H.append([m])
    return H, nopt


class Joint:
    def __init__(self, judges):
        self.J = judges
        self.pi = np.array([.6, .4]); self.q = {m: np.array([.95, .5]) for m in M}
        self.a1, self.a0 = .3, .3
        self.th = {(j, z, h): (np.array([.05, .1, .25, .6]) if z else np.array([.4, .3, .2, .1])) for j in judges for z in (0, 1) for h in range(K)}

    def joint_terms(self, c, S, H, nopt, bins):
        """log P(observations, h, z) for every h and every assignment z over S consistent with clusters; returns list of (h, z, logp)"""
        out = []
        pointed = [m for m in S if m not in nopt]
        # z constant within a cluster; models without a point are misses (z = 0)
        for zc in itertools.product((0, 1), repeat=len(H)):
            z = {m: 0 for m in nopt}
            for cl, v in zip(H, zc):
                for m in cl: z[m] = v
            hits = [len(cl) for cl, v in zip(H, zc) if v]; wrong = [len(cl) for cl, v in zip(H, zc) if not v]
            lp_part = crp_logp(hits, self.a1) + crp_logp(wrong, self.a0)
            for h in range(K):
                lp = math.log(self.pi[h]) + lp_part
                for m in S:
                    lp += math.log(self.q[m][h] if z[m] else 1 - self.q[m][h])
                for (j, m), b in bins.items():
                    if b is not None: lp += math.log(self.th[(j, z[m], h)][b])
                out.append((h, z, lp))
        return out

    def posterior(self, c, S, judges=None):
        judges = self.J if judges is None else judges
        H, nopt = observed(c, S)
        bins = {(j, m): JB[(c, j, m)] for j in judges for m in S if m not in nopt}
        T = self.joint_terms(c, S, H, nopt, bins)
        L = np.array([t[2] for t in T]); w = np.exp(L - L.max()); w /= w.sum()
        return H, nopt, T, w

    def choose(self, c, S, judges=None):
        H, nopt, T, w = self.posterior(c, S, judges)
        if not H: return None, 0.0
        pc = [sum(wi for (h, z, _), wi in zip(T, w) if z[cl[0]]) for cl in H]
        k = int(np.argmax(pc)); cl = H[k]
        gem = {m: (score(c, GEM, m) or {}).get('p') or 0 for m in cl}
        return PT[(c, max(cl, key=lambda m: gem[m]))], pc[k]

    def fit(self, data, iters=60):
        S = tuple(M)
        for _ in range(iters):
            pi = np.ones(K) * .5; qn = {m: np.ones(K) * .5 for m in M}; qd = {m: np.ones(K) for m in M}
            thc = {k: np.ones(NB) * .5 for k in self.th}; part = []
            for c in data:
                H, nopt, T, w = self.posterior(c, S)
                bins = {(j, m): JB[(c, j, m)] for j in self.J for m in S if m not in nopt}
                for (h, z, _), wi in zip(T, w):
                    pi[h] += wi
                    for m in S: qn[m][h] += wi * z[m]; qd[m][h] += wi
                    for (j, m), b in bins.items():
                        if b is not None: thc[(j, z[m], h)][b] += wi
                    part.append((wi, [len(cl) for cl in H if z[cl[0]]], [len(cl) for cl in H if not z[cl[0]]]))
            self.pi = pi / pi.sum(); self.q = {m: np.clip(qn[m] / qd[m], .01, .995) for m in M}
            self.th = {k: v / v.sum() for k, v in thc.items()}
            grid = np.exp(np.linspace(math.log(.01), math.log(5), 40))
            self.a1 = float(max(grid, key=lambda a: sum(wi * crp_logp(hs, a) for wi, hs, _ in part)))
            self.a0 = float(max(grid, key=lambda a: sum(wi * crp_logp(ws, a) for wi, _, ws in part)))
            order = np.argsort([-np.mean([self.q[m][h] for m in M]) for h in range(K)])   # keep class 0 = easy
            if order[0] != 0:
                self.pi = self.pi[order]; self.q = {m: self.q[m][order] for m in M}
                self.th = {(j, z, h): self.th[(j, z, int(order[h]))] for (j, z, h) in self.th}
        return self

    # ---------- exact policy ----------
    def outcomes(self, c, S, k):
        """predictive distribution of acquiring model k given S: list of (prob, real outcome flag) where the real outcome is
        computed from data; returns dict outcome_key -> prob, outcome_key = (cluster index among H or 'new' or 'nopt', bin)"""
        H, nopt, T, w = self.posterior(c, S, [GEM])
        pred = collections.defaultdict(float)
        for (h, z, _), wi in zip(T, w):
            hits = [i for i, cl in enumerate(H) if z[cl[0]]]; wrong = [i for i, cl in enumerate(H) if not z[cl[0]]]
            nh = sum(len(H[i]) for i in hits); nw = sum(len(H[i]) for i in wrong)
            q = self.q[k][h]
            for zk, pz in ((1, q), (0, 1 - q)):
                idx = hits if zk else wrong; n = nh if zk else nw; a = self.a1 if zk else self.a0
                opts = [(i, len(H[i]) / (n + a)) for i in idx] + [('new', a / (n + a))]
                for o, po in opts:
                    for b in range(NB):
                        pred[(o, b, zk)] += wi * pz * po * self.th[(GEM, zk, h)][b]
        return pred

    def value(self, c, S, H_override, cap, lam, memo):
        """expected objective of the optimal continuation from acquired set S with hypothetical observation state.
        For simplicity the hypothetical state is represented by the real data of S (prefix) when H_override is None."""
        raise NotImplementedError


def stop_value(post_clusters):
    return 1 - (max(post_clusters) if post_clusters else 0.0)


class PolicySim:
    """Exact backward induction with hypothetical observations. A hypothetical state is (S, clusters as lists of models,
    nopt set, gem bins). The model scores it with the same joint law."""

    def __init__(self, J): self.J = J

    def state_post(self, S, H, nopt, bins):
        T = self.J.joint_terms(None, S, H, nopt, bins)
        L = np.array([t[2] for t in T]); w = np.exp(L - L.max()); w /= w.sum()
        pc = [sum(wi for (h, z, _), wi in zip(T, w) if z[cl[0]]) for cl in H]
        return T, w, pc

    def predictive(self, S, H, nopt, bins, k):
        T, w, _ = self.state_post(S, H, nopt, bins); pred = collections.defaultdict(float)
        for (h, z, _), wi in zip(T, w):
            hits = [i for i, cl in enumerate(H) if z[cl[0]]]; wrong = [i for i, cl in enumerate(H) if not z[cl[0]]]
            nh = sum(len(H[i]) for i in hits); nw = sum(len(H[i]) for i in wrong); q = self.J.q[k][h]
            for zk, pz in ((1, q), (0, 1 - q)):
                idx = hits if zk else wrong; n = nh if zk else nw; a = self.J.a1 if zk else self.J.a0
                for o, po in [(i, len(H[i]) / (n + a)) for i in idx] + [('new', a / (n + a))]:
                    for b in range(NB):
                        pred[(o, b)] += wi * pz * po * self.J.th[(GEM, zk, h)][b]
        tot = sum(pred.values())
        return {o: p / tot for o, p in pred.items()}

    @staticmethod
    def extend(S, H, nopt, bins, k, o, b):
        H2 = [list(cl) for cl in H]
        if o == 'new': H2.append([k])
        else: H2[o].append(k)
        b2 = dict(bins); b2[(GEM, k)] = b
        return S + (k,), H2, nopt, b2

    def V(self, S, H, nopt, bins, cap, lam):
        _, _, pc = self.state_post(S, H, nopt, bins)
        best = (stop_value(pc), None)
        if len(S) < cap:
            for k in M:
                if k in S: continue
                ev = lam * ACQ_USD[k]
                for (o, b), p in self.predictive(S, H, nopt, bins, k).items():
                    if p < 1e-9: continue
                    ev += p * self.V(*self.extend(S, H, nopt, bins, k, o, b), cap, lam)[0]
                if ev < best[0] - 1e-12: best = (ev, k)
        return best


def real_state(c, S):
    H, nopt = observed(c, S)
    bins = {(GEM, m): JB[(c, GEM, m)] for m in S if m not in nopt}
    return tuple(S), H, nopt, bins


def run():
    groups = sorted({C[c]['image'] for c in items}); r = np.random.default_rng(7); perm = r.permutation(len(groups))
    fold = {groups[i]: n % 2 for n, i in enumerate(perm)}
    res = {'acq_usd': {m: round(v, 5) for m, v in ACQ_USD.items()}}
    fits = {}
    for jset in ((GEM,), tuple(M)):
        fits[jset] = {f: Joint(list(jset)).fit([c for c in items if fold[C[c]['image']] != f]) for f in (0, 1)}
    J = fits[(GEM,)]
    ex = J[0]; res['example_fit'] = dict(pi=ex.pi.round(3).tolist(), q={m: ex.q[m].round(3).tolist() for m in M}, alpha_hit=round(ex.a1, 3), alpha_wrong=round(ex.a0, 3),
                                          gem_llr_easy=[round(float(np.log(ex.th[(GEM, 1, 0)][b] / ex.th[(GEM, 0, 0)][b])), 2) for b in range(NB)],
                                          gem_llr_hard=[round(float(np.log(ex.th[(GEM, 1, 1)][b] / ex.th[(GEM, 0, 1)][b])), 2) for b in range(NB)])
    print('example fit', res['example_fit'])
    # static errors and calibration / coherence
    for jset in ((GEM,), tuple(M)):
        for S in [(m,) for m in M] + [tuple(x) for x in itertools.combinations(M, 2)] + [tuple(M)]:
            e, pcs, hits = [], [], []
            for c in items:
                md = fits[jset][fold[C[c]['image']]]
                p, pc = md.choose(c, S, list(jset)); h = HIT(p, c); e.append(1 - h); pcs.append(pc); hits.append(h)
            key = f"{'+'.join(jset)}|{'+'.join(S)}"
            res[key] = dict(error=round(100 * np.mean(e), 2), mean_post=round(float(np.mean(pcs)), 3), accuracy=round(float(np.mean(hits)), 3))
            print(f'{key:40s} error {100*np.mean(e):5.2f}%  mean posterior of chosen {np.mean(pcs):.3f}  accuracy {np.mean(hits):.3f}')
    # calibration bins at full state, Gemini judge
    pcs, hits = [], []
    for c in items:
        p, pc = J[fold[C[c]['image']]].choose(c, tuple(M), [GEM]); pcs.append(pc); hits.append(HIT(p, c))
    pcs, hits = np.array(pcs), np.array(hits); cal = []
    for lo, hi in ((0, .6), (.6, .9), (.9, .99), (.99, 1.01)):
        s = (pcs >= lo) & (pcs < hi)
        if s.sum(): cal.append(dict(bin=[lo, hi], n=int(s.sum()), mean_post=round(float(pcs[s].mean()), 3), hit=round(float(hits[s].mean()), 3)))
    res['calibration_full_gem'] = cal; print('calibration', cal)
    # exact policy curve
    curve = []
    for cap in (1, 2, 3):
        for lam in (0.0, 1.0, 3.0, 10.0):
            e, usd, nacq, first = [], [], [], collections.Counter()
            for c in items:
                P = PolicySim(J[fold[C[c]['image']]]); S = ()
                st = real_state(c, S)
                while True:
                    v, k = P.V(*st, cap, lam)
                    if k is None: break
                    if not S: first[k] += 1
                    S = S + (k,); st = real_state(c, S)
                p, _ = J[fold[C[c]['image']]].choose(c, S, [GEM]) if S else (None, 0)
                e.append(1 - HIT(p, c)); usd.append(sum(ACQ_USD[m] for m in S)); nacq.append(len(S))
            row = dict(cap=cap, lam=lam, error=round(100 * np.mean(e), 2), mean_acq=round(float(np.mean(nacq)), 2), calls=round(3 * float(np.mean(nacq)), 2),
                       usd=round(float(np.mean(usd)), 5), first=dict(first))
            curve.append(row); print('policy', row)
    res['policy_curve'] = curve
    json.dump(res, open(HERE / 'joint_model.json', 'w'), indent=1)


if __name__ == '__main__':
    run()
