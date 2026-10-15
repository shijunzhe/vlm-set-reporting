"""Unit checks for decision.py rules that the pilot cannot exercise (GPT x3, tie rule, file order invariance). No calls."""
import itertools, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision as DC
c = dict(width=1000, height=1000)


class FakeRun:
    """points per (first, zoom) key; status per key"""
    def __init__(self, pts): self.pts = pts
    def status(self, st, rid): return 'completed' if rid in self.pts else 'failed'
    def click(self, st, rid, m): return self.pts[rid]
    def stage(self, st): return ({k: {'meta': {'window': [0, 0, 1, 1]}} for k in self.pts}, {}, {})


def gx3(p0, p1, p2):
    pts = {}
    for i, (f, z) in enumerate((('C1d', 'B4d'), ('C1d1', 'B4d1'), ('C1d2', 'B4d2'))):
        p = (p0, p1, p2)[i]
        if p is not None: pts[f'x|gpt61|{f}'] = p; pts[f'x|gpt61|{z}'] = p
    return DC.gpt_x3(FakeRun(pts), 'x', c)[0]


# majority wins and returns the earliest valid rep of the winning cluster
assert gx3([0.1, 0.1], [0.5, 0.5], [0.505, 0.5]) == [0.5, 0.5]
# all different: tie -> cluster with earliest rep (rep 0)
assert gx3([0.1, 0.1], [0.5, 0.5], [0.9, 0.9]) == [0.1, 0.1]
# rep 0 missing: earliest valid is rep 1
assert gx3(None, [0.5, 0.5], [0.9, 0.9]) == [0.5, 0.5]
# all missing -> None
assert gx3(None, None, None) is None
# tie rule: among clusters with most wins, S key decides; independent of order of evaluation
sc = {'a': 0.9, 'b': 0.6, 'c': 0.7}
cls = [dict(rep={'id': k}, size=s, rank=r) for k, s, r in (('a', 1, 0), ('b', 2, 1), ('c', 1, 2))]
for perm in itertools.permutations(cls):
    best = [cl for cl in perm if cl['rep']['id'] in ('b', 'c')]
    assert max(best, key=lambda cl: DC.s_key(sc, cl))['rep']['id'] == 'c'
print('decision_test: all checks passed')
