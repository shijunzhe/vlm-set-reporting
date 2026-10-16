"""Coverage on the 600-item confirmation (descriptive; no model calls): how many items have no correct candidate when the
candidate menu is (a) one GPT-6.1 default zoomed point, (b) three independent GPT-6.1 default zoomed points (the GPTx3 menu),
(c) the D menu (GPT-6.1 default, Sonnet 5.5 default, Gemini 3.8 lowest), and the conditional rate at which an added source
hits when every earlier candidate missed (rho in Proposition coverage). Output: accfirst/conf_coverage_<subset>.json"""
import argparse, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = '/home/claude'; GUI = BASE + '/gui'; sys.path.insert(0, GUI + '/code')
cwd = os.getcwd(); import fusion_stages as F; import decision as DC; os.chdir(cwd)
G, B = F.G, F.B
ap = argparse.ArgumentParser(); ap.add_argument('--subset', default='conf_v1'); ap.add_argument('--items', default='conf'); a = ap.parse_args()
# --subset main --items main: the 260 click items of the GUI experiment (no repeated GPT-6.1 zooms were run there)
os.chdir(GUI); C = F.items_of(a.items); items = sorted(C); run = DC.Run(GUI, a.subset, G, B)
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))
def z(c, m, setting, first, zoom): f = set(); p, _ = DC.zoomed(run, c, m, setting, first, zoom, f); return hit(p, c)
H = {c: dict(g0=z(c, 'gpt61', 'default', 'C1d', 'B4d'), g1=z(c, 'gpt61', 'default', 'C1d1', 'B4d1'), g2=z(c, 'gpt61', 'default', 'C1d2', 'B4d2'),
             s=z(c, 'sonnet55', 'default', 'C1d', 'B4d'), m=z(c, 'gem38', 'min', 'C1', 'B4')) for c in items}
def F_(keys): return sum(1 for c in items if not any(H[c][k] for k in keys))
def rho(new, old):
    base = [c for c in items if not any(H[c][k] for k in old)]
    return dict(n_all_old_wrong=len(base), new_hits=sum(H[c][new] for c in base), rho=round(sum(H[c][new] for c in base) / len(base), 3))
REP = any(H[c]['g1'] for c in items)  # repeated GPT-6.1 zooms exist only for the 600-item set
Fk = [('GPT', ['g0'])] + ([('GPT x2', ['g0', 'g1']), ('GPT x3', ['g0', 'g1', 'g2'])] if REP else []) + \
     [('GPT+Sonnet', ['g0', 's']), ('GPT+Gemini', ['g0', 'm']), ('D (GPT+Sonnet+Gemini)', ['g0', 's', 'm'])]
Rk = ([('GPT repeat | GPT wrong', ('g1', ['g0']))] if REP else []) + [('Sonnet | GPT wrong', ('s', ['g0'])), ('Gemini | GPT wrong', ('m', ['g0']))] + \
     ([('third GPT | two GPT wrong', ('g2', ['g0', 'g1']))] if REP else []) + [('Gemini | GPT and Sonnet wrong', ('m', ['g0', 's']))]
res = dict(F={k: F_(v) for k, v in Fk}, rho={k: rho(*v) for k, v in Rk})
# identity check: F_new = F_old (1 - rho)
for new, old, k_new, k_old in ((('g1', ['g0'], 'GPT x2', 'GPT'),) if REP else ()) + (('s', ['g0'], 'GPT+Sonnet', 'GPT'), ('m', ['g0', 's'], 'D (GPT+Sonnet+Gemini)', 'GPT+Sonnet')):
    r = rho(new, old); assert res['F'][k_new] == r['n_all_old_wrong'] - r['new_hits']
json.dump(res, open(HERE / f'conf_coverage_{a.subset}.json', 'w'), indent=1); print(json.dumps(res, indent=1))
