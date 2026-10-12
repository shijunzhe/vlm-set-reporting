"""Does adding cross-model constructions to the few-label menu (method step 5) help? E3, LIST-min, per-image errors from
xmodel_e3.json. Menu A: any model x {single, same2_inter, same2_union, same3_unan, same3_maj, same3_union} (<= 3 calls of one model).
Menu B: menu A + cross-model {cross2_inter / cross2_union for every ordered pair, cross3_unan, cross3_maj, cross3_union} (<= 3 calls).
L labelled images chosen at random (200 draws), option with the lowest mean error on them, scored on the other images.
Reported relative to the menu-A choice on the same draw, and the share of draws in which a cross option is picked.
usage: python -I xmodel_menu.py"""
import json, random
import numpy as np
d = json.load(open('/home/claude/stageB/xmodel/xmodel_e3.json'))
M = ['gpt61', 'sonnet55', 'gem38']; out = {}
for ds, E in d['per_image'].items():
    ids = sorted(E['single|gpt61'])
    A = [f'{k}|{m}' for m in M for k in ('single', 'same2_inter', 'same2_union', 'same3_unan', 'same3_maj', 'same3_union')]
    B = A + [k for k in E if k.startswith('cross2_')] + ['cross3_unan', 'cross3_maj', 'cross3_union']
    full = {k: np.mean([E[k][i] for i in ids]) for k in B}
    res = {}
    for L in (3, 5, 10):
        rng = random.Random(L); rel = []; pick_cross = 0; regret_A = []; regret_B = []
        for _ in range(200):
            lab = rng.sample(ids, L); rest = [i for i in ids if i not in lab]
            choose = lambda opts: min(opts, key=lambda k: np.mean([E[k][i] for i in lab]))
            a = choose(A); b = choose(B); pick_cross += not b in A
            ea = np.mean([E[a][i] for i in rest]); eb = np.mean([E[b][i] for i in rest])
            rel.append((eb - ea) / ea)
        res[f'L{L}'] = dict(menuB_vs_menuA_median=round(float(np.median(rel)) * 100, 2), menuB_vs_menuA_mean=round(float(np.mean(rel)) * 100, 2),
                            worse_share=round(float(np.mean(np.array(rel) > 0)), 3), cross_picked_share=round(pick_cross / 200, 3))
    bestA = min(A, key=full.get); bestB = min(B, key=full.get)
    res['best_fixed_A'] = [bestA, round(float(full[bestA]), 2)]; res['best_fixed_B'] = [bestB, round(float(full[bestB]), 2)]
    res['best_fixed_B_vs_A'] = round(float((full[bestB] - full[bestA]) / full[bestA]) * 100, 2)
    out[ds] = res; print(ds, res)
json.dump(out, open('/home/claude/stageB/xmodel/xmodel_menu.json', 'w'), indent=1)
