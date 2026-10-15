"""Repair B. Intersection = selection (deletion only) + geometry (coordinate fusion). Propositions 1 to 3 of the paper.
For every base call A0 and partner (A1 same model, or B0 another model), on every image:
  S0 = A0's own records whose cluster is confirmed by the partner (original coordinates); Sf = fused records of confirmed clusters.
  selection = E(A) - E(S0); geometry = E(S0) - E(Sf); total = E(A) - E(Sf).
  identity checks (exact integers): E(A) - E(S0) == (c(A) - c(S0)) - 2 (r(A) - r(S0)); likewise for the geometry term.
  fixed-label proxy (Prop. 4 with no takeover): F_u - T_u, where T_u / F_u are A0 records unconfirmed and matched / unmatched in
  A0's own maximum matching; proxy - selection = takeover effect (a remaining record takes a deleted record's match).
Datasets: E3 (LIST-min) and GUI screen parsing (GroundCUA, ScreenParse). No model calls. Output: out/intersection_decomp.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fsr_core as F
OUT = Path(__file__).parent / 'out'; OUT.mkdir(exist_ok=True)
res = {}


def tally(acc, sA, sS0, sSf, proxy):
    sel = sA['E'] - sS0['E']; geo = sS0['E'] - sSf['E']
    acc['sel'] += sel; acc['geo'] += geo; acc['tot'] += sA['E'] - sSf['E']; acc['proxy'] += proxy
    acc['dc'] += sA['c'] - sS0['c']; acc['dr'] += sA['r'] - sS0['r']
    acc['id_sel_ok'] += int(sel == (sA['c'] - sS0['c']) - 2 * (sA['r'] - sS0['r']))
    acc['id_geo_ok'] += int(geo == (sS0['c'] - sSf['c']) - 2 * (sS0['r'] - sSf['r']))
    acc['n'] += 1


def summarize(acc):
    n = acc['n']; dc, dr = acc['dc'], acc['dr']
    return dict(images=n, selection=round(acc['sel'] / n, 4), geometry=round(acc['geo'] / n, 4), total=round(acc['tot'] / n, 4),
                fixed_label_proxy=round(acc['proxy'] / n, 4), takeover=round((acc['proxy'] - acc['sel']) / n, 4),
                deleted_scored_per_img=round(dc / n, 3), matches_lost_per_img=round(dr / n, 3), p1_marginal=round(dr / dc, 4) if dc else None,
                theorem1_predicts_gain=bool(dc and dr / dc < 0.5), selection_gain_positive=bool(acc['sel'] > 0),
                identity_selection_exact=f"{acc['id_sel_ok']}/{n}", identity_geometry_exact=f"{acc['id_geo_ok']}/{n}")


# ---------------- E3
d = F.E3Data()
for ds in d.datasets:
    ids = d.images(ds)
    for a in F.MODELS:
        for b in F.MODELS:
            acc = dict(sel=0, geo=0, tot=0, proxy=0, dc=0, dr=0, id_sel_ok=0, id_geo_ok=0, n=0)
            for i in ids:
                A = d.reps(ds, a, i); base = A[0]; part = A[1] if a == b else d.reps(ds, b, i)[0]
                cl = d.clusters(i, [base, part])
                own = [x for x in cl if 0 in x[2]]
                S0 = [x[1][0] for x in own if len(x[2]) == 2]; Sf = [d.fuse(x[1]) for x in own if len(x[2]) == 2]
                sA, sS0, sSf = d.score(i, base), d.score(i, S0), d.score(i, Sf)
                # fixed labels from A0's own maximum matching
                recA = [x[1][0] for x in own]; conf = np.array([len(x[2]) == 2 for x in own], bool)
                it = d.items[i]; mt = d.metric(i)
                if mt == 'point':
                    pr = d.L.match_points(d.L._pts(recA, it['labels']), [((t['point'][0] if 'point' in t else (t['box'][0] + t['box'][2]) / 2) + d.L.LABEL_OFFSET * it['labels'].index(t['label']), (t['point'][1] if 'point' in t else (t['box'][1] + t['box'][3]) / 2)) for t in it['targets']], it['tau_point']) if recA and it['targets'] else []
                else:
                    pr = d.L.match_boxes(recA, [(t['box'], t['label']) for t in it['targets']], mt[1]) if recA and it['targets'] else []
                matched = np.zeros(len(recA), bool)
                for p, _ in pr: matched[p] = True
                ign = np.array([d.L.in_ignore(r, it['ignore']) for r in recA], bool) if recA else np.zeros(0, bool)
                Tu = int((matched & ~conf).sum()); Fu = int((~matched & ~conf & ~ign).sum())
                tally(acc, sA, sS0, sSf, Fu - Tu)
            res[f'E3|{ds}|{a}|{"same" if a == b else b}'] = summarize(acc)
            print(ds, a, b, res[f'E3|{ds}|{a}|{"same" if a == b else b}'], flush=True)
# ---------------- GUI parse
g = F.GUIData()
for src in ('groundcua', 'screenparse'):
    ids = g.images(src)
    for a in F.MODELS:
        for b in F.MODELS:
            acc = dict(sel=0, geo=0, tot=0, proxy=0, dc=0, dr=0, id_sel_ok=0, id_geo_ok=0, n=0)
            for i in ids:
                base = g.draft(i, a, 1); part = g.draft(i, a, 2) if a == b else g.draft(i, b, 1)
                cl = [c for c in g.F.clusters([base, part]) if 0 in c['sup']]
                S0 = [c['members'][0] for c in cl if len(c['sup']) == 2]; Sf = [list(c['fused']) for c in cl if len(c['sup']) == 2]
                bA = [x[0] for x in base]
                sA, sS0, sSf = g.score(i, bA), g.score(i, S0), g.score(i, Sf)
                recA = [c['members'][0] for c in cl]; conf = np.array([len(c['sup']) == 2 for c in cl], bool)
                pr = g.F.lex_match(recA, g.items[i]['targets'], 0.5) if recA and g.items[i]['targets'] else []
                matched = np.zeros(len(recA), bool)
                for p, _ in pr: matched[p] = True
                tally(acc, sA, sS0, sSf, int((~matched & ~conf).sum()) - int((matched & ~conf).sum()))
            res[f'GUI|{src}|{a}|{"same" if a == b else b}'] = summarize(acc)
            print(src, a, b, res[f'GUI|{src}|{a}|{"same" if a == b else b}'], flush=True)
vals = list(res.values())
res['_summary'] = dict(
    pairs=len(vals), identity_selection_all_exact=all(v['identity_selection_exact'].split('/')[0] == v['identity_selection_exact'].split('/')[1] for v in vals),
    identity_geometry_all_exact=all(v['identity_geometry_exact'].split('/')[0] == v['identity_geometry_exact'].split('/')[1] for v in vals),
    theorem1_vs_deletion_sign=f"{sum(v['theorem1_predicts_gain'] == v['selection_gain_positive'] for v in vals)}/{len(vals)}",
    proxy_sign_vs_deletion_sign=f"{sum((v['fixed_label_proxy'] > 0) == (v['selection_gain_positive']) for v in vals)}/{len(vals)}",
    proxy_sign_vs_fused_total_sign=f"{sum((v['fixed_label_proxy'] > 0) == (v['total'] > 0) for v in vals)}/{len(vals)}",
    geometry_negative_pairs=sum(v['geometry'] < 0 for v in vals), geometry_positive_pairs=sum(v['geometry'] > 0 for v in vals),
    sign_flips_due_to_geometry=[k for k, v in res.items() if not k.startswith('_') and (v['selection'] > 0) != (v['total'] > 0)])
json.dump(res, open(OUT / 'intersection_decomp.json', 'w'), indent=1)
print(json.dumps(res['_summary'], indent=1))
