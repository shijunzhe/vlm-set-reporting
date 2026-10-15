"""Request builders for the accuracy-first pilot (subset main, 260 click items). No model calls.

CMP  fixed-candidate comparison (coauthor's method card): Gemini 3.8 compares two hypothesis clusters on items with >= 2
     hypotheses. Immutable cluster IDs (no model names). Evidence packet: view 'overview' = downscaled full screenshot with both
     candidates marked and labelled by ID; views 'crop_<ID>' = native crops (window 0.2) with the candidate marked. Output:
     preferred_id, discriminating_requirement, evidence_view_ids. Both orders (which candidate is listed and drawn first).
RG   region-first regeneration for models whose first click returned no point (a second identical request also returned no
     point in the pilot): ask for the target element's bounding box on the full view; a later zoom uses its centre.
S2   Gemini 3.8 scores second-click zoomed points (stage Z2) that are new hypotheses (farther than delta from every first
     zoomed point), with the stage-S prompt and crop.
DA   first click C1 of every model with the provider's default reasoning setting (same prompt and image as stage A);
     DB (built after DA) zooms around it as B4. DAP is a 6-request probe (2 items x 3 models) to measure per-call cost.
Usage: python3 -I code/accfirst_stages.py --stage CMP|RG|S2|DA|DAP|DB
"""
import argparse, collections, hashlib, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_stages as F
B, G = F.B, F.G
M = F.M; SUB = 'main'
SYSTEM_CMP = (
    'You are comparing two fixed predictions for the original visual task: a click on a screenshot of a computer application.\n\n'
    'Each candidate has an immutable ID and is marked on the images with its ID label. View "overview" is the full screenshot, '
    'downscaled, with both candidates marked. View "crop_<ID>" is a native-resolution crop around that candidate, with the candidate '
    'marked. Candidate source or model names are not provided.\n\n'
    'Compare the candidates using the actual task requirements:\n'
    '- the requested element or control;\n'
    '- the exact text, state, or attribute when specified;\n'
    '- the requested relation and its anchor or reference frame (panel, toolbar, neighbouring label);\n'
    '- whether the exact candidate point is inside the intended target element.\n\n'
    'Inspect the information that distinguishes the candidates. Do not merely decide whether both regions contain a control. Do not '
    'rewrite any coordinates or invent a new candidate. Cite only supplied view IDs.\n\n'
    'Output contract: one JSON object {"preferred_id": one of the supplied candidate IDs or "undecided", '
    '"discriminating_requirement": "identity" or "text" or "state" or "relation" or "geometry" or "none", '
    '"evidence_view_ids": a list of supplied view IDs}. No prose, no code fence.')
SYSTEM_RG = None


def cid_of(item, hyp):
    return 'K' + hashlib.sha256(f'{item}|{hyp}'.encode()).hexdigest()[:3].upper()


def stage_CMP(C):
    R = []; m = 'gem38'
    for cid, H in F.multi(SUB, C).items():
        c = C[cid]
        for i, j in itertools.combinations(range(len(H)), 2):
            for order, (a, b) in (('ab', (i, j)), ('ba', (j, i))):
                ia, ib = cid_of(cid, a), cid_of(cid, b); pa, pb = H[a][1], H[b][1]
                ov, so = F.render(m, c, f'{SUB}/{m}/{cid}/cmp_overview_h{a}h{b}', marks=((pa, F.RED, ia), (pb, F.BLUE, ib)))
                wa, wb = G.window(pa[0], pa[1], 0.2), G.window(pb[0], pb[1], 0.2)
                ca, sa = F.render(m, c, f'{SUB}/{m}/{cid}/cmp_crop_h{a}_red', crop=wa, marks=((pa, F.RED, ia),))
                cb, sb = F.render(m, c, f'{SUB}/{m}/{cid}/cmp_crop_h{b}_blue', crop=wb, marks=((pb, F.BLUE, ib),))
                user = (G.size_note(B.MODE[m], [so, sa, sb], ['overview', f'crop_{ia}', f'crop_{ib}']) +
                        f'Original task (platform {c["platform"]}): "{c["instruction"]}"\n'
                        f'Candidate IDs: {ia} (red), {ib} (blue).\nViews in order: overview, crop_{ia}, crop_{ib}.\n'
                        'Which candidate satisfies the original task? Follow the output contract exactly.')
                R.append(B.req(f'{cid}|{m}|CMP|h{i}h{j}|{order}', m, [ov, ca, cb], SYSTEM_CMP, user, [so, sa, sb], item=cid, arm='CMP',
                               order=order, ids={ia: a, ib: b}, point_A=pa, point_B=pb))
    return R


def stage_RG(C):
    Areq, A = F.load(SUB, 'A'); R = []
    for cid, c in C.items():
        for m in M:
            if F.first_click(SUB, A, Areq, cid, m) is not None: continue
            mode = B.MODE[m]
            sysm = ('You are a computer-use agent. Given an instruction, find the UI element it refers to in the screenshot and return its '
                    'bounding box. ' + G.coord_sentence(mode, 'IMAGE') + ' If several elements could match, return the most likely one. '
                    'Output contract: one JSON object {"x1": ..., "y1": ..., "x2": ..., "y2": ...}. No prose, no code fence.')
            fv, fs = B.view(m, c['image'], f'{cid}/full')
            R.append(B.req(f'{cid}|{m}|RG', m, [fv], sysm, G.size_note(mode, [fs], ['image']) + f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Return the bounding box of the target element.',
                           [fs], item=cid, arm='RG'))
    return R


def stage_S2(C):
    sys.path.insert(0, '/home/claude/stageB/analysis')
    Z = F.zoomed_points(SUB, C)
    Zq, Zr = F.load(SUB, 'Z2'); Areq, A = F.load(SUB, 'A'); R = []; j = 'gem38'
    for cid, c in C.items():
        if not F.complete(Z, cid): continue
        for k in M:
            r2 = A.get(f'{cid}|{k}|C2')
            if r2 is None: continue
            p1, _ = G.decode_click(r2['raw_text'], B.scale_of(k, Areq[f'{cid}|{k}|C2']['meta']['sizes'][0]))
            if p1 is None: continue
            q = Zq.get(f'{cid}|{k}|B4c2'); rr = Zr.get(f'{cid}|{k}|B4c2'); p = p1
            if q is not None and rr is not None:
                pz, _ = G.decode_click(rr['raw_text'], B.scale_of(k, q['meta']['sizes'][0])); w = q['meta']['window']
                if pz is not None: p = [w[0] + pz[0] * (w[2] - w[0]), w[1] + pz[1] * (w[3] - w[1])]
            if any(Z[(cid, o)] is not None and F.dist(p, Z[(cid, o)], c) <= F.DELTA for o in M): continue
            w = G.window(p[0], p[1], 0.2)
            f, s = B.view(j, c['image'], f'{SUB}/{j}/{cid}/score_Z2_{k}', crop=w, marker=p)
            R.append(B.req(f'{cid}|{j}|S2|{k}', j, [f], F.SYSTEM_S, F.judge_user(c, j, s), [s], item=cid, arm='S2', judge=j, candidate_model=k, window=w, click=p))
    return R


def stage_DA(C, probe=False):
    Areq, _ = F.load(SUB, 'A'); R = []
    ids = sorted(C)[:2] if probe else sorted(C)
    for cid in ids:
        for m in M:
            q = dict(Areq[f'{cid}|{m}|C1']); q['request_id'] = f'{cid}|{m}|C1d'; q['setting'] = 'default'
            q['meta'] = dict(q['meta'], arm='click_default'); R.append(q)
    return R


def stage_DB(C):
    Dq, D = F.load(SUB, 'DA'); R = []
    for cid, c in C.items():
        for m in M:
            k = f'{cid}|{m}|C1d'
            if k not in D: continue
            p, _ = G.decode_click(D[k]['raw_text'], B.scale_of(m, Dq[k]['meta']['sizes'][0]))
            if p is None: continue
            w4 = G.window(p[0], p[1], 1 / 3); f4, s4 = B.view(m, c['image'], f'{SUB}/{m}/{cid}/zoom_default', crop=w4)
            q = B.req(f'{cid}|{m}|B4d', m, [f4], G.prompts(B.MODE[m])['zoom'], F.zoom_user(c, m, s4), [s4], item=cid, arm='B4_default', window=w4, click=p)
            q['setting'] = 'default'; R.append(q)
    return R


def stage_RGZ(C):
    Rq, Rr = F.load(SUB, 'RG'); R = []
    for rid, r in Rr.items():
        q = Rq[rid]; m = q['model_key']; cid = q['meta']['item']; c = C[cid]
        v = G.first_json(r['raw_text']); sc = B.scale_of(m, q['meta']['sizes'][0])
        try:
            x1, y1, x2, y2 = [float(v[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        except Exception:
            continue
        W, H = (1000, 1000) if sc is None else sc
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        if not (0 <= cx <= 1 and 0 <= cy <= 1): continue
        w4 = G.window(cx, cy, 1 / 3); f4, s4 = B.view(m, c['image'], f'{SUB}/{m}/{cid}/zoom_rg', crop=w4)
        R.append(B.req(f'{cid}|{m}|RGZ', m, [f4], G.prompts(B.MODE[m])['zoom'], F.zoom_user(c, m, s4), [s4], item=cid, arm='RGZ', window=w4, click=[cx, cy]))
    return R


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); a = ap.parse_args()
    C = F.items_of(SUB); keep = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}; C = {k: v for k, v in C.items() if k in keep}
    R = dict(CMP=stage_CMP, RG=stage_RG, S2=stage_S2, DA=stage_DA, DAP=lambda C: stage_DA(C, True), DB=stage_DB, RGZ=stage_RGZ)[a.stage](C)
    with open(f'runs/{SUB}/requests_{a.stage}.jsonl', 'w') as f:
        for q in R: f.write(json.dumps(q) + '\n')
    print(a.stage, 'requests', len(R), collections.Counter((q['model_key'], q['meta']['arm'], q['setting']) for q in R))
