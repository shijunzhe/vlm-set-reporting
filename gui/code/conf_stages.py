"""Request builders for the 600-item confirmation run (protocol v3). Run directory runs/conf_v1 (run version 'conf-v1').
No model calls here. Stage order (each built from the previous stages' responses of the SAME run directory):
  A    first click C1, all three models, 'min'                       ids <item>|<model>|C1
  DA   first click, 'default': GPT rep 0, 1, 2 and Sonnet rep 0        ids <item>|<model>|C1d, C1d1, C1d2
  GLF1 GUI-Lens-style crop-first, GPT default                          ids <item>|gpt61|GLF1
  B    zoom B4 around C1 ('min')                                       ids <item>|<model>|B4
  DB   zoom around each default first click ('default')               ids <item>|<model>|B4d, B4d1, B4d2
  GLF2, GLF3                                                           as in gl_stages (at most 3 GLF calls)
  S    Gemini scores of the M candidates;  SD  Gemini scores of the default candidates (GPT rep 0, Sonnet rep 0)
  CMPA_M, CMPA_D   aligned comparison between representatives of the top-3 clusters (decision.py), both orders
Prompts, crops and settings are those of the pilot builders (build_gui, fusion_stages, accfirst_stages, gl_stages,
aligned_stages); each request's meta records run_version, setting, rep and parent request id.
Usage: python3 -I code/conf_stages.py --stage A|DA|GLF1|B|DB|GLF2|GLF3|S|SD|CMPA_M|CMPA_D [--check] [--batch first100|all]
(first100 = seeded random 100 items run through every stage first; then --batch all; the runner skips completed requests)
"""
import argparse, collections, hashlib, itertools, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_stages as F
import accfirst_stages as AF
import gl_stages as GL
import decision as DC
B, G = F.B, F.G
SUB = 'conf_v1'; VER = 'conf-v1'; M = DC.M; GEM = DC.GEM
ROOT = os.getcwd()


def items(batch='all'):
    C = F.items_of('conf')
    if batch == 'first100':                       # seeded random 100 items for the cost and technical check (protocol v3, D)
        import random
        keep = set(random.Random('conf-v1-batch1').sample(sorted(C), 100)); C = {k: v for k, v in C.items() if k in keep}
    return C


def run(): return DC.Run(ROOT, SUB, G, B)


def tag(q, setting, rep, parent=None):
    q['setting'] = setting; q['meta'].update(run_version=VER, setting=setting, rep=rep, parent=parent); return q


def stage_A(C):
    R = []
    for m in M:
        Pm = G.prompts(B.MODE[m])
        for cid, c in C.items():
            fv, fs = B.view(m, c['image'], f'{cid}/full')
            R.append(tag(B.req(f'{cid}|{m}|C1', m, [fv], Pm['click'], G.user_click(c['instruction'], c['platform'], B.MODE[m], fs), [fs], item=cid, arm='click'), 'min', 0))
    return R


def stage_DA(C):
    R = []
    for m, reps in (('gpt61', (0, 1, 2)), ('sonnet55', (0,))):
        Pm = G.prompts(B.MODE[m])
        for cid, c in C.items():
            fv, fs = B.view(m, c['image'], f'{cid}/full')
            for rep in reps:
                R.append(tag(B.req(f'{cid}|{m}|C1d' + (str(rep) if rep else ''), m, [fv], Pm['click'], G.user_click(c['instruction'], c['platform'], B.MODE[m], fs),
                                   [fs], item=cid, arm='click_default'), 'default', rep))
    return R


def zoom_requests(C, first_stage, keys, setting):
    rr = run(); Q, Rs, _ = rr.stage(first_stage); R = []
    for cid, c in C.items():
        for m, first, zoom, rep in keys:
            k = f'{cid}|{m}|{first}'
            if k not in Rs: continue
            p = rr.click(first_stage, k, m)
            if p is None: continue
            w4 = G.window(p[0], p[1], 1 / 3); f4, s4 = B.view(m, c['image'], f'{SUB}/{m}/{cid}/zoom_{zoom}', crop=w4)
            q = B.req(f'{cid}|{m}|{zoom}', m, [f4], G.prompts(B.MODE[m])['zoom'], F.zoom_user(c, m, s4), [s4], item=cid, arm=zoom, window=w4, click=p)
            R.append(tag(q, setting, rep, parent=k))
    return R


def stage_B(C): return zoom_requests(C, 'A', [(m, 'C1', 'B4', 0) for m in M], 'min')


def stage_DB(C):
    return zoom_requests(C, 'DA', [('gpt61', 'C1d', 'B4d', 0), ('gpt61', 'C1d1', 'B4d1', 1), ('gpt61', 'C1d2', 'B4d2', 2), ('sonnet55', 'C1d', 'B4d', 0)], 'default')


def stage_score(C, which):
    rr = run(); R = []; j = GEM
    for cid, c in C.items():
        flags = set(); A = DC.atoms(rr, 'M' if which == 'S' else 'D', cid, flags)
        for a in A:
            if (which == 'S') != (a['setting'] == 'min'): continue
            p = a['point']; w = G.window(p[0], p[1], 0.2)
            f, s = B.view(j, c['image'], f'{SUB}/{j}/{cid}/score_{which}_{a["model"]}', crop=w, marker=p)
            q = B.req(f'{cid}|{j}|{which}|{a["model"]}', j, [f], F.SYSTEM_S, F.judge_user(c, j, s), [s], item=cid, arm=which, judge=j,
                      candidate_model=a['model'], candidate_id=a['id'], window=w, click=p)
            R.append(tag(q, 'min', 0, parent=a['parent']))
    return R


def label(cand, used):
    for n in range(4, 17):
        l = 'C' + cand[:n].upper()
        if l not in used: return l
    raise ValueError('label collision')


def stage_CMPA(C, cfg):
    rr = run(); R = []; m = GEM
    for cid, c in C.items():
        A, sc, cls, flags = DC.structure(rr, cfg, cid, c)
        comp = sorted(DC.compared(sc, cls), key=lambda cl: cl['rank'])
        if len(comp) < 2: continue
        for x, y in itertools.combinations(comp, 2):
            for order, (u, v) in (('ab', (x, y)), ('ba', (y, x))):
                ra, rb = u['rep'], v['rep']; used = set(); la = label(ra['id'], used); used.add(la); lb = label(rb['id'], used)
                pa, pb = ra['point'], rb['point']
                ov, so = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_{cfg}_ov_{ra["id"]}_{rb["id"]}', marks=((pa, F.RED, la), (pb, F.BLUE, lb)))
                wa, wb = G.window(pa[0], pa[1], 0.2), G.window(pb[0], pb[1], 0.2)
                ca, sa = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_{cfg}_crop_{ra["id"]}_red', crop=wa, marks=((pa, F.RED, la),))
                cb, sb = F.render(m, c, f'{SUB}/{m}/{cid}/cmpa_{cfg}_crop_{rb["id"]}_blue', crop=wb, marks=((pb, F.BLUE, lb),))
                user = (G.size_note(B.MODE[m], [so, sa, sb], ['overview', f'crop_{la}', f'crop_{lb}']) +
                        f'Original task (platform {c["platform"]}): "{c["instruction"]}"\n'
                        f'Candidate IDs: {la} (red), {lb} (blue).\nViews in order: overview, crop_{la}, crop_{lb}.\n'
                        'Which candidate satisfies the original task? Follow the output contract exactly.')
                q = B.req(f'{cid}|{m}|CMPA_{cfg}|{ra["id"]}|{rb["id"]}|{order}', m, [ov, ca, cb], AF.SYSTEM_CMP, user, [so, sa, sb], item=cid,
                          arm=f'CMPA_{cfg}', order=order, labels={la: ra['id'], lb: rb['id']}, points={ra['id']: pa, rb['id']: pb},
                          pair=sorted([ra['id'], rb['id']]))
                R.append(tag(q, 'min', 0, parent=[ra['parent'], rb['parent']]))
    return R


def stage_GLF(C, n):
    rr = run(); R = []
    Qp, Rp, _ = rr.stage(f'GLF{n - 1}') if n > 1 else ({}, {}, {})
    for cid, c in C.items():
        if n == 1:
            fv, fs = B.view('gpt61', c['image'], f'{cid}/full')
            q = GL.req(f'{cid}|gpt61|GLF1', c, fv, fs, GL.SYSTEM_GLF1, f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Return the region.', [0, 0, 1, 1], 1)
            R.append(tag(q, 'default', 0)); continue
        k = f'{cid}|gpt61|GLF{n - 1}'
        if k not in Rp: continue
        act, val = GL.decode(Rp[k]['raw_text'], Qp[k]['meta']['frame'])
        if act != 'crop': continue
        region = GL.pad_clip(val); final = n == 3
        img, size = GL.render_region(c, f'{SUB}/gpt61/{cid}/glf{n}', region, 8 if final else 5)
        if final: sysm, user = G.prompts('grid')['zoom'], f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Return the click point in this crop.'
        else: sysm, user = GL.SYSTEM_GL.format(part='an enlarged region of the screenshot'), f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Crop or click?'
        R.append(tag(GL.req(f'{cid}|gpt61|GLF{n}', c, img, size, sysm, user, region, n), 'default', 0, parent=k))
    return R


def check(stage, R, C):
    """dry-run checks: unique ids, items in the confirmation list only, no pilot overlap, images exist, parents in this run,
    prompts identical to the pilot's for the same stage and model"""
    ids = [q['request_id'] for q in R]; assert len(ids) == len(set(ids)), 'duplicate request ids'
    full = F.items_of('conf'); assert len(full) == 600 and len({c['image'] for c in full.values()}) == 600, 'confirmation list is not 600 unique items'
    pilot = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}
    pimg = {c['image'] for c in json.load(open('data/click_items_prepped.json'))}
    its = {q['meta']['item'] for q in R}; assert its <= set(C), 'item outside confirmation list'
    assert not (its & pilot), 'pilot item in confirmation requests'
    assert not ({C[i]['image'] for i in its} & pimg), 'pilot screenshot in confirmation requests'
    for q in R:
        for p in q['images']: assert os.path.exists(p), p
        assert q['meta']['run_version'] == VER
        par = q['meta'].get('parent')
        if par:
            for pp in (par if isinstance(par, list) else [par]):
                assert any(pp in run().stage(st)[1] for st in ('A', 'DA', 'B', 'DB', 'GLF1', 'GLF2')), ('parent not in this run', pp)
    ref = {'A': 'A', 'DA': 'DA', 'B': 'B', 'DB': 'DB', 'S': 'S', 'SD': 'SD', 'CMPA_M': 'CMPA_M', 'CMPA_D': 'CMPA_D', 'GLF1': 'GLF1', 'GLF2': 'GLF2', 'GLF3': 'GLF2'}[stage]
    pil = collections.defaultdict(set)
    if os.path.exists(f'runs/main/requests_{ref}.jsonl'):
        for l in open(f'runs/main/requests_{ref}.jsonl'):
            q = json.loads(l); pil[q['model_key']].add(q['system'])
    same = sum(1 for q in R if q['system'] in pil[q['model_key']]); print(f'{stage}: {len(R)} requests, {len(its)} items, system prompt identical to pilot for {same}/{len(R)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--check', action='store_true')
    ap.add_argument('--batch', default='all', choices=['all', 'first100']); a = ap.parse_args()
    C = items(a.batch); assert len(C) == (600 if a.batch == 'all' else 100)
    fn = dict(A=stage_A, DA=stage_DA, B=stage_B, DB=stage_DB, S=lambda C: stage_score(C, 'S'), SD=lambda C: stage_score(C, 'SD'),
              CMPA_M=lambda C: stage_CMPA(C, 'M'), CMPA_D=lambda C: stage_CMPA(C, 'D'),
              GLF1=lambda C: stage_GLF(C, 1), GLF2=lambda C: stage_GLF(C, 2), GLF3=lambda C: stage_GLF(C, 3))[a.stage]
    R = fn(C); os.makedirs(f'runs/{SUB}', exist_ok=True)
    if a.check: check(a.stage, R, C)
    with open(f'runs/{SUB}/requests_{a.stage}.jsonl', 'w') as f:
        for q in R: f.write(json.dumps(q) + '\n')
    print(a.stage, 'written', len(R), collections.Counter((q['model_key'], q['setting']) for q in R))
