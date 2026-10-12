"""B3: comparator go / no-go (no model calls here).
Part 1 (click): ScreenSpot-Pro items where the single click (C1) and the two-stage zoom point (B4) differ. The model sees the
clean full screenshot plus two anonymous panels (native crops, window 0.2, a red marker at each candidate) and chooses FIRST,
SECOND or NEITHER. Both orders are asked.
Part 2 (blocks): E3 SKU-110K / DOTA, L-min three calls, clusters supported by 1 or 2 calls (the uncertain ones). Two
anonymous panels of the same native crop: the confirmed boxes of the same class, with vs without the candidate box, all drawn
alike. Both orders are asked. Truth is used only in scoring.
Usage: python -I build_b3.py [--pilot]"""
import json, os, sys, math, random, argparse
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); A = ap.parse_args()
R = C.ROOT; rng = random.Random(33)
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
GUI = '/home/claude/gui'
reqs = []; meta = {'click': {}, 'block': {}}
# ---------------- part 1: click
CI = {c['id']: c for c in json.load(open(f'{GUI}/data/click_items_prepped.json'))}
def loadg(stage):
    Q, Rr = {}, {}
    for l in open(f'{GUI}/runs/main/requests_{stage}.jsonl'): q = json.loads(l); Q[q['request_id']] = q
    for m in C.MODELS:
        for l in open(f'{GUI}/runs/main/responses_{stage}_{m}.jsonl'):
            r = json.loads(l)
            if r['status'] == 'completed': Rr[r['request_id']] = r
    return Q, Rr
QA, RA = loadg('A'); QB, RB = loadg('B')
def sc(q): return None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
CL_SYS = {}
for mode in ('grid', 'pixel'):
    CL_SYS[mode] = ('You are checking where to click in a desktop screenshot. Image 1 is the full screenshot. Images 2 and 3 are two '
                    'close-up panels of parts of the same screenshot; each has one red circle marking a candidate click point. '
                    'Decide which marked point performs the instruction. Answer with ONLY a JSON object {"choice": "FIRST"} if the '
                    'point in image 2 is right, {"choice": "SECOND"} if the point in image 3 is right, or {"choice": "NEITHER"} if '
                    'neither is right. Respond with only the JSON object, no explanation.')
os.makedirs(f'{R}/data/b3click', exist_ok=True)
for m in C.MODELS:
    cand = []
    for cid, c in CI.items():
        if c['source'] != 'sspro': continue
        k = lambda s: f'{cid}|{m}|{s}'
        if k('C1') not in RA or k('B4') not in RB: continue
        p1, _ = G.decode_click(RA[k('C1')]['raw_text'], sc(QA[k('C1')]))
        q4 = QB[k('B4')]; pt, _ = G.decode_click(RB[k('B4')]['raw_text'], sc(q4)); w = q4['meta']['window']
        if p1 is None or pt is None: continue
        p4 = [w[0] + pt[0] * (w[2] - w[0]), w[1] + pt[1] * (w[3] - w[1])]
        if math.hypot((p1[0] - p4[0]) * c['width'], (p1[1] - p4[1]) * c['height']) / max(c['width'], c['height']) <= 0.01: continue
        cand.append((cid, p1, p4))
    rng.shuffle(cand); cand = cand[:(100 if not A.pilot else 3)]
    for cid, p1, p4 in cand:
        c = CI[cid]; full = Image.open(f'{GUI}/{c["image"]}').convert('RGB')
        tag = 'a' if m == 'sonnet55' else 'g'; fullp = f'data/views2/{cid}/full_{tag}.png'
        panels = {}
        for name, p in (('c1', p1), ('b4', p4)):
            wdw = G.window(p[0], p[1], 0.2); crop = G.crop_norm(full, wdw); crop = G.draw_marker(crop, p[0], p[1], frame=wdw)
            crop = C.prepare(crop, m); path = f'data/b3click/{cid}_{m}_{name}.png'; crop.save(f'{R}/{path}'); panels[name] = path
        # copy full view into stageB tree
        dst = f'data/b3click/{cid}_{tag}_full.png'
        if not os.path.exists(f'{R}/{dst}'): Image.open(f'{GUI}/{fullp}').save(f'{R}/{dst}')
        meta['click'][f'{cid}|{m}'] = dict(p1=p1, p4=p4, bbox=c['bbox_norm'])
        for order in ('c1b4', 'b4c1'):
            a, b = order[:2], order[2:]
            reqs.append(dict(request_id=f'click|{cid}|{m}|{order}', model_key=m, setting='min', images=[dst, panels[a], panels[b]],
                             system=CL_SYS[C.MODE[m]], user=f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Which marked point should be clicked?',
                             meta=dict(task='click', item=cid, order=order)))
# ---------------- part 2: uncertain clusters (E3 SKU / DOTA)
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import e3lib as L, analyze_e3 as A0, offline_agg2, aggregate_v2 as AG
from score_e3 import load_items
sys.path.insert(0, '/home/claude/fsr_v51/model512'); from stageA_e3_rel import relation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl')
cell = {}
for x in rows:
    if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min' and x['dataset'] in ('sku110k', 'dota'): cell.setdefault((x['dataset'], x['model']), {})[(x['item'], x['rep'])] = x
BL_SYS = ('You compare two annotations of the same image region. Both panels show the same crop with boxes drawn in green. The two '
          'panels differ only in their boxes. Many objects may be unmarked in BOTH panels; ignore unmarked objects, they count the '
          'same for both panels. Judge only the boxes that differ between the panels: a box is good if it tightly covers exactly one '
          'object of the stated class (not a duplicate of another box, not background, not part of an object, not several objects). '
          'If the differing box is good, the panel that has it is better; if it is bad, the panel without it is better. '
          'Respond with ONLY a JSON object, no explanation: {"choice": "FIRST"}, {"choice": "SECOND"} or {"choice": "TIE"}.')
os.makedirs(f'{R}/data/b3block', exist_ok=True)
for (ds, m), c in sorted(cell.items()):
    pool = []
    for i in sorted({k[0] for k in c}):
        if not all((i, r) in c for r in (0, 1, 2)): continue
        it = items[i]; mt = A0.metric_of(it, A0.primary(it))
        reps = [[r for r in A0.recs_of(c[(i, k)]) if r['box'] is not None] for k in (0, 1, 2)]
        cl = AG.build_clusters(it, reps); fused = [AG.fuse(x[1]) for x in cl]
        Rm = relation(it, fused, mt)
        mm = maximum_bipartite_matching(csr_matrix(Rm.astype(np.int8)), perm_type='row') if Rm.size and Rm.any() else np.full(len(it['targets']), -1)
        matched = set(int(v) for v in mm if v >= 0)
        for ci_, x in enumerate(cl):
            if len(x[2]) in (1, 2): pool.append((i, ci_, len(x[2]), int(ci_ in matched)))
    rng.shuffle(pool); pick = pool[:(150 if not A.pilot else 3)]
    for (i, ci_, sup, tp) in pick:
        it = items[i]; reps = [[r for r in A0.recs_of(c[(i, k)]) if r['box'] is not None] for k in (0, 1, 2)]
        cl = AG.build_clusters(it, reps); fused = [AG.fuse(x[1]) for x in cl]; cb = fused[ci_]['box']; lab = fused[ci_]['label']
        img = Image.open(f'{E3}/{it["image_path"]}').convert('RGB'); W, H = img.size
        bw, bh = cb[2] - cb[0], cb[3] - cb[1]; side = max(160, 3 * max(bw, bh))
        cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
        x0 = int(max(0, min(W - side, cx - side / 2))); y0 = int(max(0, min(H - side, cy - side / 2))); x1 = int(min(W, x0 + side)); y1 = int(min(H, y0 + side))
        base = [f['box'] for k_, (f, x) in enumerate(zip(fused, cl)) if k_ != ci_ and len(x[2]) == 3 and f['label'] == lab and f['box'][2] > x0 and f['box'][0] < x1 and f['box'][3] > y0 and f['box'][1] < y1]
        def panel(with_c):
            p = img.crop((x0, y0, x1, y1)).copy(); d = ImageDraw.Draw(p); lw = max(2, round((x1 - x0) / 250))
            for b in base + ([cb] if with_c else []): d.rectangle([b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0], outline=(0, 230, 0), width=lw)
            return C.prepare(p, m)
        pk = f'data/b3block/{ds}_{i}_{m}_{ci_}_keep.png'; pd = f'data/b3block/{ds}_{i}_{m}_{ci_}_drop.png'
        panel(True).save(f'{R}/{pk}'); panel(False).save(f'{R}/{pd}')
        bid = f'{ds}|{i}|{m}|{ci_}'; meta['block'][bid] = dict(support=sup, tp=tp, label=lab, n_base=len(base))
        lname = lab if lab not in ('product', 'object') else 'retail product (one box per product facing)'
        for order in ('keepdrop', 'dropkeep'):
            ims = [pk, pd] if order == 'keepdrop' else [pd, pk]
            reqs.append(dict(request_id=f'block|{bid}|{order}', model_key=m, setting='min', images=ims, system=BL_SYS,
                             user=f'Class: "{lname}". Image 1 is panel FIRST, image 2 is panel SECOND. Which panel is better?',
                             meta=dict(task='block', block=bid, order=order, dataset=ds)))
sub = 'b3pilot' if A.pilot else 'b3'
C.write_manifest(f'{R}/runs/{sub}/requests_main.jsonl', reqs); json.dump(meta, open(f'{R}/runs/{sub}/items.json', 'w'))
from collections import Counter
print(sub, len(reqs), Counter((q['meta']['task'], q['model_key']) for q in reqs))
